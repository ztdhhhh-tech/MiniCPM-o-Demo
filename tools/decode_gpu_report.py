"""Inspect decode CUDA-event traces emitted by duplex latency logs.

Usage:
    python tools/decode_gpu_report.py logs/duplex_latency/session.jsonl
    python tools/decode_gpu_report.py session.jsonl --json --limit 50
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Iterable


def _values(items: Iterable[dict[str, Any]], key: str) -> list[float]:
    return [float(item[key]) for item in items if item.get(key) is not None]


def _stat(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "mean_ms": None}
    ordered = sorted(values)

    def percentile(p: float) -> float:
        index = min(len(ordered) - 1, int(round((len(ordered) - 1) * p)))
        return round(ordered[index], 3)

    return {
        "count": len(values),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "mean_ms": round(statistics.fmean(values), 3),
    }


def _flatten(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    units: list[dict[str, Any]] = []
    tokens: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("event") != "backend.unit.done":
            continue
        unit_key = (record.get("session_id"), record.get("trace_id") or record.get("unit_id"))
        if unit_key in seen:
            continue
        seen.add(unit_key)

        latency = (record.get("metrics") or {}).get("latency") or {}
        unit_tokens = (latency.get("metrics") or {}).get("token_timings") or []
        token_rows: list[dict[str, Any]] = []
        for index, token in enumerate(unit_tokens):
            row = {
                "session_id": record.get("session_id"),
                "unit_id": record.get("unit_id"),
                "trace_id": record.get("trace_id"),
                **token,
            }
            row.setdefault("token_index", index)
            token_rows.append(row)
            tokens.append(row)

        units.append(
            {
                "session_id": record.get("session_id"),
                "unit_id": record.get("unit_id"),
                "trace_id": record.get("trace_id"),
                "token_count": len(token_rows),
                "decode_ms": _stat(_values(token_rows, "decode_ms")),
                "decode_gpu_ms": _stat(_values(token_rows, "decode_gpu_ms")),
                "decode_prev_feed_wait_ms": _stat(
                    _values(token_rows, "decode_prev_feed_wait_ms")
                ),
                "feed_gpu_ms": _stat(_values(token_rows, "feed_gpu_ms")),
            }
        )

    return units, tokens


def _span_stats(tokens: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    values: dict[str, list[float]] = {}
    for token in tokens:
        for name, value in (token.get("decode_gpu_spans") or {}).items():
            if value is not None:
                values.setdefault(name, []).append(float(value))
    return {name: _stat(span_values) for name, span_values in sorted(values.items())}


def _relations(tokens: list[dict[str, Any]]) -> dict[str, Any]:
    both: list[float] = []
    feed_values: list[float] = []
    wait_values: list[float] = []
    ratios: list[float] = []
    waited = 0

    for current, following in zip(tokens, tokens[1:]):
        if current.get("session_id") != following.get("session_id"):
            continue
        feed_gpu = current.get("feed_gpu_ms")
        wait = following.get("decode_prev_feed_wait_ms")
        if feed_gpu is not None:
            feed_values.append(float(feed_gpu))
        if wait is not None:
            wait_values.append(float(wait))
        if feed_gpu is not None and wait is not None:
            feed_gpu = float(feed_gpu)
            wait = float(wait)
            both.append(feed_gpu - wait)
            if feed_gpu > 0:
                ratios.append(wait / feed_gpu)
            if wait > 0:
                waited += 1

    return {
        "candidate_pairs": len(feed_values),
        "waited_pairs": waited,
        "feed_gpu_ms": _stat(feed_values),
        "next_decode_wait_ms": _stat(wait_values),
        "feed_gpu_minus_wait_ms": _stat(both),
        "wait_over_feed_ratio": _stat(ratios),
    }


def aggregate(path: Path) -> dict[str, Any]:
    units, tokens = flatten_tokens(path)
    deviation = [
        float(token["decode_ms"]) - float(token["decode_gpu_ms"])
        for token in tokens
        if token.get("decode_ms") is not None and token.get("decode_gpu_ms") is not None
    ]
    return {
        "unit_table": units,
        "token_table": tokens,
        "decode_gpu_spans": _span_stats(tokens),
        "decode_wall_vs_gpu": {
            **_stat(deviation),
            "definition": "decode_ms - decode_gpu_ms; includes CPU launch, Python, and sync wait",
        },
        "feed_to_next_decode_wait": _relations(tokens),
    }


def flatten_tokens(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return _flatten(path)


def _render_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> None:
    if not rows:
        print("(no rows)")
        return
    rendered: list[list[str]] = []
    for row in rows:
        values = []
        for _, key in columns:
            value = row.get(key)
            if value is None:
                values.append("-")
            elif isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        rendered.append(values)
    widths = [
        max(len(title), *(len(row[index]) for row in rendered))
        for index, (title, _) in enumerate(columns)
    ]
    print("  ".join(title.ljust(widths[index]) for index, (title, _) in enumerate(columns)))
    print("  ".join("-" * width for width in widths))
    for row in rendered:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def _print_report(report: dict[str, Any], token_limit: int) -> None:
    unit_columns = [
        ("trace", "trace_id"),
        ("tokens", "token_count"),
        ("decode p50", "decode_p50_ms"),
        ("gpu p50", "gpu_p50_ms"),
        ("wait p50", "wait_p50_ms"),
    ]
    table_rows = []
    for unit in report["unit_table"]:
        table_rows.append(
            {
                "trace_id": unit.get("trace_id"),
                "token_count": unit.get("token_count"),
                "decode_p50_ms": unit["decode_ms"].get("p50_ms"),
                "gpu_p50_ms": unit["decode_gpu_ms"].get("p50_ms"),
                "wait_p50_ms": unit["decode_prev_feed_wait_ms"].get("p50_ms"),
            }
        )
    print("Units")
    _render_table(table_rows, unit_columns)

    token_columns = [
        ("trace", "trace_id"),
        ("idx", "token_index"),
        ("id", "token_id"),
        ("type", "token_type"),
        ("wall", "decode_ms"),
        ("gpu", "decode_gpu_ms"),
        ("wait", "decode_prev_feed_wait_ms"),
        ("feed gpu", "feed_gpu_ms"),
    ]
    print("\nTokens")
    _render_table(report["token_table"][:token_limit], token_columns)

    print("\nDecode GPU spans")
    span_rows = [
        {"name": name, **stats}
        for name, stats in report["decode_gpu_spans"].items()
    ]
    _render_table(
        span_rows,
        [("span", "name"), ("count", "count"), ("p50", "p50_ms"), ("p95", "p95_ms"), ("mean", "mean_ms")],
    )

    print("\nWall vs GPU")
    print(json.dumps(report["decode_wall_vs_gpu"], ensure_ascii=False))
    print("\nFeed -> next decode wait")
    print(json.dumps(report["feed_to_next_decode_wait"], ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--json", action="store_true", help="emit the full report as JSON")
    parser.add_argument("--limit", type=int, default=20, help="token rows shown in table mode")
    args = parser.parse_args()
    report = aggregate(args.log)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_report(report, max(0, args.limit))


if __name__ == "__main__":
    main()
