"""Aggregate MiniCPM-o duplex JSONL latency logs.

Usage:
    python tools/duplex_latency_report.py logs/duplex_latency/session.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, int(round((len(values) - 1) * p)))
    return round(values[index], 3)


def aggregate(path: Path) -> dict[str, Any]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    span_values: dict[str, list[float]] = {}
    token_values: dict[str, list[float]] = {}
    token_total_values: list[float] = []
    first_token_values: list[float] = []
    loop_overhead_values: list[float] = []
    supply: list[float] = []
    gaps: list[float] = []
    for record in records:
        metrics = record.get("metrics") or {}
        latency = metrics.get("latency") or {}
        previous_finalize = metrics.get("last_finalize_latency") or {}
        latency_metrics = latency.get("metrics") or {}
        token_timings = latency_metrics.get("token_timings") or []
        for index, token in enumerate(token_timings):
            for key in ("decode_ms", "item_sync_ms", "tokenizer_ms", "tokenizer_char_check_ms", "feed_wall_ms", "feed_gpu_ms", "token_total_ms"):
                value = token.get(key)
                if value is not None:
                    token_values.setdefault(key, []).append(float(value))
            if token.get("token_total_ms") is not None:
                token_total_values.append(float(token["token_total_ms"]))
                if index == 0:
                    first_token_values.append(float(token["token_total_ms"]))
        if latency_metrics.get("llm_loop_overhead_ms") is not None:
            loop_overhead_values.append(float(latency_metrics["llm_loop_overhead_ms"]))
        pcm_ms = metrics.get("pcm_duration_ms")
        interval_ms = metrics.get("output_interval_ms")
        if pcm_ms is not None and interval_ms and interval_ms > 0:
            supply.append(float(pcm_ms) / float(interval_ms))
        for span in record.get("spans") or []:
            duration = span.get("duration_ms")
            if duration is not None:
                span_values.setdefault(span.get("name", "unknown"), []).append(float(duration))
        for span in latency.get("spans") or []:
            duration = span.get("duration_ms")
            if duration is not None:
                span_values.setdefault(span.get("name", "unknown"), []).append(float(duration))
        for span in previous_finalize.get("spans") or []:
            duration = span.get("duration_ms")
            if duration is not None:
                span_values.setdefault(span.get("name", "unknown"), []).append(float(duration))
        if record.get("event") == "transport.ws_send" and record.get("duration_ms") is not None:
            span_values.setdefault("transport.ws_send", []).append(float(record["duration_ms"]))
        if record.get("event") == "client.audio.gap" and record.get("gap_ms") is not None:
            gaps.append(float(record["gap_ms"]))

    summary = {
        "records": len(records),
        "components": {
            name: {
                "count": len(values),
                "p50_ms": _percentile(values, 0.50),
                "p90_ms": _percentile(values, 0.90),
                "p95_ms": _percentile(values, 0.95),
                "p99_ms": _percentile(values, 0.99),
                "mean_ms": round(statistics.fmean(values), 3),
            }
            for name, values in sorted(span_values.items())
        },
        "audio_supply_ratio": {
            "count": len(supply),
            "p05": _percentile(supply, 0.05),
            "median": _percentile(supply, 0.50),
            "min": round(min(supply), 3) if supply else None,
        },
        "gaps": {
            "count": len(gaps),
            "p95_ms": _percentile(gaps, 0.95),
            "max_ms": round(max(gaps), 3) if gaps else None,
        },
        "token_components": {
            key: {
                "count": len(values),
                "p50_ms": _percentile(values, 0.50),
                "p95_ms": _percentile(values, 0.95),
                "p99_ms": _percentile(values, 0.99),
                "mean_ms": round(statistics.fmean(values), 3),
            }
            for key, values in sorted(token_values.items())
        },
        "token_summary": {
            "count": len(token_total_values),
            "first_token_p50_ms": _percentile(first_token_values, 0.50),
            "inter_token_p50_ms": _percentile(token_total_values[1:], 0.50),
            "tokens_per_second": round(1000 / statistics.fmean(token_total_values), 3)
            if token_total_values and statistics.fmean(token_total_values) > 0 else None,
            "llm_loop_overhead_p50_ms": _percentile(loop_overhead_values, 0.50),
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = aggregate(args.log)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
