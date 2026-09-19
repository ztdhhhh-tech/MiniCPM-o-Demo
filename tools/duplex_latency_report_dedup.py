"""Aggregate MiniCPM-o duplex JSONL latency logs without double counting.

The backend emits unit-level spans and token timings in both
``backend.unit.done`` and ``backend.audio.done`` events. ``backend.audio.done``
duplicates the producing unit's metrics, so this aggregator takes unit-level
data only from ``backend.unit.done`` and per-output supply data only from
``backend.audio.done``.

Usage:
    python tools/duplex_latency_report_dedup.py logs/duplex_latency/session.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

TOKEN_KEYS = (
    "decode_ms",
    "item_sync_ms",
    "tokenizer_ms",
    "tokenizer_char_check_ms",
    "feed_wall_ms",
    "feed_gpu_ms",
    "decode_prev_feed_wait_ms",
    "decode_gpu_ms",
    "token_total_ms",
)


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, int(round((len(values) - 1) * p)))
    return round(values[index], 3)


def _unit_key(record: dict[str, Any]) -> tuple[Any, Any]:
    return record.get("session_id"), record.get("trace_id") or record.get("unit_id")


def _output_key(record: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (*_unit_key(record), record.get("output_seq"))


def aggregate(path: Path) -> dict[str, Any]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    span_values: dict[str, list[float]] = {}
    span_gpu_values: dict[str, list[float]] = {}
    token_values: dict[str, list[float]] = {}
    decode_gpu_span_values: dict[str, list[float]] = {}
    token_total_values: list[float] = []
    first_token_values: list[float] = []
    loop_overhead_values: list[float] = []
    gpu_sync_values: list[float] = []
    supply: list[float] = []
    gaps: list[float] = []

    seen_units: set[tuple[Any, Any]] = set()
    seen_outputs: set[tuple[Any, Any, Any]] = set()
    seen_tokens: set[tuple[Any, Any, Any]] = set()
    seen_finalizes: set[tuple[Any, Any]] = set()

    def add_span(name: str | None, duration: Any) -> None:
        if duration is not None:
            span_values.setdefault(name or "unknown", []).append(float(duration))

    def add_gpu_span(span: dict[str, Any]) -> None:
        gpu_ms = span.get("gpu_duration_ms")
        if gpu_ms is not None:
            span_gpu_values.setdefault(span.get("name") or "unknown", []).append(float(gpu_ms))

    for record in records:
        event = record.get("event")
        metrics = record.get("metrics") or {}

        # Standalone span events are independent records, not duplicated copies.
        for span in record.get("spans") or []:
            add_span(span.get("name"), span.get("duration_ms"))
            add_gpu_span(span)

        if event == "transport.ws_send" and record.get("duration_ms") is not None:
            span_values.setdefault("transport.ws_send", []).append(float(record["duration_ms"]))

        if event == "client.audio.gap" and record.get("gap_ms") is not None:
            gaps.append(float(record["gap_ms"]))

        if event == "backend.unit.done":
            unit_key = _unit_key(record)
            if unit_key in seen_units:
                continue
            seen_units.add(unit_key)

            latency = metrics.get("latency") or {}
            latency_metrics = latency.get("metrics") or {}
            for index, token in enumerate(latency_metrics.get("token_timings") or []):
                token_index = token.get("token_index", index)
                token_key = (*unit_key, token_index)
                if token_key in seen_tokens:
                    continue
                seen_tokens.add(token_key)
                for key in TOKEN_KEYS:
                    value = token.get(key)
                    if value is not None:
                        token_values.setdefault(key, []).append(float(value))
                for span_name, span_value in (token.get("decode_gpu_spans") or {}).items():
                    if span_value is not None:
                        decode_gpu_span_values.setdefault(span_name, []).append(float(span_value))
                if token.get("token_total_ms") is not None:
                    token_total_values.append(float(token["token_total_ms"]))
                    if index == 0:
                        first_token_values.append(float(token["token_total_ms"]))

            if latency_metrics.get("llm_loop_overhead_ms") is not None:
                loop_overhead_values.append(float(latency_metrics["llm_loop_overhead_ms"]))

            if latency_metrics.get("gpu_sync_overhead_ms") is not None:
                gpu_sync_values.append(float(latency_metrics["gpu_sync_overhead_ms"]))

            for span in latency.get("spans") or []:
                add_span(span.get("name"), span.get("duration_ms"))
                add_gpu_span(span)

            previous_finalize = metrics.get("last_finalize_latency") or {}
            finalize_key = (
                previous_finalize.get("session_id") or record.get("session_id"),
                previous_finalize.get("trace_id")
                or previous_finalize.get("unit_id")
                or unit_key[1],
            )
            if finalize_key not in seen_finalizes:
                seen_finalizes.add(finalize_key)
                for span in previous_finalize.get("spans") or []:
                    add_span(span.get("name"), span.get("duration_ms"))

        elif event == "backend.audio.done":
            output_key = _output_key(record)
            if output_key in seen_outputs:
                continue
            seen_outputs.add(output_key)
            pcm_ms = metrics.get("pcm_duration_ms")
            interval_ms = metrics.get("output_interval_ms")
            if pcm_ms is not None and interval_ms and interval_ms > 0:
                supply.append(float(pcm_ms) / float(interval_ms))

    return {
        "records": len(records),
        "units": len(seen_units),
        "audio_outputs": len(seen_outputs),
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
        "gpu_components": {
            name: {
                "count": len(values),
                "p50_ms": _percentile(values, 0.50),
                "p90_ms": _percentile(values, 0.90),
                "p95_ms": _percentile(values, 0.95),
                "p99_ms": _percentile(values, 0.99),
                "mean_ms": round(statistics.fmean(values), 3),
            }
            for name, values in sorted(span_gpu_values.items())
        },
        "gpu_sync_overhead_ms": {
            "count": len(gpu_sync_values),
            "p50_ms": _percentile(gpu_sync_values, 0.50),
            "p95_ms": _percentile(gpu_sync_values, 0.95),
            "mean_ms": round(statistics.fmean(gpu_sync_values), 3) if gpu_sync_values else None,
        },
        "decode_gpu_components": {
            name: {
                "count": len(values),
                "p50_ms": _percentile(values, 0.50),
                "p95_ms": _percentile(values, 0.95),
                "mean_ms": round(statistics.fmean(values), 3),
            }
            for name, values in sorted(decode_gpu_span_values.items())
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
            if token_total_values and statistics.fmean(token_total_values) > 0
            else None,
            "llm_loop_overhead_p50_ms": _percentile(loop_overhead_values, 0.50),
        },
    }


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
