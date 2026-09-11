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
    supply: list[float] = []
    gaps: list[float] = []
    for record in records:
        metrics = record.get("metrics") or {}
        latency = metrics.get("latency") or {}
        previous_finalize = metrics.get("last_finalize_latency") or {}
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
