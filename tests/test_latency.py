import json
from pathlib import Path

from core.telemetry import LatencyCollector, TraceContext
from tools.duplex_latency_report import aggregate


def test_latency_collector_disabled_by_default(monkeypatch):
    monkeypatch.setenv("MINICPM_LATENCY_MODE", "off")
    collector = LatencyCollector(TraceContext.create("s", 1, "in", 2))
    with collector.span("test"):
        pass
    assert collector.finish() == {}


def test_latency_collector_records_span(monkeypatch):
    monkeypatch.setenv("MINICPM_LATENCY_MODE", "light")
    collector = LatencyCollector(TraceContext.create("s", 1, "in", 2))
    with collector.span("test", value=3):
        pass
    payload = collector.finish()
    assert payload["trace_id"] == "s_u_2"
    assert payload["spans"][0]["name"] == "test"
    assert payload["spans"][0]["attrs"]["value"] == 3


def test_latency_report_aggregates_components(tmp_path: Path):
    log = tmp_path / "session.jsonl"
    log.write_text(
        json.dumps({
            "event": "backend.unit.done",
            "spans": [{"name": "model.tts", "duration_ms": 10}],
            "metrics": {"pcm_duration_ms": 1000, "output_interval_ms": 900},
        }) + "\n",
        encoding="utf-8",
    )
    result = aggregate(log)
    assert result["components"]["model.tts"]["p50_ms"] == 10.0
    assert result["audio_supply_ratio"]["median"] == 1.111
