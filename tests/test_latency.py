import json
from pathlib import Path

import torch
from core.telemetry import LatencyCollector, TraceContext
from core.telemetry.latency import SpanMeasurement, _Span
from MiniCPMO45.utils import StreamDecoder
from tools.duplex_latency_report import aggregate
from tools.duplex_latency_report_dedup import aggregate as aggregate_dedup
from tools.decode_gpu_report import aggregate as aggregate_decode_gpu


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


def test_latency_collector_measure_exposes_duration(monkeypatch):
    monkeypatch.setenv("MINICPM_LATENCY_MODE", "light")
    collector = LatencyCollector(TraceContext.create("s", 1, "in", 2))
    with collector.measure("feed") as measurement:
        pass
    assert measurement.duration_ms >= 0
    assert measurement.gpu_duration_ms is None


def test_latency_collector_token_aggregates(monkeypatch):
    monkeypatch.setenv("MINICPM_LATENCY_MODE", "light")
    collector = LatencyCollector(TraceContext.create("s", 1, "in", 2))
    token = {"token_total_ms": 4.0, "decode_ms": 1.0, "feed_wall_ms": 2.0}
    collector.update({"token_timings": [token]})
    payload = collector.finish()
    assert payload["metrics"]["token_total_sum_ms"] == 4.0
    assert payload["metrics"]["feed_wall_total_ms"] == 2.0


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


def test_decode_gpu_measurement_resolves_into_matching_token(monkeypatch):
    monkeypatch.setenv("MINICPM_LATENCY_MODE", "gpu")
    collector = LatencyCollector(TraceContext.create("s", 1, "in", 2))
    token = {
        "decode_prev_feed_wait_ms": None,
        "decode_gpu_ms": None,
        "decode_gpu_spans": None,
    }
    root = SpanMeasurement(_Span("model.generate.decode", 0))
    child = SpanMeasurement(_Span("decode.logits_clone", 0))
    root.item.attrs["gpu_duration_ms"] = 18.6
    child.item.attrs["gpu_duration_ms"] = 0.08
    root.set_wait_cpu_ms(31.2)

    collector.set_pending_decode_gpu(root, {"logits_clone": child})
    collector.attach_pending_decode_gpu(token)
    collector.finish(resolve_gpu=False)

    assert token["decode_gpu_ms"] == 18.6
    assert token["decode_prev_feed_wait_ms"] == 31.2
    assert token["decode_gpu_spans"] == {"logits_clone": 0.08}


def test_decode_gpu_fields_stay_null_without_cuda(monkeypatch):
    monkeypatch.setenv("MINICPM_LATENCY_MODE", "gpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    collector = LatencyCollector(TraceContext.create("s", 1, "in", 2))
    token = {
        "decode_prev_feed_wait_ms": None,
        "decode_gpu_ms": None,
        "decode_gpu_spans": None,
    }

    with collector.measure("model.generate.decode") as root:
        with collector.measure("decode.logits_clone", parent=root):
            pass
    collector.set_pending_decode_gpu(root)
    collector.attach_pending_decode_gpu(token)
    collector.finish()

    assert collector._gpu_spans == []
    assert token["decode_gpu_ms"] is None
    assert token["decode_prev_feed_wait_ms"] is None
    assert token["decode_gpu_spans"] == {}


def test_decode_trace_disabled_does_not_create_measurements(monkeypatch):
    monkeypatch.setenv("MINICPM_LATENCY_MODE", "gpu")
    collector = LatencyCollector(TraceContext.create("s", 1, "in", 2))

    class TracingDecoder:
        pass

    decoder = TracingDecoder.__new__(TracingDecoder)
    decoder.chunk_eos_id = -1
    decoder.forbidden_token_ids = []
    decoder.generated_tokens = []
    decoder.generated_special_tokens = []
    decoder.special_token_ids = []
    decoder.listen_id = 2
    decoder.turn_eos_id = 3
    decoder.context = ""
    decoder.decode = lambda *args, **kwargs: StreamDecoder.decode(decoder, *args, **kwargs)
    logits = __import__("torch").tensor([[0.0, -2.0, -4.0, -6.0, -8.0]])

    def unexpected_measure(*args, **kwargs):
        raise AssertionError("disabled decode trace must not create measurements")

    collector.measure = unexpected_measure
    decoder.decode(
        logits,
        latency_trace=collector,
        decode_gpu_trace=False,
        isolate_prev_feed=True,
    )

    assert collector._gpu_spans == []
    assert collector._pending_decode is None
    assert collector._pending_decode_measurements == []


def test_decode_gpu_report_and_dedup_aggregation(tmp_path: Path):
    token = {
        "token_index": 0,
        "decode_ms": 57.3,
        "decode_prev_feed_wait_ms": 31.2,
        "decode_gpu_ms": 18.6,
        "decode_gpu_spans": {"top_k_topp_sampling": 12.4},
        "feed_gpu_ms": 40.0,
    }
    record = {
        "event": "backend.unit.done",
        "session_id": "session",
        "trace_id": "session_u_1",
        "metrics": {"latency": {"metrics": {"token_timings": [token]}}},
    }
    log = tmp_path / "session.jsonl"
    log.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8")

    dedup = aggregate_dedup(log)
    gpu_report = aggregate_decode_gpu(log)

    assert dedup["token_components"]["decode_gpu_ms"]["p50_ms"] == 18.6
    assert dedup["token_components"]["decode_prev_feed_wait_ms"]["p50_ms"] == 31.2
    assert dedup["decode_gpu_components"]["top_k_topp_sampling"]["p50_ms"] == 12.4
    assert gpu_report["unit_table"][0]["token_count"] == 1
    assert gpu_report["decode_gpu_spans"]["top_k_topp_sampling"]["p50_ms"] == 12.4
    assert gpu_report["decode_wall_vs_gpu"]["p50_ms"] == 38.7
