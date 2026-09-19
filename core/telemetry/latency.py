"""Low-overhead component latency tracing.

The tracer is deliberately dependency-light so it can be imported by model
code and the websocket adapter.  ``light`` mode never synchronizes CUDA.
``gpu`` mode records CUDA events and resolves them with one synchronization
at the end of a unit; the synchronization cost is reported separately.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Mapping, Optional


def get_latency_mode() -> str:
    mode = os.getenv("MINICPM_LATENCY_MODE", "off").strip().lower()
    return mode if mode in {"off", "light", "gpu", "profile"} else "off"


def _sampled() -> bool:
    try:
        rate = float(os.getenv("MINICPM_LATENCY_SAMPLE_RATE", "1.0"))
    except ValueError:
        rate = 1.0
    rate = max(0.0, min(1.0, rate))
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    # Stable, cheap sampling per trace rather than per span.
    return (uuid.uuid4().int % 10_000) < int(rate * 10_000)


def decode_gpu_trace_enabled() -> bool:
    value = os.getenv("MINICPM_DECODE_GPU_TRACE", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def decode_isolate_prev_feed_enabled() -> bool:
    value = os.getenv("MINICPM_DECODE_ISOLATE_PREV_FEED", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class TraceContext:
    session_id: str
    turn_id: int
    input_chunk_id: str
    unit_id: int
    trace_id: str
    output_seq: Optional[int] = None

    @classmethod
    def create(cls, session_id: str, turn_id: int, input_chunk_id: str, unit_id: int) -> "TraceContext":
        return cls(
            session_id=session_id,
            turn_id=turn_id,
            input_chunk_id=input_chunk_id,
            unit_id=unit_id,
            trace_id=f"{session_id}_u_{unit_id}",
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "input_chunk_id": self.input_chunk_id,
            "unit_id": self.unit_id,
            "trace_id": self.trace_id,
            "output_seq": self.output_seq,
        }


@dataclass
class _Span:
    name: str
    start_ns: int
    attrs: Dict[str, Any] = field(default_factory=dict)
    end_ns: Optional[int] = None
    error: Optional[str] = None
    gpu_start: Any = None
    gpu_end: Any = None


@dataclass
class SpanMeasurement:
    """A completed measurement whose GPU duration may resolve later."""

    item: Optional[_Span] = None
    _wait_cpu_ms: Optional[float] = None

    @property
    def duration_ms(self) -> float:
        if self.item is None:
            return 0.0
        end_ns = self.item.end_ns or time.perf_counter_ns()
        return max(0.0, (end_ns - self.item.start_ns) / 1_000_000)

    @property
    def gpu_duration_ms(self) -> Optional[float]:
        if self.item is None:
            return None
        value = self.item.attrs.get("gpu_duration_ms")
        return float(value) if value is not None else None

    @property
    def wait_cpu_ms(self) -> Optional[float]:
        return self._wait_cpu_ms

    def wait_cpu(self) -> Optional[float]:
        """Wait on the stream for this span's completion event.

        This is intentionally opt-in: normal spans stay asynchronous and are
        resolved together by ``LatencyCollector.finish()``.
        """
        if self.item is None or self.item.gpu_end is None:
            return None
        if self._wait_cpu_ms is not None:
            return self._wait_cpu_ms
        try:
            import torch

            wait_start = time.perf_counter_ns()
            self.item.gpu_end.synchronize()
            self._wait_cpu_ms = round((time.perf_counter_ns() - wait_start) / 1_000_000, 3)
        except Exception:
            self._wait_cpu_ms = None
        return self._wait_cpu_ms

    def set_wait_cpu_ms(self, value: Optional[float]) -> None:
        self._wait_cpu_ms = round(float(value), 3) if value is not None else None


@dataclass
class _PendingDecodeMeasurement:
    root: SpanMeasurement
    children: Dict[str, SpanMeasurement]


class LatencyCollector:
    """Collect CPU spans and optional CUDA event spans for one unit."""

    def __init__(self, context: Optional[TraceContext] = None, mode: Optional[str] = None) -> None:
        self.context = context
        self.mode = mode or get_latency_mode()
        self.enabled = self.mode != "off" and _sampled()
        self._spans: list[_Span] = []
        self._metrics: Dict[str, Any] = {}
        self._gpu_spans: list[_Span] = []
        self._gpu_resolved = False
        self._token_measurements: list[tuple[Dict[str, Any], SpanMeasurement]] = []
        self._pending_decode: Optional[_PendingDecodeMeasurement] = None
        self._pending_decode_measurements: list[tuple[Dict[str, Any], _PendingDecodeMeasurement]] = []

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator["LatencyCollector"]:
        if not self.enabled:
            yield self
            return
        item = _Span(name=name, start_ns=time.perf_counter_ns(), attrs=dict(attrs))
        if self.mode == "gpu":
            try:
                import torch

                if torch.cuda.is_available():
                    item.gpu_start = torch.cuda.Event(enable_timing=True)
                    item.gpu_end = torch.cuda.Event(enable_timing=True)
                    item.gpu_start.record()
                    self._gpu_spans.append(item)
            except Exception:
                item.gpu_start = item.gpu_end = None
        self._spans.append(item)
        try:
            yield self
        except Exception as exc:
            item.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            item.end_ns = time.perf_counter_ns()
            if item.gpu_end is not None:
                try:
                    item.gpu_end.record()
                except Exception:
                    item.gpu_end = None

    @contextmanager
    def measure(self, name: str, parent: Optional[SpanMeasurement] = None, **attrs: Any) -> Iterator[SpanMeasurement]:
        """Measure one operation and expose CPU/GPU durations to the caller.

        GPU events are resolved only by ``finish()``; this method never calls
        ``torch.cuda.synchronize()`` on the hot token path.
        """
        if not self.enabled:
            yield SpanMeasurement()
            return

        item = _Span(name=name, start_ns=time.perf_counter_ns(), attrs=dict(attrs))
        if parent is not None and parent.item is not None:
            item.attrs["parent_span"] = parent.item.name
        if self.mode == "gpu":
            try:
                import torch

                if torch.cuda.is_available():
                    item.gpu_start = torch.cuda.Event(enable_timing=True)
                    item.gpu_end = torch.cuda.Event(enable_timing=True)
                    item.gpu_start.record()
                    self._gpu_spans.append(item)
            except Exception:
                item.gpu_start = item.gpu_end = None
        self._spans.append(item)
        measurement = SpanMeasurement(item)
        try:
            yield measurement
        except Exception as exc:
            item.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            item.end_ns = time.perf_counter_ns()
            if item.gpu_end is not None:
                try:
                    item.gpu_end.record()
                except Exception:
                    item.gpu_end = None

    def attach_token_measurement(
        self, token_record: Dict[str, Any], measurement: Optional[SpanMeasurement]
    ) -> None:
        if self.enabled and measurement is not None:
            self._token_measurements.append((token_record, measurement))

    def set_pending_decode_gpu(
        self,
        root: SpanMeasurement,
        children: Optional[Mapping[str, SpanMeasurement]] = None,
    ) -> None:
        """Stage a decode root/child pair for resolution at unit finish."""
        if self.enabled:
            self._pending_decode = _PendingDecodeMeasurement(root, dict(children or {}))

    def attach_pending_decode_gpu(self, token_record: Dict[str, Any]) -> None:
        """Bind the latest decode measurement to the token that produced it."""
        if not self.enabled or self._pending_decode is None:
            self._pending_decode = None
            return
        self._pending_decode_measurements.append((token_record, self._pending_decode))
        self._pending_decode = None

    def _write_pending_decode(self) -> None:
        for token_record, pending in self._pending_decode_measurements:
            token_record["decode_gpu_ms"] = pending.root.gpu_duration_ms
            token_record["decode_gpu_spans"] = {
                name: measurement.gpu_duration_ms
                for name, measurement in pending.children.items()
            }
            token_record["decode_prev_feed_wait_ms"] = pending.root.wait_cpu_ms
        self._pending_decode_measurements.clear()

    def metric(self, name: str, value: Any) -> None:
        if self.enabled and value is not None:
            self._metrics[name] = value

    def record(self, name: str, duration_ms: float, **attrs: Any) -> None:
        """Add a completed span when the caller already owns the timer."""
        if not self.enabled:
            return
        now = time.perf_counter_ns()
        self._spans.append(
            _Span(
                name=name,
                start_ns=now - int(max(0.0, duration_ms) * 1_000_000),
                end_ns=now,
                attrs=dict(attrs),
            )
        )

    def update(self, values: Optional[Dict[str, Any]]) -> None:
        if self.enabled and values:
            self._metrics.update({k: v for k, v in values.items() if v is not None})

    def _resolve_gpu(self) -> Optional[float]:
        if not self.enabled or self.mode != "gpu" or not self._gpu_spans or self._gpu_resolved:
            return None
        try:
            import torch

            sync_start = time.perf_counter_ns()
            torch.cuda.synchronize()
            sync_ms = (time.perf_counter_ns() - sync_start) / 1_000_000
            self._metrics["gpu_sync_overhead_ms"] = round(sync_ms, 3)
            for item in self._gpu_spans:
                if item.gpu_start is not None and item.gpu_end is not None:
                    item.attrs["gpu_duration_ms"] = round(item.gpu_start.elapsed_time(item.gpu_end), 3)
            self._gpu_resolved = True
            return sync_ms
        except Exception:
            return None

    def finish(self, resolve_gpu: bool = True) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        if resolve_gpu:
            self._resolve_gpu()
        for token_record, measurement in self._token_measurements:
            token_record["feed_gpu_ms"] = measurement.gpu_duration_ms
        self._write_pending_decode()
        token_timings = self._metrics.get("token_timings")
        if isinstance(token_timings, list):
            self._metrics.update({
                "token_total_sum_ms": round(sum(float(t.get("token_total_ms", 0.0)) for t in token_timings), 3),
                "decode_total_ms": round(sum(float(t.get("decode_ms", 0.0)) for t in token_timings), 3),
                "item_sync_total_ms": round(sum(float(t.get("item_sync_ms", 0.0)) for t in token_timings), 3),
                "tokenizer_total_ms": round(sum(float(t.get("tokenizer_ms", 0.0)) for t in token_timings), 3),
                "tokenizer_char_check_total_ms": round(sum(float(t.get("tokenizer_char_check_ms", 0.0)) for t in token_timings), 3),
                "feed_wall_total_ms": round(sum(float(t.get("feed_wall_ms", 0.0)) for t in token_timings), 3),
            })
            gpu_values = [t.get("feed_gpu_ms") for t in token_timings if t.get("feed_gpu_ms") is not None]
            self._metrics["feed_gpu_total_ms"] = round(sum(gpu_values), 3) if gpu_values else None
        spans = []
        for item in self._spans:
            end_ns = item.end_ns or time.perf_counter_ns()
            record = {
                "name": item.name,
                "start_ns": item.start_ns,
                "duration_ms": round((end_ns - item.start_ns) / 1_000_000, 3),
                "attrs": dict(item.attrs),
            }
            if item.error:
                record["error"] = item.error
            gpu_ms = record["attrs"].pop("gpu_duration_ms", None)
            if gpu_ms is not None:
                record["gpu_duration_ms"] = gpu_ms
            spans.append(record)
        payload: Dict[str, Any] = {"mode": self.mode, "spans": spans, "metrics": dict(self._metrics)}
        if self.context:
            payload.update(self.context.as_dict())
        return payload


class JsonlLatencyWriter:
    """Bounded background JSONL writer; drops only diagnostic events."""

    def __init__(self, session_id: str, log_dir: Optional[str] = None) -> None:
        self.log_dir = log_dir or os.getenv("MINICPM_LATENCY_LOG_DIR", "logs/duplex_latency")
        self.enabled = get_latency_mode() != "off"
        self.dropped_event_count = 0
        try:
            max_events = max(1, int(os.getenv("MINICPM_LATENCY_MAX_EVENTS", "20000")))
        except ValueError:
            max_events = 20000
        self._queue: queue.Queue[Optional[Dict[str, Any]]] = queue.Queue(
            maxsize=max_events
        )
        self._thread: Optional[threading.Thread] = None
        self._path = os.path.join(self.log_dir, f"{session_id}.jsonl")
        if self.enabled:
            os.makedirs(self.log_dir, exist_ok=True)
            self._thread = threading.Thread(target=self._run, name="duplex-latency-writer", daemon=True)
            self._thread.start()

    def write(self, event: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self.dropped_event_count += 1

    def close(self) -> None:
        if not self.enabled or self._thread is None:
            return
        self._queue.put(None)
        self._thread.join(timeout=2.0)
        self._thread = None

    def _run(self) -> None:
        with open(self._path, "a", encoding="utf-8") as handle:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                handle.write(json_dumps(item) + "\n")
                handle.flush()


def json_dumps(value: Any) -> str:
    """Safe serializer for telemetry records (used by async log writers)."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
