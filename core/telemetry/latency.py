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
from typing import Any, Dict, Iterator, Optional


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
