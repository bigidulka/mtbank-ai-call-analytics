"""Privacy-safe structured telemetry for the internal monitoring boundary."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock
from uuid import UUID

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, Tracer

_REQUEST_ID: ContextVar[str | None] = ContextVar("mtbank_request_id", default=None)
_RUN_ID: ContextVar[str | None] = ContextVar("mtbank_run_id", default=None)

_FORBIDDEN_KEYS = frozenset(
    {
        "audio",
        "content",
        "data",
        "prompt",
        "provider_response",
        "response_body",
        "transcript",
        "url",
        "api_key",
        "authorization",
        "cookie",
        "query",
    }
)
_FORBIDDEN_VALUE_MARKERS = ("sk-", "bearer ", "http://", "https://")


@dataclass(frozen=True, slots=True)
class SpanRecord:
    name: str
    duration_seconds: float
    status: str
    attributes: Mapping[str, str | int | float | bool]


@dataclass(slots=True)
class _Histogram:
    buckets: tuple[float, ...]
    values: list[float] = field(default_factory=list)


class MetricsRegistry:
    """Small dependency-free Prometheus exposition registry with fixed label sets."""

    _LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 180.0)

    def __init__(self) -> None:
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._histograms: dict[tuple[str, tuple[tuple[str, str], ...]], _Histogram] = {}
        self._lock = Lock()

    def increment(self, name: str, value: float = 1.0, **labels: str | int | bool) -> None:
        key = (name, _labels(labels))
        with self._lock:
            self._counters[key] += value

    def gauge(self, name: str, value: float, **labels: str | int | bool) -> None:
        key = (name, _labels(labels))
        with self._lock:
            self._gauges[key] = value

    def observe(self, name: str, seconds: float, **labels: str | int | bool) -> None:
        key = (name, _labels(labels))
        with self._lock:
            histogram = self._histograms.get(key)
            if histogram is None:
                histogram = _Histogram(self._LATENCY_BUCKETS)
                self._histograms[key] = histogram
            histogram.values.append(max(0.0, seconds))

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            for (name, labels), value in sorted(self._counters.items()):
                lines.append(f"{name}{_render_labels(labels)} {value:g}")
            for (name, labels), value in sorted(self._gauges.items()):
                lines.append(f"{name}{_render_labels(labels)} {value:g}")
            for (name, labels), histogram in sorted(self._histograms.items()):
                for bucket in histogram.buckets:
                    count = sum(value <= bucket for value in histogram.values)
                    lines.append(f"{name}_bucket{_render_labels((*labels, ('le', str(bucket))))} {count}")
                lines.append(f"{name}_bucket{_render_labels((*labels, ('le', '+Inf')))} {len(histogram.values)}")
                lines.append(f"{name}_count{_render_labels(labels)} {len(histogram.values)}")
                lines.append(f"{name}_sum{_render_labels(labels)} {sum(histogram.values):g}")
        return "\n".join(lines) + "\n"


class Telemetry:
    """Collects safe local metrics and spans; exporters can consume ``spans`` externally."""

    def __init__(self, metrics: MetricsRegistry | None = None, *, tracer: Tracer | None = None) -> None:
        self.metrics = metrics or MetricsRegistry()
        self._spans: list[SpanRecord] = []
        self._lock = Lock()
        self._tracer = tracer or trace.get_tracer("mtbank_ai")

    @property
    def spans(self) -> tuple[SpanRecord, ...]:
        with self._lock:
            return tuple(self._spans)

    @contextmanager
    def span(self, name: str, **attributes: str | int | float | bool) -> Iterator[None]:
        started = time.monotonic()
        status = "ok"
        safe_attributes = _safe_attributes(attributes)
        with self._tracer.start_as_current_span(
            name,
            attributes=safe_attributes,
            record_exception=False,
            set_status_on_exception=False,
        ) as otel_span:
            try:
                yield
            except BaseException as error:
                status = "error"
                error_attributes = _sanitized_error_attributes(error)
                safe_attributes.update(error_attributes)
                otel_span.set_attributes(error_attributes)
                otel_span.set_status(Status(StatusCode.ERROR))
                raise
            finally:
                duration = max(0.0, time.monotonic() - started)
                with self._lock:
                    self._spans.append(SpanRecord(name, duration, status, safe_attributes))
                self.metrics.observe("mtbank_stage_latency_seconds", duration, stage=name, status=status)

    @contextmanager
    def context(self, *, request_id: UUID | str | None = None, run_id: UUID | str | None = None) -> Iterator[None]:
        request_token = _REQUEST_ID.set(str(request_id)) if request_id is not None else None
        run_token = _RUN_ID.set(str(run_id)) if run_id is not None else None
        try:
            yield
        finally:
            if run_token is not None:
                _RUN_ID.reset(run_token)
            if request_token is not None:
                _REQUEST_ID.reset(request_token)

    def event(self, name: str, **attributes: str | int | float | bool) -> None:
        self.metrics.increment("mtbank_events_total", event=name)
        StructuredJsonLogger(logging.getLogger("mtbank_ai.telemetry")).info(name, **attributes)


class StructuredJsonLogger:
    """JSON logger which discards content-bearing fields rather than attempting masking."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def info(self, event: str, **fields: object) -> None:
        payload: dict[str, object] = {"event": event}
        if _REQUEST_ID.get() is not None:
            payload["request_id"] = _REQUEST_ID.get()
        if _RUN_ID.get() is not None:
            payload["run_id"] = _RUN_ID.get()
        payload.update(_redact(fields))
        self._logger.info(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def _labels(values: Mapping[str, str | int | bool]) -> tuple[tuple[str, str], ...]:
    labels = tuple(
        sorted((key, str(value).lower() if isinstance(value, bool) else str(value)) for key, value in values.items())
    )
    if any(key in _FORBIDDEN_KEYS or len(value) > 128 for key, value in labels):
        raise ValueError("telemetry label is content-bearing or unbounded")
    return labels


def _render_labels(labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    values = ",".join(f'{key}="{value.replace(chr(34), chr(92) + chr(34))}"' for key, value in labels)
    return "{" + values + "}"


def _safe_attributes(values: Mapping[str, str | int | float | bool]) -> dict[str, str | int | float | bool]:
    return {key: value for key, value in values.items() if key not in _FORBIDDEN_KEYS and not _is_sensitive(value)}


def _sanitized_error_attributes(error: BaseException) -> dict[str, str]:
    if isinstance(error, asyncio.CancelledError):
        return {"error.type": "cancelled", "error.code": "cancelled"}
    if isinstance(error, TimeoutError):
        return {"error.type": "timeout", "error.code": "deadline_exceeded"}
    if isinstance(error, ValueError):
        return {"error.type": "validation", "error.code": "invalid_input"}
    if isinstance(error, RuntimeError):
        return {"error.type": "runtime", "error.code": "internal"}
    return {"error.type": "unknown", "error.code": "internal"}


def _redact(values: Mapping[str, object]) -> dict[str, object]:
    safe: dict[str, object] = {}
    for key, value in values.items():
        if key.casefold() in _FORBIDDEN_KEYS or _is_sensitive(value):
            safe[key] = "[redacted]"
        elif isinstance(value, Mapping):
            safe[key] = _redact({str(nested_key): nested_value for nested_key, nested_value in value.items()})
        elif isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        else:
            safe[key] = "[redacted]"
    return safe


def _is_sensitive(value: object) -> bool:
    return isinstance(value, str) and any(marker in value.casefold() for marker in _FORBIDDEN_VALUE_MARKERS)
