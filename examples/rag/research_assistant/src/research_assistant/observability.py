from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .models import ExecutionEvent, UsageSnapshot


@dataclass
class EventLog:
    events: list[ExecutionEvent] = field(default_factory=list)

    def record(self, event_type: str, **details: Any) -> None:
        self.events.append(ExecutionEvent(event_type=event_type, details=details))


@dataclass
class Metrics:
    started_at: float = field(default_factory=time.perf_counter)
    elapsed_ms: float = 0.0
    usage: list[UsageSnapshot] = field(default_factory=list)

    def complete(self) -> None:
        self.elapsed_ms = (time.perf_counter() - self.started_at) * 1000

    def add_usage(self, raw: Mapping[str, Any] | None) -> None:
        if raw is None:
            return
        self.usage.append(
            UsageSnapshot(
                provider=raw.get("provider"),
                model=raw.get("model"),
                prompt_tokens=_to_int(raw.get("prompt_tokens")),
                completion_tokens=_to_int(raw.get("completion_tokens")),
                total_tokens=_to_int(raw.get("total_tokens")),
                cost_usd=_to_float(raw.get("cost_usd")),
                latency_ms=_to_float(raw.get("latency_ms")),
            )
        )


def _to_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
