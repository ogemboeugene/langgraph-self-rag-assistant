from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .models import ExecutionEvent, UsageSnapshot


@dataclass
class EventLog:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    events: list[ExecutionEvent] = field(default_factory=list)

    def record(
        self,
        event_type: str,
        stage: str,
        status: str,
        *,
        attempt: int | None = None,
        duration_ms: float | None = None,
        error_type: str | None = None,
        **details: Any,
    ) -> None:
        self.events.append(
            ExecutionEvent(
                run_id=self.run_id,
                event_type=event_type,
                stage=stage,
                status=status,
                timestamp=datetime.now(UTC).isoformat(),
                duration_ms=duration_ms,
                attempt=attempt,
                error_type=error_type,
                details=details,
            )
        )


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "elapsed_ms": self.elapsed_ms,
            "usage": [item.to_dict() for item in self.usage],
        }


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
