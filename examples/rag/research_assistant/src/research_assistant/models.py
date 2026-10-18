from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class TerminalStatus(StrEnum):
    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CITATION_VALIDATION_FAILED = "citation_validation_failed"
    MODEL_ERROR = "model_error"
    RETRY_LIMIT_EXCEEDED = "retry_limit_exceeded"


@dataclass(frozen=True)
class EvidenceChunk:
    evidence_id: str
    source_id: str
    chunk_id: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Claim:
    claim_id: str
    text: str
    citation_ids: list[str]


@dataclass(frozen=True)
class ResearchReport:
    question: str
    summary: str
    claims: list[Claim]


@dataclass(frozen=True)
class CitationIssue:
    claim_id: str
    citation_id: str
    reason: str
    support_score: float | None = None
    matched_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "citation_id": self.citation_id,
            "reason": self.reason,
            "support_score": self.support_score,
            "matched_terms": list(self.matched_terms),
        }


@dataclass(frozen=True)
class UnsupportedClaim:
    claim_id: str
    reason: str


@dataclass(frozen=True)
class UsageSnapshot:
    provider: str | None = None
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
        }


@dataclass(frozen=True)
class ExecutionEvent:
    run_id: str
    event_type: str
    stage: str
    status: str
    timestamp: str
    duration_ms: float | None = None
    attempt: int | None = None
    error_type: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "event_type": self.event_type,
            "stage": self.stage,
            "status": self.status,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "attempt": self.attempt,
            "error_type": self.error_type,
            "details": dict(self.details),
        }
