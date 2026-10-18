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


@dataclass(frozen=True)
class ExecutionEvent:
    event_type: str
    details: dict[str, Any] = field(default_factory=dict)
