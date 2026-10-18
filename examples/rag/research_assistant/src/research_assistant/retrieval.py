from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .models import EvidenceChunk


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RetrievalResponse:
    chunks: list[RetrievedChunk]
    usage: dict[str, Any] | None = None


class Retriever(Protocol):
    def retrieve(self, query: str, limit: int) -> RetrievalResponse: ...


class QueryRewriter(Protocol):
    def rewrite(self, question: str, attempt: int, query_history: list[str]) -> str: ...


def normalize_chunks(chunks: list[RetrievedChunk]) -> list[EvidenceChunk]:
    normalized: list[EvidenceChunk] = []
    source_positions: dict[str, int] = {}
    source_ids: dict[str, int] = {}

    for index, chunk in enumerate(chunks, start=1):
        source_key = chunk.source.strip() or "unknown"
        source_positions[source_key] = source_positions.get(source_key, 0) + 1
        if source_key not in source_ids:
            source_ids[source_key] = len(source_ids) + 1
        source_idx = source_ids[source_key]
        chunk_id = source_positions[source_key]
        evidence_id = f"source-{source_idx:02d}#chunk-{chunk_id:03d}"
        metadata = {"source": source_key, "index": index, **chunk.metadata}
        normalized.append(
            EvidenceChunk(
                evidence_id=evidence_id,
                source_id=f"source-{source_idx:02d}",
                chunk_id=chunk_id,
                text=chunk.text,
                metadata=metadata,
            )
        )
    return normalized
