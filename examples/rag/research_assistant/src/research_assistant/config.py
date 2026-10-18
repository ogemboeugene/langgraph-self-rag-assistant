from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


@dataclass(frozen=True)
class RuntimeConfig:
    max_retrieval_attempts: int = 3
    max_correction_attempts: int = 2
    max_graph_steps: int = 20
    min_evidence_chunks: int = 1
    max_retrieved_chunks: int = 6

    @classmethod
    def from_env(cls) -> RuntimeConfig:
        return cls(
            max_retrieval_attempts=_env_int("RESEARCH_ASSISTANT_MAX_RETRIEVAL_ATTEMPTS", 3),
            max_correction_attempts=_env_int("RESEARCH_ASSISTANT_MAX_CORRECTION_ATTEMPTS", 2),
            max_graph_steps=_env_int("RESEARCH_ASSISTANT_MAX_GRAPH_STEPS", 20),
            min_evidence_chunks=_env_int("RESEARCH_ASSISTANT_MIN_EVIDENCE_CHUNKS", 1),
            max_retrieved_chunks=_env_int("RESEARCH_ASSISTANT_MAX_RETRIEVED_CHUNKS", 6),
        )
