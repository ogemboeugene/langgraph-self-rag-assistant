from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from research_assistant.config import RuntimeConfig
from research_assistant.evaluation import compare_workflows, evaluate_cases, load_fixture
from research_assistant.generation import GenerationResponse
from research_assistant.graph import ResearchAssistant
from research_assistant.retrieval import RetrievalResponse, RetrievedChunk


@dataclass
class StaticRetriever:
    chunks: list[RetrievedChunk]

    def retrieve(self, query: str, limit: int) -> RetrievalResponse:
        del query, limit
        return RetrievalResponse(chunks=self.chunks)


@dataclass
class FixtureRetriever:
    fallback_chunks: list[RetrievedChunk]

    def retrieve(self, query: str, limit: int) -> RetrievalResponse:
        del limit
        if "unknown" in query.lower():
            return RetrievalResponse(chunks=[])
        if "rewrite" in query.lower() and "retry" not in query.lower():
            return RetrievalResponse(chunks=[])
        return RetrievalResponse(chunks=self.fallback_chunks)


class DeterministicGenerator:
    def __init__(self, invalid_first: bool = False) -> None:
        self.invalid_first = invalid_first
        self.calls = 0

    def draft(self, question, evidence, attempt):
        del question, attempt
        self.calls += 1
        citation = "missing-id" if self.invalid_first else evidence[0].evidence_id
        return GenerationResponse(
            payload={
                "summary": "Deterministic summary",
                "claims": [
                    {
                        "claim_id": "claim-01",
                        "text": "LangGraph enables stateful multi-step agents.",
                        "citation_ids": [citation],
                    }
                ],
            }
        )

    def correct(
        self,
        question,
        evidence,
        previous_payload,
        citation_issues,
        unsupported_claims,
        attempt,
        parse_error_context=None,
    ):
        del (
            question,
            previous_payload,
            citation_issues,
            unsupported_claims,
            attempt,
            parse_error_context,
        )
        return GenerationResponse(
            payload={
                "summary": "Corrected deterministic summary",
                "claims": [
                    {
                        "claim_id": "claim-01",
                        "text": "LangGraph enables stateful multi-step agents.",
                        "citation_ids": [evidence[0].evidence_id],
                    }
                ],
            }
        )


if __name__ == "__main__":
    evidence = [
        RetrievedChunk(
            text="LangGraph enables stateful, multi-step, durable AI workflows.",
            source="https://docs.langchain.com/oss/python/langgraph/overview",
            metadata={"title": "LangGraph overview"},
        )
    ]

    baseline = ResearchAssistant(
        retriever=StaticRetriever(chunks=evidence),
        generator=DeterministicGenerator(invalid_first=False),
        config=RuntimeConfig(max_correction_attempts=0),
    )
    improved = ResearchAssistant(
        retriever=StaticRetriever(chunks=evidence),
        generator=DeterministicGenerator(invalid_first=True),
        config=RuntimeConfig(max_correction_attempts=2),
    )

    result = compare_workflows(
        question="What is LangGraph useful for?",
        baseline=baseline,
        assistant=improved,
    )

    fixture_path = Path(__file__).resolve().parents[1] / "evaluation" / "deterministic_fixture.json"
    fixture_cases = load_fixture(fixture_path)
    evaluation_assistant = ResearchAssistant(
        retriever=FixtureRetriever(fallback_chunks=evidence),
        generator=DeterministicGenerator(invalid_first=False),
        config=RuntimeConfig(max_correction_attempts=1, max_retrieval_attempts=2),
    )
    summary = evaluate_cases(evaluation_assistant, fixture_cases)

    print("Measured deterministic comparison")
    print(result)
    print("Measured deterministic evaluation")
    print(json.dumps(summary.to_dict(), indent=2))
    print(
        "Provider-backed runs are illustrative and require adapter injection plus credentials. "
        "No provider-backed metrics are reported here."
    )
