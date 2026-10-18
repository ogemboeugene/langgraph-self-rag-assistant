from __future__ import annotations

from dataclasses import dataclass

from research_assistant.config import RuntimeConfig
from research_assistant.evaluation import compare_workflows
from research_assistant.generation import GenerationResponse
from research_assistant.graph import ResearchAssistant
from research_assistant.retrieval import RetrievalResponse, RetrievedChunk


@dataclass
class StaticRetriever:
    chunks: list[RetrievedChunk]

    def retrieve(self, query: str, limit: int) -> RetrievalResponse:
        del query, limit
        return RetrievalResponse(chunks=self.chunks)


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
        self, question, evidence, previous_payload, citation_issues, unsupported_claims, attempt
    ):
        del question, previous_payload, citation_issues, unsupported_claims, attempt
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
    print("Measured deterministic comparison")
    print(result)
    print("Provider-backed runs are illustrative and require configured credentials.")
