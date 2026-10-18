from __future__ import annotations

from dataclasses import dataclass

from research_assistant.config import RuntimeConfig
from research_assistant.generation import GenerationResponse
from research_assistant.graph import ResearchAssistant
from research_assistant.models import TerminalStatus
from research_assistant.retrieval import RetrievalResponse, RetrievedChunk


@dataclass
class SequenceRetriever:
    responses: list[RetrievalResponse | Exception]

    def __post_init__(self) -> None:
        self.calls = 0

    def retrieve(self, query: str, limit: int) -> RetrievalResponse:
        del query, limit
        item = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


class ScriptedGenerator:
    def __init__(
        self, drafts: list[GenerationResponse | Exception], corrections: list[GenerationResponse]
    ):
        self.drafts = drafts
        self.corrections = corrections
        self.draft_calls = 0
        self.correct_calls = 0

    def draft(self, question, evidence, attempt):
        del question, evidence, attempt
        item = self.drafts[min(self.draft_calls, len(self.drafts) - 1)]
        self.draft_calls += 1
        if isinstance(item, Exception):
            raise item
        return item

    def correct(
        self, question, evidence, previous_payload, citation_issues, unsupported_claims, attempt
    ):
        del question, evidence, previous_payload, citation_issues, unsupported_claims, attempt
        item = self.corrections[min(self.correct_calls, len(self.corrections) - 1)]
        self.correct_calls += 1
        return item


def _chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(text=text, source="doc://langgraph", metadata={})


def test_successful_retrieval_and_answer() -> None:
    retriever = SequenceRetriever([RetrievalResponse(chunks=[_chunk("durable stateful agents")])])
    generator = ScriptedGenerator(
        drafts=[
            GenerationResponse(
                payload={
                    "summary": "LangGraph supports durable stateful workflows.",
                    "claims": [
                        {
                            "claim_id": "c1",
                            "text": "LangGraph supports durable stateful workflows",
                            "citation_ids": ["source-01#chunk-001"],
                        }
                    ],
                },
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )
        ],
        corrections=[],
    )
    result = ResearchAssistant(retriever, generator).run("What does LangGraph support?")
    assert result.status == TerminalStatus.ANSWERED
    assert result.report is not None
    assert len(result.metrics.usage) == 1


def test_empty_retrieval_leads_to_insufficient_evidence() -> None:
    retriever = SequenceRetriever([RetrievalResponse(chunks=[]), RetrievalResponse(chunks=[])])
    generator = ScriptedGenerator(drafts=[], corrections=[])
    assistant = ResearchAssistant(
        retriever,
        generator,
        config=RuntimeConfig(max_retrieval_attempts=2, max_correction_attempts=1),
    )
    result = assistant.run("Unanswerable question")
    assert result.status == TerminalStatus.INSUFFICIENT_EVIDENCE


def test_retrieval_failure_hits_retry_limit() -> None:
    retriever = SequenceRetriever([RuntimeError("network down"), RuntimeError("still down")])
    generator = ScriptedGenerator(drafts=[], corrections=[])
    assistant = ResearchAssistant(
        retriever,
        generator,
        config=RuntimeConfig(max_retrieval_attempts=2, max_correction_attempts=1),
    )
    result = assistant.run("Question")
    assert result.status == TerminalStatus.RETRY_LIMIT_EXCEEDED
    assert any("retriever_error" in err for err in result.errors)


def test_malformed_output_triggers_model_error() -> None:
    retriever = SequenceRetriever([RetrievalResponse(chunks=[_chunk("stateful durable agents")])])
    generator = ScriptedGenerator(
        drafts=[GenerationResponse(payload={"summary": "missing claims"})],
        corrections=[GenerationResponse(payload={"summary": "still bad"})],
    )
    assistant = ResearchAssistant(
        retriever,
        generator,
        config=RuntimeConfig(max_correction_attempts=1),
    )
    result = assistant.run("Question")
    assert result.status == TerminalStatus.MODEL_ERROR


def test_invalid_citation_then_correction_success() -> None:
    retriever = SequenceRetriever(
        [RetrievalResponse(chunks=[_chunk("langgraph durable execution")])]
    )
    generator = ScriptedGenerator(
        drafts=[
            GenerationResponse(
                payload={
                    "summary": "Draft",
                    "claims": [
                        {
                            "claim_id": "c1",
                            "text": "langgraph durable execution",
                            "citation_ids": ["bad-id"],
                        }
                    ],
                }
            )
        ],
        corrections=[
            GenerationResponse(
                payload={
                    "summary": "Corrected",
                    "claims": [
                        {
                            "claim_id": "c1",
                            "text": "langgraph durable execution",
                            "citation_ids": ["source-01#chunk-001"],
                        }
                    ],
                }
            )
        ],
    )
    assistant = ResearchAssistant(
        retriever, generator, config=RuntimeConfig(max_correction_attempts=2)
    )
    result = assistant.run("Question")
    assert result.status == TerminalStatus.ANSWERED
    assert result.correction_attempts >= 1


def test_unsupported_claims_exhaust_corrections() -> None:
    retriever = SequenceRetriever([RetrievalResponse(chunks=[_chunk("langgraph only")])])
    bad_payload = GenerationResponse(
        payload={
            "summary": "Draft",
            "claims": [
                {
                    "claim_id": "c1",
                    "text": "kubernetes autoscaling strategies",
                    "citation_ids": ["source-01#chunk-001"],
                }
            ],
        }
    )
    generator = ScriptedGenerator(drafts=[bad_payload], corrections=[bad_payload, bad_payload])
    assistant = ResearchAssistant(
        retriever, generator, config=RuntimeConfig(max_correction_attempts=1)
    )
    result = assistant.run("Question")
    assert result.status == TerminalStatus.CITATION_VALIDATION_FAILED


def test_metrics_and_events_are_recorded() -> None:
    retriever = SequenceRetriever(
        [
            RetrievalResponse(
                chunks=[_chunk("stateful workflows")],
                usage={"provider": "fake", "latency_ms": 8.5},
            )
        ]
    )
    generator = ScriptedGenerator(
        drafts=[
            GenerationResponse(
                payload={
                    "summary": "ok",
                    "claims": [
                        {
                            "claim_id": "c1",
                            "text": "stateful workflows",
                            "citation_ids": ["source-01#chunk-001"],
                        }
                    ],
                },
                usage={"provider": "fake", "total_tokens": 12, "cost_usd": 0.001},
            )
        ],
        corrections=[],
    )
    result = ResearchAssistant(retriever, generator).run("Question")
    assert result.metrics.elapsed_ms >= 0
    assert len(result.events) >= 3
    assert len(result.metrics.usage) == 2
