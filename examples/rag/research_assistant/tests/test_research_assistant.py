from __future__ import annotations

from dataclasses import dataclass

from research_assistant.config import RuntimeConfig
from research_assistant.generation import (
    GenerationResponse,
    validate_citations,
)
from research_assistant.graph import ResearchAssistant
from research_assistant.models import Claim, EvidenceChunk, ResearchReport, TerminalStatus
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
        self,
        drafts: list[GenerationResponse | Exception],
        corrections: list[GenerationResponse | Exception],
    ):
        self.drafts = drafts
        self.corrections = corrections
        self.draft_calls = 0
        self.correct_calls = 0
        self.correction_parse_contexts: list[dict | None] = []

    def draft(self, question, evidence, attempt):
        del question, evidence, attempt
        item = self.drafts[min(self.draft_calls, len(self.drafts) - 1)]
        self.draft_calls += 1
        if isinstance(item, Exception):
            raise item
        return item

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
        del question, evidence, previous_payload, citation_issues, unsupported_claims, attempt
        self.correction_parse_contexts.append(parse_error_context)
        item = self.corrections[min(self.correct_calls, len(self.corrections) - 1)]
        self.correct_calls += 1
        if isinstance(item, Exception):
            raise item
        return item


class ScriptedRewriter:
    def rewrite(self, question: str, attempt: int, query_history: list[str]) -> str:
        del query_history
        return f"{question} retry-{attempt}"


def _chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(text=text, source="doc://langgraph", metadata={})


def _claim_payload(claim_text: str, citation_ids: list[str]) -> GenerationResponse:
    return GenerationResponse(
        payload={
            "summary": "Draft",
            "claims": [
                {
                    "claim_id": "c1",
                    "text": claim_text,
                    "citation_ids": citation_ids,
                }
            ],
        }
    )


def test_successful_retrieval_and_answer() -> None:
    retriever = SequenceRetriever([RetrievalResponse(chunks=[_chunk("durable stateful agents")])])
    generator = ScriptedGenerator(
        drafts=[
            _claim_payload(
                "LangGraph supports durable stateful workflows",
                ["source-01#chunk-001"],
            )
        ],
        corrections=[],
    )
    result = ResearchAssistant(retriever, generator).run("What does LangGraph support?")
    assert result.status == TerminalStatus.ANSWERED
    assert result.report is not None
    assert result.generation_attempts == 1
    assert result.total_graph_steps > 0


def test_empty_retrieval_leads_to_insufficient_evidence() -> None:
    retriever = SequenceRetriever([RetrievalResponse(chunks=[]), RetrievalResponse(chunks=[])])
    generator = ScriptedGenerator(drafts=[], corrections=[])
    assistant = ResearchAssistant(
        retriever,
        generator,
        rewriter=ScriptedRewriter(),
        config=RuntimeConfig(max_retrieval_attempts=2, max_correction_attempts=2),
    )
    result = assistant.run("Unanswerable question")
    assert result.status == TerminalStatus.INSUFFICIENT_EVIDENCE
    assert result.query_rewrites == 1


def test_malformed_output_first_goes_to_correction() -> None:
    retriever = SequenceRetriever([RetrievalResponse(chunks=[_chunk("stateful durable agents")])])
    generator = ScriptedGenerator(
        drafts=[GenerationResponse(payload={"summary": "missing claims"})],
        corrections=[
            _claim_payload(
                "stateful durable agents",
                ["source-01#chunk-001"],
            )
        ],
    )
    assistant = ResearchAssistant(
        retriever,
        generator,
        config=RuntimeConfig(max_correction_attempts=2),
    )
    result = assistant.run("Question")
    assert result.status == TerminalStatus.ANSWERED
    assert generator.draft_calls == 1
    assert generator.correct_calls == 1
    assert generator.correction_parse_contexts[0] is not None
    assert "error" in generator.correction_parse_contexts[0]


def test_provider_exception_is_model_error() -> None:
    retriever = SequenceRetriever([RetrievalResponse(chunks=[_chunk("stateful durable agents")])])
    generator = ScriptedGenerator(drafts=[RuntimeError("provider timeout")], corrections=[])
    assistant = ResearchAssistant(retriever, generator)
    result = assistant.run("Question")
    assert result.status == TerminalStatus.MODEL_ERROR
    assert any("model_error" in err for err in result.errors)


def test_citation_validation_detects_missing_and_unsupported() -> None:
    report = ResearchReport(
        question="q",
        summary="s",
        claims=[
            Claim(
                claim_id="c1", text="Kubernetes autoscaling", citation_ids=["source-01#chunk-001"]
            ),
            Claim(claim_id="c2", text="Missing citation claim", citation_ids=[]),
            Claim(claim_id="c3", text="Unknown citation", citation_ids=["missing-id"]),
        ],
    )
    evidence = [
        EvidenceChunk(
            evidence_id="source-01#chunk-001",
            source_id="source-01",
            chunk_id=1,
            text="LangGraph durable agents",
            metadata={},
        )
    ]
    issues = validate_citations(report, evidence)
    reasons = {issue.reason for issue in issues}
    assert "claim has no citations" in reasons
    assert "citation id not found in retrieved evidence" in reasons
    assert "no cited passage lexically supports claim" in reasons


def test_citation_validation_handles_multiple_citations_and_normalization() -> None:
    report = ResearchReport(
        question="q",
        summary="s",
        claims=[
            Claim(
                claim_id="c1",
                text="LangGraph supports STATEFUL workflows.",
                citation_ids=["source-01#chunk-001", "source-01#chunk-002"],
            )
        ],
    )
    evidence = [
        EvidenceChunk(
            evidence_id="source-01#chunk-001",
            source_id="source-01",
            chunk_id=1,
            text="Stateful workflows are common.",
            metadata={},
        ),
        EvidenceChunk(
            evidence_id="source-01#chunk-002",
            source_id="source-01",
            chunk_id=2,
            text="LANGGRAPH supports stateful workflow patterns!",
            metadata={},
        ),
    ]
    issues = validate_citations(report, evidence)
    assert issues == []


def test_graph_step_exhaustion_terminates() -> None:
    retriever = SequenceRetriever([RetrievalResponse(chunks=[])])
    generator = ScriptedGenerator(drafts=[], corrections=[])
    assistant = ResearchAssistant(
        retriever,
        generator,
        rewriter=ScriptedRewriter(),
        config=RuntimeConfig(max_graph_steps=2, max_retrieval_attempts=10),
    )
    result = assistant.run("Need retries")
    assert result.status == TerminalStatus.RETRY_LIMIT_EXCEEDED
    assert any("retry_limit_exceeded" in err for err in result.errors)


def test_correction_failure_reports_validation_failed() -> None:
    retriever = SequenceRetriever([RetrievalResponse(chunks=[_chunk("langgraph only")])])
    bad = _claim_payload("kubernetes autoscaling", ["source-01#chunk-001"])
    generator = ScriptedGenerator(drafts=[bad], corrections=[bad, bad])
    assistant = ResearchAssistant(
        retriever, generator, config=RuntimeConfig(max_correction_attempts=1)
    )
    result = assistant.run("Question")
    assert result.status == TerminalStatus.CITATION_VALIDATION_FAILED


def test_event_and_result_serialization_and_metadata_normalization() -> None:
    retriever = SequenceRetriever(
        [
            RetrievalResponse(
                chunks=[_chunk("stateful workflows")],
                usage={"provider": "fake", "latency_ms": "8.5", "prompt_tokens": "bad"},
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
                usage={"provider": "fake", "total_tokens": "12", "cost_usd": "0.001"},
            )
        ],
        corrections=[],
    )
    result = ResearchAssistant(retriever, generator).run("Question")
    serialized = result.to_dict()
    assert serialized["status"] == "answered"
    assert serialized["events"][0]["run_id"]
    assert serialized["events"][0]["timestamp"]
    assert serialized["metrics"]["usage"][0]["prompt_tokens"] is None
    assert serialized["metrics"]["usage"][0]["latency_ms"] == 8.5
