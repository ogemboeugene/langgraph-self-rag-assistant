from __future__ import annotations

import inspect
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from .config import RuntimeConfig
from .generation import (
    ReportGenerator,
    StructuredOutputError,
    detect_unsupported_claims,
    parse_report_payload,
    validate_citations,
)
from .models import CitationIssue, EvidenceChunk, ResearchReport, TerminalStatus, UnsupportedClaim
from .observability import EventLog, Metrics
from .retrieval import QueryRewriter, Retriever, normalize_chunks

try:
    from langgraph.graph import END, START, StateGraph
except ModuleNotFoundError:  # pragma: no cover - exercised only in non-monorepo envs
    repo_root = Path(__file__).resolve().parents[5]
    local_langgraph = repo_root / "libs" / "langgraph"
    if local_langgraph.exists():
        sys.path.insert(0, str(local_langgraph))
    from langgraph.graph import END, START, StateGraph


@dataclass(frozen=True)
class RunResult:
    status: TerminalStatus
    report: ResearchReport | None
    query_history: list[str]
    retrieval_attempts: int
    query_rewrites: int
    generation_attempts: int
    correction_attempts: int
    total_graph_steps: int
    events: list
    metrics: Metrics
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "report": {
                "question": self.report.question,
                "summary": self.report.summary,
                "claims": [
                    {
                        "claim_id": claim.claim_id,
                        "text": claim.text,
                        "citation_ids": list(claim.citation_ids),
                    }
                    for claim in (self.report.claims if self.report else [])
                ],
            }
            if self.report
            else None,
            "query_history": list(self.query_history),
            "retrieval_attempts": self.retrieval_attempts,
            "query_rewrites": self.query_rewrites,
            "generation_attempts": self.generation_attempts,
            "correction_attempts": self.correction_attempts,
            "total_graph_steps": self.total_graph_steps,
            "events": [event.to_dict() for event in self.events],
            "metrics": self.metrics.to_dict(),
            "errors": list(self.errors),
        }


class IdentityRewriter:
    def rewrite(self, question: str, attempt: int, query_history: list[str]) -> str:
        return f"{question} (retry {attempt})"


class AssistantState(TypedDict):
    question: str
    current_query: str
    query_history: list[str]
    retrieval_attempts: int
    query_rewrites: int
    generation_attempts: int
    correction_attempts: int
    total_graph_steps: int
    evidence: list[EvidenceChunk]
    payload: dict[str, Any] | None
    report: ResearchReport | None
    citation_issues: list[CitationIssue]
    unsupported_claims: list[UnsupportedClaim]
    structured_output_errors: list[str]
    parse_error_context: dict[str, Any] | None
    terminal_status: TerminalStatus | None
    terminal_reason: str | None
    events: EventLog
    metrics: Metrics
    errors: list[str]


class ResearchAssistant:
    def __init__(
        self,
        retriever: Retriever,
        generator: ReportGenerator,
        config: RuntimeConfig | None = None,
        rewriter: QueryRewriter | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.config = config or RuntimeConfig()
        self.rewriter = rewriter or IdentityRewriter()
        self._graph = self._build_graph()

    def run(self, question: str) -> RunResult:
        state: AssistantState = {
            "question": question,
            "current_query": question,
            "query_history": [question],
            "retrieval_attempts": 0,
            "query_rewrites": 0,
            "generation_attempts": 0,
            "correction_attempts": 0,
            "total_graph_steps": 0,
            "evidence": [],
            "payload": None,
            "report": None,
            "citation_issues": [],
            "unsupported_claims": [],
            "structured_output_errors": [],
            "parse_error_context": None,
            "terminal_status": None,
            "terminal_reason": None,
            "events": EventLog(),
            "metrics": Metrics(),
            "errors": [],
        }
        final_state = self._graph.invoke(state)
        final_state["metrics"].complete()
        status = final_state["terminal_status"] or TerminalStatus.RETRY_LIMIT_EXCEEDED
        if final_state["terminal_status"] is None:
            final_state["errors"].append(
                "retry_limit_exceeded: graph terminated without terminal status"
            )
        return RunResult(
            status=status,
            report=final_state["report"],
            query_history=final_state["query_history"],
            retrieval_attempts=final_state["retrieval_attempts"],
            query_rewrites=final_state["query_rewrites"],
            generation_attempts=final_state["generation_attempts"],
            correction_attempts=final_state["correction_attempts"],
            total_graph_steps=final_state["total_graph_steps"],
            events=final_state["events"].events,
            metrics=final_state["metrics"],
            errors=final_state["errors"],
        )

    def _build_graph(self):
        builder = StateGraph(AssistantState)
        builder.add_node("validate_input", self._validate_input)
        builder.add_node("retrieve_evidence", self._retrieve_evidence)
        builder.add_node("assess_evidence", self._assess_evidence)
        builder.add_node("rewrite_query", self._rewrite_query)
        builder.add_node("draft_report", self._draft_report)
        builder.add_node("validate_report", self._validate_report)
        builder.add_node("correct_report", self._correct_report)
        builder.add_node("terminate", self._terminate)

        builder.add_edge(START, "validate_input")
        builder.add_conditional_edges(
            "validate_input",
            self._route_after_validate_input,
            {
                "retrieve_evidence": "retrieve_evidence",
                "terminate": "terminate",
            },
        )
        builder.add_edge("retrieve_evidence", "assess_evidence")
        builder.add_conditional_edges(
            "assess_evidence",
            self._route_after_assess_evidence,
            {
                "draft_report": "draft_report",
                "rewrite_query": "rewrite_query",
                "terminate": "terminate",
            },
        )
        builder.add_edge("rewrite_query", "retrieve_evidence")
        builder.add_conditional_edges(
            "draft_report",
            self._route_after_draft,
            {
                "validate_report": "validate_report",
                "correct_report": "correct_report",
                "terminate": "terminate",
            },
        )
        builder.add_conditional_edges(
            "validate_report",
            self._route_after_validate_report,
            {
                "terminate": "terminate",
                "correct_report": "correct_report",
            },
        )
        builder.add_conditional_edges(
            "correct_report",
            self._route_after_correct,
            {
                "validate_report": "validate_report",
                "correct_report": "correct_report",
                "terminate": "terminate",
            },
        )
        builder.add_edge("terminate", END)
        return builder.compile()

    def _mark_step(self, state: AssistantState, stage: str) -> bool:
        state["total_graph_steps"] += 1
        if state["total_graph_steps"] > self.config.max_graph_steps:
            state["terminal_status"] = TerminalStatus.RETRY_LIMIT_EXCEEDED
            state["terminal_reason"] = (
                f"max_graph_steps exceeded at stage '{stage}' ({self.config.max_graph_steps})"
            )
            state["errors"].append(f"retry_limit_exceeded: {state['terminal_reason']}")
            state["events"].record(
                "run.retry_limit_exceeded",
                stage=stage,
                status="error",
                error_type="graph_step_limit",
                limit=self.config.max_graph_steps,
            )
            return False
        return True

    def _validate_input(self, state: AssistantState) -> AssistantState:
        if not self._mark_step(state, "validate_input"):
            return state
        question = state["question"]
        if not question.strip():
            state["terminal_status"] = TerminalStatus.INSUFFICIENT_EVIDENCE
            state["terminal_reason"] = "Question cannot be empty"
            state["errors"].append("validation_error: Question cannot be empty")
            state["events"].record(
                "input.invalid",
                stage="validate_input",
                status="error",
                error_type="validation_error",
            )
            return state
        state["events"].record("input.valid", stage="validate_input", status="ok")
        return state

    def _retrieve_evidence(self, state: AssistantState) -> AssistantState:
        if not self._mark_step(state, "retrieve_evidence"):
            return state
        state["retrieval_attempts"] += 1
        attempt = state["retrieval_attempts"]
        started = time.perf_counter()
        state["events"].record(
            "retrieve.start",
            stage="retrieve_evidence",
            status="started",
            attempt=attempt,
            query=state["current_query"],
        )
        try:
            retrieval = self.retriever.retrieve(
                query=state["current_query"],
                limit=self.config.max_retrieved_chunks,
            )
        except Exception as exc:  # noqa: BLE001
            duration_ms = (time.perf_counter() - started) * 1000
            message = f"retriever_error: {exc}"
            state["errors"].append(message)
            state["events"].record(
                "retrieve.error",
                stage="retrieve_evidence",
                status="error",
                attempt=attempt,
                duration_ms=duration_ms,
                error_type=type(exc).__name__,
                message=str(exc),
            )
            state["evidence"] = []
            return state

        duration_ms = (time.perf_counter() - started) * 1000
        state["metrics"].add_usage(retrieval.usage)
        state["evidence"] = normalize_chunks(retrieval.chunks)
        state["events"].record(
            "retrieve.complete",
            stage="retrieve_evidence",
            status="ok",
            attempt=attempt,
            duration_ms=duration_ms,
            evidence_count=len(state["evidence"]),
        )
        return state

    def _assess_evidence(self, state: AssistantState) -> AssistantState:
        if not self._mark_step(state, "assess_evidence"):
            return state
        evidence_count = len(state["evidence"])
        if evidence_count >= self.config.min_evidence_chunks:
            state["events"].record(
                "evidence.sufficient",
                stage="assess_evidence",
                status="ok",
                evidence_count=evidence_count,
            )
            return state

        if state["retrieval_attempts"] >= self.config.max_retrieval_attempts:
            state["terminal_status"] = TerminalStatus.INSUFFICIENT_EVIDENCE
            state["terminal_reason"] = (
                f"retrieval_attempt_limit_reached: {self.config.max_retrieval_attempts}"
            )
            state["errors"].append(state["terminal_reason"])
            state["events"].record(
                "evidence.insufficient",
                stage="assess_evidence",
                status="error",
                attempt=state["retrieval_attempts"],
                error_type="retrieval_attempt_limit",
                evidence_count=evidence_count,
            )
            return state

        state["events"].record(
            "evidence.insufficient",
            stage="assess_evidence",
            status="retry",
            attempt=state["retrieval_attempts"],
            evidence_count=evidence_count,
        )
        return state

    def _rewrite_query(self, state: AssistantState) -> AssistantState:
        if not self._mark_step(state, "rewrite_query"):
            return state
        next_attempt = state["query_rewrites"] + 1
        rewritten = self.rewriter.rewrite(state["question"], next_attempt, state["query_history"])
        state["query_rewrites"] = next_attempt
        state["current_query"] = rewritten
        state["query_history"].append(rewritten)
        state["events"].record(
            "query.rewritten",
            stage="rewrite_query",
            status="ok",
            attempt=next_attempt,
            query=rewritten,
        )
        return state

    def _draft_report(self, state: AssistantState) -> AssistantState:
        if not self._mark_step(state, "draft_report"):
            return state
        self._run_generation(state, is_correction=False)
        return state

    def _correct_report(self, state: AssistantState) -> AssistantState:
        if not self._mark_step(state, "correct_report"):
            return state
        self._run_generation(state, is_correction=True)
        return state

    def _run_generation(self, state: AssistantState, *, is_correction: bool) -> None:
        stage = "correct_report" if is_correction else "draft_report"
        state["generation_attempts"] += 1
        generation_attempt = state["generation_attempts"]
        correction_attempt = state["correction_attempts"] + 1 if is_correction else None
        if is_correction:
            state["correction_attempts"] += 1

        started = time.perf_counter()
        state["events"].record(
            "generation.start",
            stage=stage,
            status="started",
            attempt=correction_attempt or generation_attempt,
            generation_attempt=generation_attempt,
            correction_attempt=state["correction_attempts"],
        )
        try:
            if is_correction:
                generation = self._call_correct(state)
            else:
                generation = self.generator.draft(
                    question=state["question"],
                    evidence=state["evidence"],
                    attempt=generation_attempt,
                )
            state["metrics"].add_usage(generation.usage)
            state["payload"] = generation.payload
            state["report"] = parse_report_payload(state["question"], generation.payload)
            state["parse_error_context"] = None
            duration_ms = (time.perf_counter() - started) * 1000
            state["events"].record(
                "generation.complete",
                stage=stage,
                status="ok",
                duration_ms=duration_ms,
                attempt=correction_attempt or generation_attempt,
                generation_attempt=generation_attempt,
            )
            return
        except StructuredOutputError as exc:
            duration_ms = (time.perf_counter() - started) * 1000
            message = f"structured_output_error: {exc}"
            state["errors"].append(message)
            state["structured_output_errors"].append(str(exc))
            state["parse_error_context"] = {
                "error": str(exc),
                "stage": stage,
                "generation_attempt": generation_attempt,
                "correction_attempt": state["correction_attempts"],
                "payload": state["payload"],
            }
            state["events"].record(
                "generation.malformed",
                stage=stage,
                status="error",
                duration_ms=duration_ms,
                attempt=correction_attempt or generation_attempt,
                generation_attempt=generation_attempt,
                error_type=type(exc).__name__,
                message=str(exc),
            )
            if state["correction_attempts"] >= self.config.max_correction_attempts:
                state["terminal_status"] = TerminalStatus.MODEL_ERROR
                state["terminal_reason"] = (
                    "malformed_output_correction_limit_reached: "
                    f"{self.config.max_correction_attempts}"
                )
                state["errors"].append(state["terminal_reason"])
            return
        except Exception as exc:  # noqa: BLE001
            duration_ms = (time.perf_counter() - started) * 1000
            message = f"model_error: {exc}"
            state["errors"].append(message)
            state["events"].record(
                "generation.error",
                stage=stage,
                status="error",
                duration_ms=duration_ms,
                attempt=correction_attempt or generation_attempt,
                generation_attempt=generation_attempt,
                error_type=type(exc).__name__,
                message=str(exc),
            )
            state["terminal_status"] = TerminalStatus.MODEL_ERROR
            state["terminal_reason"] = f"provider_exception: {type(exc).__name__}"
            return

    def _call_correct(self, state: AssistantState):
        kwargs = {
            "question": state["question"],
            "evidence": state["evidence"],
            "previous_payload": state["payload"] or {},
            "citation_issues": state["citation_issues"],
            "unsupported_claims": state["unsupported_claims"],
            "attempt": state["correction_attempts"],
        }
        signature = inspect.signature(self.generator.correct)
        accepts_context = "parse_error_context" in signature.parameters or any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
        )
        if accepts_context:
            kwargs["parse_error_context"] = state["parse_error_context"]
        return self.generator.correct(**kwargs)

    def _validate_report(self, state: AssistantState) -> AssistantState:
        if not self._mark_step(state, "validate_report"):
            return state
        report = state["report"]
        if report is None:
            state["terminal_status"] = TerminalStatus.MODEL_ERROR
            state["terminal_reason"] = "missing_report_after_generation"
            state["errors"].append(state["terminal_reason"])
            return state

        state["citation_issues"] = validate_citations(report, state["evidence"])
        state["unsupported_claims"] = detect_unsupported_claims(report, state["evidence"])
        if not state["citation_issues"] and not state["unsupported_claims"]:
            state["terminal_status"] = TerminalStatus.ANSWERED
            state["terminal_reason"] = "answered"
            state["events"].record(
                "report.valid",
                stage="validate_report",
                status="ok",
                claim_count=len(report.claims),
            )
            return state

        if state["correction_attempts"] >= self.config.max_correction_attempts:
            state["terminal_status"] = TerminalStatus.CITATION_VALIDATION_FAILED
            state["terminal_reason"] = (
                f"correction_attempt_limit_reached: {self.config.max_correction_attempts}"
            )
            state["errors"].append(state["terminal_reason"])

        state["events"].record(
            "report.validation_failed",
            stage="validate_report",
            status="error",
            attempt=state["correction_attempts"],
            citation_issues=[issue.to_dict() for issue in state["citation_issues"]],
            unsupported_claims=[
                {"claim_id": item.claim_id, "reason": item.reason}
                for item in state["unsupported_claims"]
            ],
        )
        return state

    def _terminate(self, state: AssistantState) -> AssistantState:
        if state["terminal_status"] is None:
            state["terminal_status"] = TerminalStatus.RETRY_LIMIT_EXCEEDED
            state["terminal_reason"] = "route_reached_terminate_without_status"
            state["errors"].append(state["terminal_reason"])
        state["events"].record(
            "run.terminated",
            stage="terminate",
            status=state["terminal_status"].value,
            terminal_reason=state["terminal_reason"],
            retrieval_attempts=state["retrieval_attempts"],
            query_rewrites=state["query_rewrites"],
            generation_attempts=state["generation_attempts"],
            correction_attempts=state["correction_attempts"],
            total_graph_steps=state["total_graph_steps"],
        )
        return state

    def _route_after_validate_input(self, state: AssistantState) -> str:
        if state["terminal_status"] is not None:
            return "terminate"
        return "retrieve_evidence"

    def _route_after_assess_evidence(self, state: AssistantState) -> str:
        if state["terminal_status"] is not None:
            return "terminate"
        if len(state["evidence"]) >= self.config.min_evidence_chunks:
            return "draft_report"
        return "rewrite_query"

    def _route_after_draft(self, state: AssistantState) -> str:
        if state["terminal_status"] is not None:
            return "terminate"
        if state["report"] is not None:
            return "validate_report"
        return "correct_report"

    def _route_after_validate_report(self, state: AssistantState) -> str:
        if state["terminal_status"] is TerminalStatus.ANSWERED:
            return "terminate"
        if state["terminal_status"] is TerminalStatus.CITATION_VALIDATION_FAILED:
            return "terminate"
        if state["terminal_status"] is TerminalStatus.MODEL_ERROR:
            return "terminate"
        return "correct_report"

    def _route_after_correct(self, state: AssistantState) -> str:
        if state["terminal_status"] is not None:
            return "terminate"
        if state["report"] is not None:
            return "validate_report"
        if state["correction_attempts"] >= self.config.max_correction_attempts:
            state["terminal_status"] = TerminalStatus.MODEL_ERROR
            state["terminal_reason"] = (
                f"correction_attempt_limit_reached: {self.config.max_correction_attempts}"
            )
            state["errors"].append(state["terminal_reason"])
            return "terminate"
        return "correct_report"
