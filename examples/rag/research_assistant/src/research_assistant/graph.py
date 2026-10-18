from __future__ import annotations

from dataclasses import dataclass

from .config import RuntimeConfig
from .generation import (
    ReportGenerator,
    StructuredOutputError,
    detect_unsupported_claims,
    parse_report_payload,
    validate_citations,
)
from .models import ResearchReport, TerminalStatus
from .observability import EventLog, Metrics
from .retrieval import QueryRewriter, Retriever, normalize_chunks


@dataclass(frozen=True)
class RunResult:
    status: TerminalStatus
    report: ResearchReport | None
    query_history: list[str]
    retrieval_attempts: int
    correction_attempts: int
    events: list
    metrics: Metrics
    errors: list[str]


class IdentityRewriter:
    def rewrite(self, question: str, attempt: int, query_history: list[str]) -> str:
        return f"{question} (retry {attempt})"


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

    def run(self, question: str) -> RunResult:
        events = EventLog()
        metrics = Metrics()
        errors: list[str] = []
        report: ResearchReport | None = None
        query_history = [question]
        current_query = question
        retrieval_attempts = 0
        correction_attempts = 0
        step_count = 0
        payload: dict | None = None
        evidence = []
        citation_issues = []
        unsupported_claims = []

        if not question.strip():
            metrics.complete()
            return RunResult(
                status=TerminalStatus.INSUFFICIENT_EVIDENCE,
                report=None,
                query_history=query_history,
                retrieval_attempts=0,
                correction_attempts=0,
                events=events.events,
                metrics=metrics,
                errors=["Question cannot be empty"],
            )

        while step_count < self.config.max_graph_steps:
            step_count += 1
            if not evidence:
                retrieval_attempts += 1
                events.record(
                    "retrieve.start",
                    query=current_query,
                    attempt=retrieval_attempts,
                )
                try:
                    retrieval = self.retriever.retrieve(
                        query=current_query,
                        limit=self.config.max_retrieved_chunks,
                    )
                    metrics.add_usage(retrieval.usage)
                    evidence = normalize_chunks(retrieval.chunks)
                except Exception as exc:  # noqa: BLE001
                    message = f"retriever_error: {exc}"
                    errors.append(message)
                    events.record("retrieve.error", error=str(exc), attempt=retrieval_attempts)
                    if retrieval_attempts >= self.config.max_retrieval_attempts:
                        break
                    current_query = self.rewriter.rewrite(
                        question, retrieval_attempts, query_history
                    )
                    query_history.append(current_query)
                    continue

                events.record("retrieve.complete", evidence_count=len(evidence))
                if len(evidence) < self.config.min_evidence_chunks:
                    events.record("retrieve.insufficient", evidence_count=len(evidence))
                    evidence = []
                    if retrieval_attempts >= self.config.max_retrieval_attempts:
                        metrics.complete()
                        return RunResult(
                            status=TerminalStatus.INSUFFICIENT_EVIDENCE,
                            report=None,
                            query_history=query_history,
                            retrieval_attempts=retrieval_attempts,
                            correction_attempts=correction_attempts,
                            events=events.events,
                            metrics=metrics,
                            errors=errors,
                        )
                    current_query = self.rewriter.rewrite(
                        question, retrieval_attempts, query_history
                    )
                    query_history.append(current_query)
                    continue

            generation_attempt = correction_attempts + 1
            events.record("generation.start", attempt=generation_attempt)
            try:
                if payload is None:
                    generation = self.generator.draft(
                        question=question,
                        evidence=evidence,
                        attempt=generation_attempt,
                    )
                else:
                    generation = self.generator.correct(
                        question=question,
                        evidence=evidence,
                        previous_payload=payload,
                        citation_issues=citation_issues,
                        unsupported_claims=unsupported_claims,
                        attempt=generation_attempt,
                    )
                metrics.add_usage(generation.usage)
                payload = generation.payload
                report = parse_report_payload(question, payload)
            except StructuredOutputError as exc:
                message = f"structured_output_error: {exc}"
                errors.append(message)
                events.record("generation.malformed", error=str(exc), attempt=generation_attempt)
                correction_attempts += 1
                if correction_attempts > self.config.max_correction_attempts:
                    metrics.complete()
                    return RunResult(
                        status=TerminalStatus.MODEL_ERROR,
                        report=None,
                        query_history=query_history,
                        retrieval_attempts=retrieval_attempts,
                        correction_attempts=correction_attempts,
                        events=events.events,
                        metrics=metrics,
                        errors=errors,
                    )
                continue
            except Exception as exc:  # noqa: BLE001
                message = f"model_error: {exc}"
                errors.append(message)
                events.record("generation.error", error=str(exc), attempt=generation_attempt)
                correction_attempts += 1
                if correction_attempts > self.config.max_correction_attempts:
                    metrics.complete()
                    return RunResult(
                        status=TerminalStatus.MODEL_ERROR,
                        report=None,
                        query_history=query_history,
                        retrieval_attempts=retrieval_attempts,
                        correction_attempts=correction_attempts,
                        events=events.events,
                        metrics=metrics,
                        errors=errors,
                    )
                continue

            citation_issues = validate_citations(report, evidence)
            unsupported_claims = detect_unsupported_claims(report, evidence)
            if not citation_issues and not unsupported_claims:
                events.record("report.answered", claim_count=len(report.claims))
                metrics.complete()
                return RunResult(
                    status=TerminalStatus.ANSWERED,
                    report=report,
                    query_history=query_history,
                    retrieval_attempts=retrieval_attempts,
                    correction_attempts=correction_attempts,
                    events=events.events,
                    metrics=metrics,
                    errors=errors,
                )

            events.record(
                "report.validation_failed",
                citation_issue_count=len(citation_issues),
                unsupported_claim_count=len(unsupported_claims),
            )
            correction_attempts += 1
            if correction_attempts > self.config.max_correction_attempts:
                metrics.complete()
                return RunResult(
                    status=TerminalStatus.CITATION_VALIDATION_FAILED,
                    report=report,
                    query_history=query_history,
                    retrieval_attempts=retrieval_attempts,
                    correction_attempts=correction_attempts,
                    events=events.events,
                    metrics=metrics,
                    errors=errors,
                )

        metrics.complete()
        status = TerminalStatus.RETRY_LIMIT_EXCEEDED
        events.record("run.terminated", status=status.value)
        return RunResult(
            status=status,
            report=report,
            query_history=query_history,
            retrieval_attempts=retrieval_attempts,
            correction_attempts=correction_attempts,
            events=events.events,
            metrics=metrics,
            errors=errors,
        )
