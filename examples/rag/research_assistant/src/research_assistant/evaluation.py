from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from .graph import ResearchAssistant, RunResult


@dataclass(frozen=True)
class ComparisonResult:
    baseline_status: str
    assistant_status: str
    baseline_claims: int
    assistant_claims: int


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    question: str
    expected_status: str
    expected_min_retrieval_attempts: int = 1


@dataclass(frozen=True)
class EvaluationMetrics:
    total_cases: int
    answered_cases: int
    citation_valid_cases: int
    supported_claim_rate: float
    avg_retrieval_attempts: float
    avg_correction_attempts: float
    avg_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "answered_cases": self.answered_cases,
            "citation_valid_cases": self.citation_valid_cases,
            "supported_claim_rate": self.supported_claim_rate,
            "avg_retrieval_attempts": self.avg_retrieval_attempts,
            "avg_correction_attempts": self.avg_correction_attempts,
            "avg_latency_ms": self.avg_latency_ms,
        }


@dataclass(frozen=True)
class EvaluationSummary:
    metrics: EvaluationMetrics
    runs: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics.to_dict(),
            "runs": [dict(run) for run in self.runs],
        }


def compare_workflows(
    question: str, baseline: ResearchAssistant, assistant: ResearchAssistant
) -> ComparisonResult:
    baseline_result: RunResult = baseline.run(question)
    assistant_result: RunResult = assistant.run(question)
    return ComparisonResult(
        baseline_status=baseline_result.status.value,
        assistant_status=assistant_result.status.value,
        baseline_claims=len(baseline_result.report.claims if baseline_result.report else []),
        assistant_claims=len(assistant_result.report.claims if assistant_result.report else []),
    )


def load_fixture(path: str | Path) -> list[EvaluationCase]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    return [
        EvaluationCase(
            case_id=str(item["case_id"]),
            question=str(item["question"]),
            expected_status=str(item["expected_status"]),
            expected_min_retrieval_attempts=int(item.get("expected_min_retrieval_attempts", 1)),
        )
        for item in cases
    ]


def evaluate_cases(assistant: ResearchAssistant, cases: list[EvaluationCase]) -> EvaluationSummary:
    runs: list[dict[str, Any]] = []
    retrieval_attempts: list[int] = []
    correction_attempts: list[int] = []
    latencies_ms: list[float] = []

    for case in cases:
        result = assistant.run(case.question)
        retrieval_attempts.append(result.retrieval_attempts)
        correction_attempts.append(result.correction_attempts)
        latencies_ms.append(result.metrics.elapsed_ms)
        supported_claims = 0
        total_claims = 0
        for event in result.events:
            if event.event_type != "report.validation_failed":
                continue
            total_claims += len(event.details.get("unsupported_claims", []))
        if result.report:
            total_claims = len(result.report.claims)
            unsupported_ids: set[str] = set()
            for event in result.events:
                if event.event_type != "report.validation_failed":
                    continue
                for unsupported in event.details.get("unsupported_claims", []):
                    claim_id = unsupported.get("claim_id")
                    if claim_id:
                        unsupported_ids.add(str(claim_id))
            supported_claims = max(total_claims - len(unsupported_ids), 0)

        runs.append(
            {
                "case_id": case.case_id,
                "question": case.question,
                "expected_status": case.expected_status,
                "status": result.status.value,
                "status_match": result.status.value == case.expected_status,
                "retrieval_attempts": result.retrieval_attempts,
                "expected_min_retrieval_attempts": case.expected_min_retrieval_attempts,
                "retrieval_attempt_expectation_met": (
                    result.retrieval_attempts >= case.expected_min_retrieval_attempts
                ),
                "correction_attempts": result.correction_attempts,
                "latency_ms": result.metrics.elapsed_ms,
                "supported_claims": supported_claims,
                "total_claims": total_claims,
                "citation_valid": result.status.value == "answered",
            }
        )

    total_claims = sum(run["total_claims"] for run in runs)
    supported_claims = sum(run["supported_claims"] for run in runs)
    metrics = EvaluationMetrics(
        total_cases=len(cases),
        answered_cases=sum(1 for run in runs if run["status"] == "answered"),
        citation_valid_cases=sum(1 for run in runs if run["citation_valid"]),
        supported_claim_rate=(supported_claims / total_claims if total_claims else 0.0),
        avg_retrieval_attempts=mean(retrieval_attempts) if retrieval_attempts else 0.0,
        avg_correction_attempts=mean(correction_attempts) if correction_attempts else 0.0,
        avg_latency_ms=mean(latencies_ms) if latencies_ms else 0.0,
    )
    return EvaluationSummary(metrics=metrics, runs=runs)
