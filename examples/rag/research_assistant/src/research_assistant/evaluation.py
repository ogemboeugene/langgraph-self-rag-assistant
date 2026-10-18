from __future__ import annotations

from dataclasses import dataclass

from .graph import ResearchAssistant, RunResult


@dataclass(frozen=True)
class ComparisonResult:
    baseline_status: str
    assistant_status: str
    baseline_claims: int
    assistant_claims: int


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
