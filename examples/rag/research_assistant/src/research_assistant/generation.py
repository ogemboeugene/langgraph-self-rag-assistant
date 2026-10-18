from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from .models import CitationIssue, Claim, EvidenceChunk, ResearchReport, UnsupportedClaim


@dataclass(frozen=True)
class GenerationResponse:
    payload: dict[str, Any]
    usage: dict[str, Any] | None = None


class ReportGenerator(Protocol):
    def draft(
        self, question: str, evidence: list[EvidenceChunk], attempt: int
    ) -> GenerationResponse: ...

    def correct(
        self,
        question: str,
        evidence: list[EvidenceChunk],
        previous_payload: dict[str, Any],
        citation_issues: list[CitationIssue],
        unsupported_claims: list[UnsupportedClaim],
        attempt: int,
        parse_error_context: dict[str, Any] | None = None,
    ) -> GenerationResponse: ...


class StructuredOutputError(ValueError):
    pass


TOKEN_RE = re.compile(r"[a-z0-9]+")


def _normalize_token(token: str) -> str:
    lowered = token.lower().strip()
    if lowered.endswith("ies") and len(lowered) > 4:
        return f"{lowered[:-3]}y"
    if lowered.endswith("es") and len(lowered) > 4:
        return lowered[:-2]
    if lowered.endswith("s") and len(lowered) > 3:
        return lowered[:-1]
    return lowered


def _normalize_text_tokens(text: str) -> list[str]:
    return [_normalize_token(token) for token in TOKEN_RE.findall(text.lower())]


def parse_report_payload(question: str, payload: dict[str, Any]) -> ResearchReport:
    summary = payload.get("summary")
    claims_raw = payload.get("claims")

    if not isinstance(summary, str) or not summary.strip():
        msg = "Generator output must contain a non-empty 'summary' string"
        raise StructuredOutputError(msg)
    if not isinstance(claims_raw, list):
        msg = "Generator output must contain a 'claims' list"
        raise StructuredOutputError(msg)

    claims: list[Claim] = []
    for idx, item in enumerate(claims_raw, start=1):
        if not isinstance(item, dict):
            msg = f"Claim at index {idx} must be an object"
            raise StructuredOutputError(msg)
        text = item.get("text")
        citation_ids = item.get("citation_ids")
        claim_id = str(item.get("claim_id") or f"claim-{idx:02d}")
        if not isinstance(text, str) or not text.strip():
            msg = f"Claim {claim_id} must include non-empty text"
            raise StructuredOutputError(msg)
        if not isinstance(citation_ids, list) or not all(isinstance(c, str) for c in citation_ids):
            msg = f"Claim {claim_id} must include citation_ids as list[str]"
            raise StructuredOutputError(msg)
        claims.append(Claim(claim_id=claim_id, text=text.strip(), citation_ids=citation_ids))

    return ResearchReport(question=question, summary=summary.strip(), claims=claims)


def validate_citations(
    report: ResearchReport, evidence: list[EvidenceChunk]
) -> list[CitationIssue]:
    evidence_by_id = {item.evidence_id: item for item in evidence}
    normalized_passages = {
        item.evidence_id: set(_normalize_text_tokens(item.text)) for item in evidence
    }
    issues: list[CitationIssue] = []
    for claim in report.claims:
        claim_tokens = {token for token in _normalize_text_tokens(claim.text) if len(token) > 3}
        if not claim.citation_ids:
            issues.append(
                CitationIssue(
                    claim_id=claim.claim_id,
                    citation_id="",
                    reason="claim has no citations",
                    support_score=0.0,
                )
            )
            continue
        claim_supported = False
        for citation_id in claim.citation_ids:
            if citation_id not in evidence_by_id:
                issues.append(
                    CitationIssue(
                        claim_id=claim.claim_id,
                        citation_id=citation_id,
                        reason="citation id not found in retrieved evidence",
                        support_score=0.0,
                    )
                )
                continue
            if not claim_tokens:
                continue
            passage_tokens = normalized_passages.get(citation_id, set())
            matched_terms = sorted(claim_tokens.intersection(passage_tokens))
            support_score = len(matched_terms) / max(len(claim_tokens), 1)
            if matched_terms:
                claim_supported = True
            issues.append(
                CitationIssue(
                    claim_id=claim.claim_id,
                    citation_id=citation_id,
                    reason=(
                        "citation provides lexical support"
                        if matched_terms
                        else "citation passage does not lexically support claim"
                    ),
                    support_score=support_score,
                    matched_terms=matched_terms,
                )
            )
        if claim_tokens and not claim_supported:
            issues.append(
                CitationIssue(
                    claim_id=claim.claim_id,
                    citation_id="",
                    reason="no cited passage lexically supports claim",
                    support_score=0.0,
                )
            )
    return [issue for issue in issues if issue.reason != "citation provides lexical support"]


def detect_unsupported_claims(
    report: ResearchReport, evidence: list[EvidenceChunk]
) -> list[UnsupportedClaim]:
    evidence_lookup = {
        chunk.evidence_id: set(_normalize_text_tokens(chunk.text)) for chunk in evidence
    }
    unsupported: list[UnsupportedClaim] = []
    for claim in report.claims:
        claim_tokens = {token for token in _normalize_text_tokens(claim.text) if len(token) > 3}
        if not claim_tokens:
            continue
        supported = False
        for citation_id in claim.citation_ids:
            passage = evidence_lookup.get(citation_id, set())
            overlap = claim_tokens.intersection(passage)
            if overlap:
                supported = True
                break
        if not supported:
            unsupported.append(
                UnsupportedClaim(
                    claim_id=claim.claim_id,
                    reason="no lexical overlap between claim and cited passages",
                )
            )
    return unsupported
