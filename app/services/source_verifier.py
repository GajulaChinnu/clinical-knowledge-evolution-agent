"""Deterministic source quotation and excerpt verification service."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class VerificationResult:
    """Outcome of recommendation source verification."""
    is_verified: bool
    reason: Optional[str] = None


def normalize_whitespace(text: str) -> str:
    """Collapse arbitrary whitespace into single spaces for robust comparison."""
    return " ".join(text.split())


def verify_recommendation_provenance(
    verbatim_text: str,
    source_excerpt: str,
    source_section_text: str,
    page: int,
    expected_page: int,
) -> VerificationResult:
    """Verify that an extracted recommendation and its excerpt strictly exist in the source section.

    Args:
        verbatim_text: Exact quotation reported by the extraction model.
        source_excerpt: Surrounding context reported by the extraction model.
        source_section_text: Raw ground-truth text of the source section.
        page: Page number reported by the extraction model.
        expected_page: Ground truth page number of the candidate section.

    Returns:
        VerificationResult indicating whether the extraction matches source provenance.
    """
    # 1. Page provenance verification
    if page != expected_page:
        return VerificationResult(
            is_verified=False,
            reason=f"Page mismatch: reported page {page} does not match section page {expected_page}.",
        )

    norm_verbatim = normalize_whitespace(verbatim_text)
    norm_excerpt = normalize_whitespace(source_excerpt)
    norm_source = normalize_whitespace(source_section_text)

    # 2. Check if verbatim text is empty
    if not norm_verbatim:
        return VerificationResult(
            is_verified=False,
            reason="Verbatim recommendation text is empty.",
        )

    # 3. Check if verbatim text exists in ground truth section text
    if norm_verbatim not in norm_source:
        return VerificationResult(
            is_verified=False,
            reason="Verbatim recommendation text not found in source section content.",
        )

    # 4. Check if source excerpt exists in ground truth section text
    if norm_excerpt and norm_excerpt not in norm_source:
        return VerificationResult(
            is_verified=False,
            reason="Source excerpt not found in source section content.",
        )

    # 5. Check if verbatim text exists within the excerpt
    if norm_excerpt and norm_verbatim not in norm_excerpt:
        return VerificationResult(
            is_verified=False,
            reason="Verbatim recommendation text not found within the reported source excerpt.",
        )

    return VerificationResult(is_verified=True)
