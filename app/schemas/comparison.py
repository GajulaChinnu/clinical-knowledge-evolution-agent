"""Pydantic schemas for LLM-based protocol comparison results."""

from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.gaps import ComparisonResult, DifferenceType


class ComparisonResponse(BaseModel):
    """Structured clinical recommendation vs protocol section comparison output."""

    model_config = ConfigDict(validate_assignment=True)

    comparison_result: ComparisonResult = Field(
        ...,
        description="Conclusion of the comparison: gap, no_gap, ambiguous, or no_match.",
    )
    matched_protocol_section: str = Field(
        ...,
        min_length=1,
        description="Name or title of the matched candidate protocol section.",
    )
    protocol_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the compared protocol (e.g. PROT-DM-001).",
    )
    protocol_version: str = Field(
        ...,
        min_length=1,
        description="Immutable version of the compared protocol (e.g. v1.0).",
    )
    section_id: str = Field(
        ...,
        min_length=1,
        description="Section identifier within the protocol (e.g. SEC-3).",
    )
    exact_protocol_text: str = Field(
        ...,
        min_length=1,
        description="Verbatim text quotation from the candidate protocol section directly compared against.",
    )
    specific_difference: str = Field(
        ...,
        min_length=1,
        description="Precise textual summary of the clinical difference between recommendation and protocol.",
    )
    difference_type: DifferenceType = Field(
        ...,
        description="Controlled category of difference detected.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Comparison confidence score between 0.0 and 1.0.",
    )
    rationale: str = Field(
        ...,
        min_length=1,
        description="Clinical reasoning describing the textual difference and alignment.",
    )

    @field_validator(
        "matched_protocol_section",
        "protocol_id",
        "protocol_version",
        "section_id",
        "exact_protocol_text",
        "specific_difference",
        "rationale",
    )
    @classmethod
    def _validate_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty or whitespace only.")
        return v.strip()

    def is_internally_consistent(self) -> bool:
        """Check for contradictions or logical inconsistencies in model comparison output."""
        # 1. No gap claimed but a material difference category was selected
        if self.comparison_result == ComparisonResult.NO_GAP:
            if self.difference_type not in (DifferenceType.NONE, DifferenceType.NO_MATERIAL_DIFFERENCE):
                return False
        # 2. Gap claimed but difference type is none or no_material_difference
        if self.comparison_result == ComparisonResult.GAP:
            if self.difference_type in (DifferenceType.NONE, DifferenceType.NO_MATERIAL_DIFFERENCE, DifferenceType.NO_MATCH):
                return False
        # 3. No match claimed but difference type is something else
        if self.comparison_result == ComparisonResult.NO_MATCH:
            if self.difference_type != DifferenceType.NO_MATCH:
                return False
        return True
