"""Pydantic schemas for LLM-based clinical recommendation extraction."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


def _enforce_strict_json_schema(schema: Dict[str, Any]) -> None:
    """Enforce Groq strict structured outputs JSON schema requirements.

    Groq requires:
    1. 'additionalProperties: false' on every object (root and all $defs).
    2. All properties listed in the 'required' array.
    """
    schema["additionalProperties"] = False
    if "properties" in schema:
        schema["required"] = list(schema["properties"].keys())
        for prop_schema in schema["properties"].values():
            if isinstance(prop_schema, dict) and "default" in prop_schema:
                del prop_schema["default"]


class ExtractedRecommendation(BaseModel):
    """Structured clinical recommendation extracted by Groq from candidate text."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        json_schema_extra=_enforce_strict_json_schema,
    )

    verbatim_text: str = Field(
        ...,
        min_length=1,
        description="Exact verbatim recommendation sentence(s) from the source section.",
    )
    recommendation_type: str = Field(
        ...,
        min_length=1,
        description="Type of recommendation (e.g. treatment, diagnostic, monitoring, dosing).",
    )
    target_population: str = Field(
        ...,
        min_length=1,
        description="Specific patient demographic or clinical population targeted.",
    )
    intervention: str = Field(
        ...,
        min_length=1,
        description="Specific clinical action, therapy, drug, or test recommended.",
    )
    evidence_grade: Optional[str] = Field(
        default=None,
        description="Formal strength/evidence grade if explicitly stated (e.g. Grade A, Level 1a).",
    )
    page: int = Field(
        ...,
        ge=1,
        description="Source document page number where the recommendation appears.",
    )
    section: str = Field(
        ...,
        min_length=1,
        description="Section heading or title under which recommendation is located.",
    )
    source_excerpt: str = Field(
        ...,
        min_length=1,
        description="Verbatim surrounding excerpt or paragraph containing the recommendation.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Provisional routing confidence score between 0.0 and 1.0.",
    )

    @field_validator("verbatim_text", "target_population", "intervention", "section", "source_excerpt")
    @classmethod
    def _validate_non_empty(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Field cannot be empty or whitespace only")
        return clean


class ExtractionResponse(BaseModel):
    """Top-level structured container for recommendations extracted from a section."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        json_schema_extra=_enforce_strict_json_schema,
    )

    recommendations: List[ExtractedRecommendation] = Field(
        default_factory=list,
        description="List of discrete clinical recommendations identified in the candidate section.",
    )
