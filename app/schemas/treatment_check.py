"""Structured artifacts passed between the six agents for a clinician Treatment Check."""

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Verdict = Literal[
    "consistent_with_latest_guidance",
    "guidance_updated_follow_new_version",
    "conflicts_with_latest_guidance",
    "insufficient_grounded_evidence",
]

VERDICT_LABELS = {
    "consistent_with_latest_guidance": "Matches the latest guidance",
    "guidance_updated_follow_new_version": "Guidance updated: follow the new version",
    "conflicts_with_latest_guidance": "Conflicts with current guidance",
    "insufficient_grounded_evidence": "No grounded guidance covers this",
}

DISCLAIMER = "Decision support only. The treating clinician decides."

FindingKind = Literal[
    "governing_recommendation", "supporting_recommendation", "contraindication",
    "safety_warning", "withdrawal", "evidence",
]
FindingRelation = Literal[
    "matches_plan", "differs_from_plan", "applies", "not_applicable", "check_applicability",
    "withdrawn_matches_plan", "informational",
]


class Citation(BaseModel):
    """A verbatim excerpt from a stored source version, located by offsets."""

    model_config = ConfigDict(extra="forbid")

    statement_id: str
    document_id: str
    source_identity: str
    watchlist_id: Optional[str] = None
    source_title: Optional[str] = None
    source_type: Optional[str] = None
    publisher: Optional[str] = None
    version: Optional[str] = None
    published_date: Optional[str] = None
    section_heading: Optional[str] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    excerpt: str
    verified: bool = False


class AttributeDiff(BaseModel):
    kind: str
    before: str
    after: str


class Finding(BaseModel):
    """One grounded statement and how it relates to the clinician's plan."""

    model_config = ConfigDict(extra="forbid")

    kind: FindingKind
    relation: FindingRelation
    citation: Citation
    explanation: str
    differences: List[AttributeDiff] = Field(default_factory=list)
    previous_citation: Optional[Citation] = None
    change_category: Optional[str] = None
    treatments: List[str] = Field(default_factory=list)
    departments: List[str] = Field(default_factory=list)
    pathways: List[str] = Field(default_factory=list)
    applicability_basis: Optional[str] = None
    # Filled by the Impact Agent (deterministic, rule ids recorded)
    relevance: Optional[int] = None
    urgency: Optional[int] = None
    source_quality: Optional[int] = None
    novelty: Optional[str] = None
    duplicate_of: Optional[str] = None
    priority_score: Optional[float] = None
    ranking_basis: dict = Field(default_factory=dict)


class VersionComparison(BaseModel):
    """What changed for this treatment between the previous and latest version of a source."""

    model_config = ConfigDict(extra="forbid")

    source_identity: str
    source_title: Optional[str] = None
    change_category: str
    previous: Optional[Citation] = None
    latest: Optional[Citation] = None
    differences: List[AttributeDiff] = Field(default_factory=list)
    plan_matches_previous: bool = False


class ProtocolItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: str
    section_heading: str
    section_text: str
    status: Literal["aligned", "out_of_date", "missing_from_protocol"]
    reason: str
    latest_citation: Optional[Citation] = None
    previous_citation: Optional[Citation] = None
    differences: List[AttributeDiff] = Field(default_factory=list)


class ProtocolPosition(BaseModel):
    """How the institutional protocol compares with the latest guidance for this treatment."""

    model_config = ConfigDict(extra="forbid")

    protocol_id: str
    protocol_version: str
    title: Optional[str] = None
    effective_date: Optional[str] = None
    status: Literal["aligned", "out_of_date", "not_covered"]
    items: List[ProtocolItem] = Field(default_factory=list)


class SourceChecked(BaseModel):
    model_config = ConfigDict(extra="forbid")

    watchlist_id: str
    title: str
    source_type: str
    checked_at: str
    latest_version: Optional[str] = None
    latest_published: Optional[str] = None
    previous_version: Optional[str] = None
    previous_published: Optional[str] = None
    new_versions_ingested: int = 0
    error: Optional[str] = None


class ComparisonOutcome(BaseModel):
    """Comparison Agent output for a clinician query."""

    model_config = ConfigDict(extra="forbid")

    findings: List[Finding] = Field(default_factory=list)
    version_changes: List[VersionComparison] = Field(default_factory=list)
    protocol_positions: List[ProtocolPosition] = Field(default_factory=list)
    retrieval: str = "taxonomy"
    llm_used: bool = False
    notes: List[str] = Field(default_factory=list)


class StepRecord(BaseModel):
    """Progress of one agent step, shown live in the UI."""

    model_config = ConfigDict(extra="forbid")

    agent: Literal["Monitoring", "Extraction", "Comparison", "Impact", "Briefing", "Governance"]
    status: Literal["completed", "failed", "skipped"]
    summary: str
    details: List[str] = Field(default_factory=list)


class GovernanceStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_id: Optional[str] = None
    brief_id: Optional[str] = None
    status: str
    message: str
    reviewers: List[str] = Field(default_factory=list)


class ClinicianAnswer(BaseModel):
    """The answer brief produced for a clinician Treatment Check."""

    model_config = ConfigDict(extra="forbid")

    query_id: Optional[str] = None
    department: str
    department_name: str
    treatment: str
    condition: Optional[str] = None
    verdict: Optional[Verdict] = None
    verdict_label: str
    verdict_basis: str
    summary: List[str] = Field(default_factory=list, max_length=5)
    latest_guidance: List[Citation] = Field(default_factory=list)
    what_changed: List[VersionComparison] = Field(default_factory=list)
    safety_notices: List[Finding] = Field(default_factory=list)
    check_before_prescribing: List[Finding] = Field(default_factory=list)
    protocol_positions: List[ProtocolPosition] = Field(default_factory=list)
    findings: List[Finding] = Field(default_factory=list)
    affected_departments: List[str] = Field(default_factory=list)
    affected_pathways: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    sources_checked: List[SourceChecked] = Field(default_factory=list)
    governance: List[GovernanceStatus] = Field(default_factory=list)
    steps: List[StepRecord] = Field(default_factory=list)
    degraded_mode: bool = False
    disclaimer: str = DISCLAIMER
    generated_at: str
