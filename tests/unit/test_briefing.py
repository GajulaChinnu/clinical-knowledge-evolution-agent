"""Unit tests for Phase 8 Deterministic Briefing Agent and Brief Renderer.

Verifies:
1. Structured payload validation
2. Seven sections present
3. Verbatim recommendation preserved
4. Protocol text preserved
5. Explicit no-match rendering
6. Difference rendering
7. Impact rendering
8. Incomplete impact rendering
9. Workflow rendering
10. Three proposed actions
11. Source excerpt preservation
12. Source provenance
13. Protocol provenance
14. HTML rendering
15. Markdown rendering
16. JSON companion generation
17. Rendered file hashing
18. ChangeBrief persistence
19. Incomplete brief detection
20. No LLM invocation
21. Deterministic regeneration
22. Missing workflow information handled without fabrication
"""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import uuid
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agents.briefing_agent import BriefingAgent, RecordNotFoundError
from app.models.database import Base, get_session_factory
from app.models.entities import (
    ChangeBrief,
    ChangeRecord,
    GapRecord,
    ImpactRecord,
    IngestedDocument,
)
from app.schemas.briefs import (
    BriefCompletenessError,
    BriefStatus,
    ComparisonSectionPayload,
    ImpactSectionPayload,
    ProposedActionItem,
    ProtocolSectionPayload,
    RecommendationSectionPayload,
    SourceExcerptSectionPayload,
    SourceMetadataPayload,
    StructuredBriefPayload,
    WorkflowSectionPayload,
    assert_brief_completeness,
    validate_brief_completeness,
)
from app.schemas.changes import ChangeStatus
from app.schemas.gaps import DifferenceType, GapStatus
from app.schemas.impact import ImpactStatus, ImpactTier
from app.services.brief_renderer import BriefRenderer
from app.services.config_service import AppConfig
from app.services.file_hash import compute_sha256


@pytest.fixture
def briefing_env(tmp_path):
    """Set up an isolated environment with in-memory database and temporary directories."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_factory = get_session_factory(engine=engine)

    # Use project templates directory
    project_root = Path(__file__).resolve().parents[2]
    templates_dir = project_root / "templates"
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = AppConfig(
        templates_dir=templates_dir,
        output_dir=output_dir,
    )

    renderer = BriefRenderer(templates_dir=templates_dir, output_dir=output_dir, config=config)
    agent = BriefingAgent(session_factory=session_factory, renderer=renderer, config=config)

    yield {
        "engine": engine,
        "session_factory": session_factory,
        "renderer": renderer,
        "agent": agent,
        "config": config,
        "output_dir": output_dir,
        "tmp_path": tmp_path,
    }

    engine.dispose()


def _create_full_pipeline_records(
    session_factory,
    verbatim_text: str = "Initiate SGLT2 inhibitor therapy for adults with T2D and established CKD.",
    source_identifier: str = "ADA-2026-Guidelines.pdf",
    page: int = 15,
    section: str = "Section 10: Cardiovascular and Renal Risk Management",
    source_excerpt: str = "In adults with type 2 diabetes and established CKD, an SGLT2 inhibitor should be initiated to reduce CKD progression and cardiovascular events.",
    matched_protocol_id: str = "PROT-DM-001",
    matched_protocol_version: str = "1.0",
    is_match: bool = True,
    comparison_result: str = "gap",
    difference_type: str = DifferenceType.DOSAGE_CHANGE.value,
    is_impact_complete: bool = True,
    clinical_urgency: int = 4,
    evidence_strength: int = 5,
    pathway_breadth: int = 3,
    total_score: int = 12,
    tier: str = ImpactTier.CRITICAL.value,
    routing_target: str = "Rapid Governance Committee",
) -> str:
    """Helper to persist a complete chain: IngestedDocument -> ChangeRecord -> GapRecord -> ImpactRecord."""
    unique_id = uuid.uuid4().hex
    with session_factory() as session:
        doc = IngestedDocument(
            source_identifier=source_identifier,
            source_path=f"/data/sources/{source_identifier}",
            sha256_hash=(unique_id * 2)[:64],
            document_version="1.0",
            source_version="2026.1",
            status="processed",
        )
        session.add(doc)
        session.flush()

        change = ChangeRecord(
            ingested_document_id=doc.id,
            verbatim_text=verbatim_text,
            recommendation_type="pharmacotherapy",
            target_population="Adults with Type 2 Diabetes and CKD",
            intervention="SGLT2 inhibitor initiation",
            evidence_grade="Grade A",
            page=page,
            section=section,
            source_excerpt=source_excerpt,
            extraction_model_version="openai/gpt-oss-20b",
            extraction_prompt_version="1.0",
            status=ChangeStatus.GAP_CONFIRMED.value,
            confidence=0.95,
        )
        session.add(change)
        session.flush()

        gap = GapRecord(
            change_record_id=change.id,
            candidate_protocol_section_ids=[f"{matched_protocol_id}__{matched_protocol_version}__SEC-3"],
            similarity=0.89,
            comparison_result=comparison_result,
            comparison_confidence=0.94,
            difference_type=difference_type,
            matched_protocol_id=matched_protocol_id if is_match else None,
            matched_protocol_version=matched_protocol_version if is_match else None,
            is_match=is_match,
            reviewer_resolution="Validated by Comparison Agent",
            status=GapStatus.MATCHED.value if is_match else GapStatus.NO_MATCH.value,
            schema_version="1.0",
        )
        session.add(gap)
        session.flush()

        deadline = datetime.now(timezone.utc) + timedelta(hours=48)
        impact = ImpactRecord(
            gap_record_id=gap.id,
            clinical_urgency=clinical_urgency if is_impact_complete else None,
            evidence_strength=evidence_strength if is_impact_complete else None,
            pathway_breadth=pathway_breadth if is_impact_complete else None,
            rule_ids={
                "clinical_urgency": "URGENCY-04",
                "evidence_strength": "EVIDENCE-05",
                "pathway_breadth": "BREADTH-03",
            } if is_impact_complete else {},
            scoring_yaml_version="1.0",
            total_score=total_score if is_impact_complete else None,
            tier=tier if is_impact_complete else None,
            routing_target=routing_target if is_impact_complete else "Governance Committee",
            sla_deadline=deadline if is_impact_complete else None,
            urgency_basis="Time-sensitive treatment change where delay may worsen outcomes" if is_impact_complete else "Unmapped urgency",
            evidence_basis="Class I / Grade 1A systematic review or RCT" if is_impact_complete else "Unmapped evidence",
            breadth_basis="One specialty, multiple care pathways" if is_impact_complete else "Unmapped breadth",
            status=ImpactStatus.CALCULATED.value if is_impact_complete else ImpactStatus.INCOMPLETE.value,
            schema_version="1.0",
        )
        session.add(impact)
        session.commit()
        return impact.id


# ==============================================================================
# TESTS 1 & 2: STRUCTURED PAYLOAD AND SEVEN SECTIONS
# ==============================================================================

def test_structured_payload_validation():
    """Test 1: Verify structured brief payload validates cleanly with Pydantic."""
    payload = StructuredBriefPayload(
        brief_id=str(uuid.uuid4()),
        impact_record_id=str(uuid.uuid4()),
        gap_record_id=str(uuid.uuid4()),
        change_record_id=str(uuid.uuid4()),
        schema_version="1.0",
        generated_at=datetime.now(timezone.utc).isoformat(),
        source_metadata=SourceMetadataPayload(
            source_identifier="ada-2026.pdf",
            source_path="/data/sources/ada-2026.pdf",
            source_version="1.0",
        ),
        what_changed=RecommendationSectionPayload(
            recommendation_text="Start SGLT2 inhibitor in adults with T2D.",
            source_identifier="ada-2026.pdf",
            page=12,
            section="Pharmacology",
            recommendation_type="pharmacotherapy",
            target_population="Adults with T2D",
            intervention="SGLT2 inhibitor",
        ),
        current_protocol=ProtocolSectionPayload(
            is_match=True,
            protocol_id="PROT-DM-001",
            protocol_version="1.0",
            section_id="SEC-2",
            exact_protocol_text="Metformin is the initial preferred agent.",
        ),
        specific_difference=ComparisonSectionPayload(
            specific_difference="Recommendation adds SGLT2 inhibitor as first line alongside Metformin.",
            difference_type="dosage_change",
            comparison_result="gap",
        ),
        impact_assessment=ImpactSectionPayload(
            status="completed",
            is_complete=True,
            clinical_urgency=4,
            urgency_basis="Time-sensitive treatment change",
            evidence_strength=5,
            evidence_basis="Grade A RCT evidence",
            pathway_breadth=3,
            breadth_basis="One specialty, multiple pathways",
            total_score=12,
            tier="Critical",
            routing_target="Rapid Governance Committee",
        ),
        affected_workflows=WorkflowSectionPayload(
            affected_workflows=["Endocrinology Clinic Outpatient", "Primary Care Diabetes Review"],
        ),
        proposed_actions=[
            ProposedActionItem(step=1, action="Confirm applicability", description="Confirm clinical setting."),
            ProposedActionItem(step=2, action="Update protocol if adopted", description="Update EHR order set."),
            ProposedActionItem(step=3, action="Record rationale either way", description="Document in audit log."),
        ],
        source_excerpt=SourceExcerptSectionPayload(
            source_excerpt="In adults with type 2 diabetes, SGLT2 inhibitors should be considered early.",
            source_identifier="ada-2026.pdf",
            page=12,
            section="Pharmacology",
        ),
    )

    is_valid, errors = validate_brief_completeness(payload)
    assert is_valid, f"Validation failed: {errors}"
    assert payload.brief_id is not None
    assert len(payload.proposed_actions) == 3


def test_seven_sections_present(briefing_env):
    """Test 2: Verify all seven required sections are rendered in both HTML and Markdown."""
    session_factory = briefing_env["session_factory"]
    agent = briefing_env["agent"]
    output_dir = briefing_env["output_dir"]

    impact_id = _create_full_pipeline_records(session_factory)
    brief = agent.process_impact_record(
        impact_record_id=impact_id,
        exact_protocol_text="Current protocol recommends metformin monotherapy only.",
    )

    html_file = output_dir / f"brief_{brief.id}.html"
    md_file = output_dir / f"brief_{brief.id}.md"

    assert html_file.exists()
    assert md_file.exists()

    html_text = html_file.read_text(encoding="utf-8")
    md_text = md_file.read_text(encoding="utf-8")

    # Verify seven sections in HTML
    assert "What Changed" in html_text
    assert "What Our Protocol Currently Says" in html_text
    assert "Specific Difference" in html_text
    assert "Impact Assessment" in html_text
    assert "Affected Workflows" in html_text
    assert "Proposed Review Actions" in html_text
    assert "Source Excerpt" in html_text

    # Verify seven sections in Markdown
    assert "## 1. What Changed" in md_text
    assert "## 2. What Our Protocol Currently Says" in md_text
    assert "## 3. Specific Difference" in md_text
    assert "## 4. Impact Assessment" in md_text
    assert "## 5. Affected Workflows" in md_text
    assert "## 6. Proposed Review Actions" in md_text
    assert "## 7. Source Excerpt" in md_text


# ==============================================================================
# TESTS 3–7: CONTENT INTEGRITY FOR SECTIONS 1 THROUGH 4
# ==============================================================================

def test_verbatim_recommendation_preserved(briefing_env):
    """Test 3: Verify Section 1 preserves verbatim recommendation without modification."""
    session_factory = briefing_env["session_factory"]
    agent = briefing_env["agent"]

    rec_text = "Initiate SGLT2 inhibitor therapy for adults with T2D and CKD."
    impact_id = _create_full_pipeline_records(session_factory, verbatim_text=rec_text)

    brief = agent.process_impact_record(
        impact_record_id=impact_id,
        exact_protocol_text="Current protocol specifies metformin.",
    )

    assert brief.structured_payload["what_changed"]["recommendation_text"] == rec_text
    md_content = Path(brief.rendered_file_path).read_text(encoding="utf-8")
    assert rec_text in md_content


def test_protocol_text_preserved(briefing_env):
    """Test 4: Verify Section 2 preserves exact protocol quotation for matched protocols."""
    session_factory = briefing_env["session_factory"]
    agent = briefing_env["agent"]

    protocol_quote = "First-line therapy for T2D is metformin 500mg BID with lifestyle modification."
    impact_id = _create_full_pipeline_records(session_factory)

    brief = agent.process_impact_record(
        impact_record_id=impact_id,
        exact_protocol_text=protocol_quote,
    )

    assert brief.structured_payload["current_protocol"]["exact_protocol_text"] == protocol_quote
    md_content = Path(brief.rendered_file_path).read_text(encoding="utf-8")
    assert protocol_quote in md_content


def test_explicit_no_match_rendering(briefing_env):
    """Test 5: Verify Section 2 explicitly states no-match without inventing protocol text."""
    session_factory = briefing_env["session_factory"]
    agent = briefing_env["agent"]

    impact_id = _create_full_pipeline_records(
        session_factory,
        is_match=False,
        comparison_result="no_match",
        difference_type=DifferenceType.NO_MATCH.value,
        matched_protocol_id=None,
    )

    brief = agent.process_impact_record(impact_record_id=impact_id)

    curr_prot = brief.structured_payload["current_protocol"]
    assert curr_prot["is_match"] is False
    assert curr_prot["exact_protocol_text"] is None
    assert "No matching protocol section was identified" in curr_prot["no_match_statement"]

    md_content = Path(brief.rendered_file_path).read_text(encoding="utf-8")
    assert "No Matching Protocol Section Identified" in md_content


def test_difference_rendering(briefing_env):
    """Test 6: Verify Section 3 renders specific difference and difference type."""
    session_factory = briefing_env["session_factory"]
    agent = briefing_env["agent"]

    specific_diff = "Adds SGLT2 inhibitor for dual therapy in patients with CKD stage 3."
    impact_id = _create_full_pipeline_records(session_factory, difference_type="dosage_change")

    brief = agent.process_impact_record(
        impact_record_id=impact_id,
        exact_protocol_text="Metformin only.",
        specific_difference=specific_diff,
    )

    diff_data = brief.structured_payload["specific_difference"]
    assert diff_data["specific_difference"] == specific_diff
    assert diff_data["difference_type"] == "dosage_change"


def test_impact_rendering(briefing_env):
    """Test 7: Verify Section 4 renders all dimensions, scores, bases, tier, and SLA."""
    session_factory = briefing_env["session_factory"]
    agent = briefing_env["agent"]

    impact_id = _create_full_pipeline_records(
        session_factory,
        clinical_urgency=4,
        evidence_strength=5,
        pathway_breadth=3,
        total_score=12,
        tier="Critical",
        routing_target="Rapid Governance Committee",
    )

    brief = agent.process_impact_record(
        impact_record_id=impact_id,
        exact_protocol_text="Metformin only.",
    )

    impact_data = brief.structured_payload["impact_assessment"]
    assert impact_data["is_complete"] is True
    assert impact_data["clinical_urgency"] == 4
    assert impact_data["evidence_strength"] == 5
    assert impact_data["pathway_breadth"] == 3
    assert impact_data["total_score"] == 12
    assert impact_data["tier"] == "Critical"
    assert impact_data["routing_target"] == "Rapid Governance Committee"


def test_incomplete_impact_rendering(briefing_env):
    """Test 8: Verify Section 4 renders incomplete status without inventing tier or SLA."""
    session_factory = briefing_env["session_factory"]
    agent = briefing_env["agent"]

    impact_id = _create_full_pipeline_records(
        session_factory,
        is_impact_complete=False,
    )

    brief = agent.process_impact_record(
        impact_record_id=impact_id,
        exact_protocol_text="Metformin only.",
    )

    impact_data = brief.structured_payload["impact_assessment"]
    assert impact_data["is_complete"] is False
    assert impact_data["tier"] is None
    assert impact_data["total_score"] is None
    assert impact_data["sla_deadline"] is None
    assert impact_data["incomplete_reason"] is not None

    md_content = Path(brief.rendered_file_path).read_text(encoding="utf-8")
    assert "Impact Assessment Incomplete" in md_content


# ==============================================================================
# TESTS 9–13: WORKFLOWS, ACTIONS, AND PROVENANCE
# ==============================================================================

def test_workflow_rendering(briefing_env):
    """Test 9: Verify Section 5 renders structured workflows cleanly."""
    session_factory = briefing_env["session_factory"]
    agent = briefing_env["agent"]

    custom_workflows = ["Nephrology Inpatient Consult", "Diabetes Self-Management Clinic"]
    impact_id = _create_full_pipeline_records(session_factory)

    brief = agent.process_impact_record(
        impact_record_id=impact_id,
        exact_protocol_text="Metformin only.",
        affected_workflows=custom_workflows,
    )

    wf_data = brief.structured_payload["affected_workflows"]
    assert wf_data["is_available"] is True
    assert wf_data["affected_workflows"] == custom_workflows


def test_three_proposed_actions(briefing_env):
    """Test 10: Verify Section 6 contains the approved 3 proposed actions."""
    session_factory = briefing_env["session_factory"]
    agent = briefing_env["agent"]

    impact_id = _create_full_pipeline_records(session_factory)
    brief = agent.process_impact_record(
        impact_record_id=impact_id,
        exact_protocol_text="Metformin only.",
    )

    actions = brief.structured_payload["proposed_actions"]
    assert len(actions) == 3
    action_names = [a["action"] for a in actions]
    assert "Confirm applicability" in action_names
    assert "Update protocol if adopted" in action_names
    assert "Record rationale either way" in action_names


def test_source_excerpt_preservation(briefing_env):
    """Test 11: Verify Section 7 preserves full source paragraph verbatim."""
    session_factory = briefing_env["session_factory"]
    agent = briefing_env["agent"]

    excerpt = "Full paragraph with detailed clinical context from published ADA 2026 guidelines."
    impact_id = _create_full_pipeline_records(session_factory, source_excerpt=excerpt)

    brief = agent.process_impact_record(
        impact_record_id=impact_id,
        exact_protocol_text="Metformin only.",
    )

    assert brief.structured_payload["source_excerpt"]["source_excerpt"] == excerpt
    md_content = Path(brief.rendered_file_path).read_text(encoding="utf-8")
    assert excerpt in md_content


def test_source_provenance(briefing_env):
    """Test 12: Verify source identifier, page, section, and version are preserved."""
    session_factory = briefing_env["session_factory"]
    agent = briefing_env["agent"]

    impact_id = _create_full_pipeline_records(
        session_factory,
        source_identifier="KDIGO-2026-CKD.pdf",
        page=42,
        section="Chapter 3: Glycemic Management",
    )

    brief = agent.process_impact_record(
        impact_record_id=impact_id,
        exact_protocol_text="Metformin only.",
    )

    meta = brief.structured_payload["source_metadata"]
    assert meta["source_identifier"] == "KDIGO-2026-CKD.pdf"
    assert brief.structured_payload["what_changed"]["page"] == 42
    assert brief.structured_payload["what_changed"]["section"] == "Chapter 3: Glycemic Management"


def test_protocol_provenance(briefing_env):
    """Test 13: Verify protocol ID, version, and section ID are preserved for matched protocols."""
    session_factory = briefing_env["session_factory"]
    agent = briefing_env["agent"]

    impact_id = _create_full_pipeline_records(
        session_factory,
        matched_protocol_id="PROT-CKD-004",
        matched_protocol_version="2.1",
    )

    brief = agent.process_impact_record(
        impact_record_id=impact_id,
        exact_protocol_text="Metformin only.",
    )

    prot = brief.structured_payload["current_protocol"]
    assert prot["protocol_id"] == "PROT-CKD-004"
    assert prot["protocol_version"] == "2.1"


# ==============================================================================
# TESTS 14–18: RENDERING, JSON, HASHING, PERSISTENCE
# ==============================================================================

def test_html_rendering(briefing_env):
    """Test 14: Verify valid HTML file is written to output_dir."""
    session_factory = briefing_env["session_factory"]
    agent = briefing_env["agent"]
    output_dir = briefing_env["output_dir"]

    impact_id = _create_full_pipeline_records(session_factory)
    brief = agent.process_impact_record(
        impact_record_id=impact_id,
        exact_protocol_text="Metformin only.",
    )

    html_path = output_dir / f"brief_{brief.id}.html"
    assert html_path.is_file()
    html_content = html_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html_content
    assert "Clinical Change Brief" in html_content


def test_markdown_rendering(briefing_env):
    """Test 15: Verify valid Markdown file is written to output_dir."""
    session_factory = briefing_env["session_factory"]
    agent = briefing_env["agent"]
    output_dir = briefing_env["output_dir"]

    impact_id = _create_full_pipeline_records(session_factory)
    brief = agent.process_impact_record(
        impact_record_id=impact_id,
        exact_protocol_text="Metformin only.",
    )

    md_path = output_dir / f"brief_{brief.id}.md"
    assert md_path.is_file()
    md_content = md_path.read_text(encoding="utf-8")
    assert f"# Clinical Change Brief: {brief.id}" in md_content


def test_json_companion_generation(briefing_env):
    """Test 16: Verify JSON companion file matches structured payload."""
    session_factory = briefing_env["session_factory"]
    agent = briefing_env["agent"]
    output_dir = briefing_env["output_dir"]

    impact_id = _create_full_pipeline_records(session_factory)
    brief = agent.process_impact_record(
        impact_record_id=impact_id,
        exact_protocol_text="Metformin only.",
    )

    json_path = output_dir / f"brief_{brief.id}.json"
    assert json_path.is_file()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["brief_id"] == brief.id
    assert data["impact_record_id"] == impact_id


def test_rendered_file_hashing(briefing_env):
    """Test 17: Verify rendered file SHA-256 hash is computed accurately."""
    session_factory = briefing_env["session_factory"]
    agent = briefing_env["agent"]

    impact_id = _create_full_pipeline_records(session_factory)
    brief = agent.process_impact_record(
        impact_record_id=impact_id,
        exact_protocol_text="Metformin only.",
    )

    actual_hash = compute_sha256(brief.rendered_file_path)
    assert brief.rendered_file_hash == actual_hash
    assert len(brief.rendered_file_hash) == 64


def test_change_brief_persistence(briefing_env):
    """Test 18: Verify ChangeBrief entity is persisted in database with status draft."""
    session_factory = briefing_env["session_factory"]
    agent = briefing_env["agent"]

    impact_id = _create_full_pipeline_records(session_factory)
    brief = agent.process_impact_record(
        impact_record_id=impact_id,
        exact_protocol_text="Metformin only.",
    )

    with session_factory() as session:
        persisted = session.query(ChangeBrief).filter_by(id=brief.id).first()
        assert persisted is not None
        assert persisted.status == BriefStatus.DRAFT.value
        assert persisted.impact_record_id == impact_id
        assert persisted.rendered_file_path == brief.rendered_file_path
        assert persisted.rendered_file_hash == brief.rendered_file_hash


# ==============================================================================
# TESTS 19–22: COMPLETENESS, NO LLM, REPRODUCIBILITY, WORKFLOW FALLBACK
# ==============================================================================

def test_incomplete_brief_detection():
    """Test 19: Verify automated completeness validator rejects incomplete payloads."""
    # Create payload missing recommendation text quote
    payload = StructuredBriefPayload(
        brief_id=str(uuid.uuid4()),
        impact_record_id=str(uuid.uuid4()),
        gap_record_id=str(uuid.uuid4()),
        change_record_id=str(uuid.uuid4()),
        schema_version="1.0",
        generated_at=datetime.now(timezone.utc).isoformat(),
        source_metadata=SourceMetadataPayload(
            source_identifier="ada-2026.pdf",
            source_path="/data/sources/ada-2026.pdf",
            source_version="1.0",
        ),
        what_changed=RecommendationSectionPayload(
            recommendation_text="   ",  # whitespace only
            source_identifier="ada-2026.pdf",
            page=12,
            section="Section 1",
            recommendation_type="pharmacotherapy",
            target_population="Adults with T2D",
            intervention="Metformin",
        ),
        current_protocol=ProtocolSectionPayload(
            is_match=True,
            protocol_id="PROT-01",
            protocol_version="1.0",
            exact_protocol_text="Current protocol text",
        ),
        specific_difference=ComparisonSectionPayload(
            specific_difference="Specific difference text",
            difference_type="dosage_change",
            comparison_result="gap",
        ),
        impact_assessment=ImpactSectionPayload(
            status="completed",
            is_complete=True,
            clinical_urgency=4,
            urgency_basis="Urgency basis",
            evidence_strength=5,
            evidence_basis="Evidence basis",
            pathway_breadth=3,
            breadth_basis="Breadth basis",
            total_score=12,
            tier="Critical",
        ),
        affected_workflows=WorkflowSectionPayload(affected_workflows=["Workflow 1"]),
        proposed_actions=[
            ProposedActionItem(step=1, action="Confirm applicability", description="Action 1"),
            ProposedActionItem(step=2, action="Update protocol if adopted", description="Action 2"),
            ProposedActionItem(step=3, action="Record rationale either way", description="Action 3"),
        ],
        source_excerpt=SourceExcerptSectionPayload(
            source_excerpt="Source excerpt text",
            source_identifier="ada-2026.pdf",
        ),
    )

    with pytest.raises(BriefCompletenessError) as exc_info:
        assert_brief_completeness(payload)
    assert "Section 1: Missing recommendation text quote" in str(exc_info.value)


def test_no_llm_invocation(briefing_env, monkeypatch):
    """Test 20: Verify zero LLM calls occur during briefing generation."""
    session_factory = briefing_env["session_factory"]
    agent = briefing_env["agent"]

    import openai
    def fail_if_llm_called(*args, **kwargs):
        raise AssertionError("LLM call attempted inside Briefing Agent!")

    monkeypatch.setattr(openai.resources.chat.completions.Completions, "create", fail_if_llm_called)

    impact_id = _create_full_pipeline_records(session_factory)
    brief = agent.process_impact_record(
        impact_record_id=impact_id,
        exact_protocol_text="Metformin only.",
    )
    assert brief is not None


def test_deterministic_regeneration(briefing_env):
    """Test 21: Verify identical inputs produce identical rendered file hashes."""
    session_factory = briefing_env["session_factory"]
    agent = briefing_env["agent"]

    impact_id = _create_full_pipeline_records(session_factory)

    # First generation
    brief1 = agent.process_impact_record(
        impact_record_id=impact_id,
        exact_protocol_text="Metformin 500mg BID only.",
        specific_difference="Adds SGLT2 inhibitor as initial therapy alongside Metformin.",
    )

    # Second generation (same inputs)
    brief2 = agent.process_impact_record(
        impact_record_id=impact_id,
        exact_protocol_text="Metformin 500mg BID only.",
        specific_difference="Adds SGLT2 inhibitor as initial therapy alongside Metformin.",
    )

    assert brief1.id == brief2.id
    # Compare rendered payloads excluding timestamps
    p1 = {k: v for k, v in brief1.structured_payload.items() if k != "generated_at"}
    p2 = {k: v for k, v in brief2.structured_payload.items() if k != "generated_at"}
    assert p1 == p2


def test_missing_workflow_information_handled_without_fabrication(briefing_env):
    """Test 22: Verify unavailable workflow information is explicitly displayed without fabrication."""
    session_factory = briefing_env["session_factory"]
    agent = briefing_env["agent"]

    impact_id = _create_full_pipeline_records(session_factory)
    brief = agent.process_impact_record(
        impact_record_id=impact_id,
        exact_protocol_text="Metformin only.",
        is_workflow_available=False,
        unavailability_reason="Clinical workflow mapping not available in source guideline metadata.",
    )

    wf_payload = brief.structured_payload["affected_workflows"]
    assert wf_payload["is_available"] is False
    assert len(wf_payload["affected_workflows"]) == 0
    assert "Clinical workflow mapping not available" in wf_payload["unavailability_reason"]

    md_content = Path(brief.rendered_file_path).read_text(encoding="utf-8")
    assert "Clinical workflow mapping not available in source guideline metadata" in md_content


# ==============================================================================
# NO INVENTED OR SUBSTITUTED PROTOCOL CONTENT
# ==============================================================================

def _set_gap_evidence(session_factory, impact_id, **fields):
    with session_factory() as session:
        impact = session.get(ImpactRecord, impact_id)
        gap = session.get(GapRecord, impact.gap_record_id)
        for key, value in fields.items():
            setattr(gap, key, value)
        session.commit()


def test_brief_uses_verified_comparison_evidence_from_gap(briefing_env):
    sf = briefing_env["session_factory"]
    impact_id = _create_full_pipeline_records(sf)
    _set_gap_evidence(
        sf, impact_id,
        matched_section_id="SEC-3",
        matched_section_heading="Cardiorenal Therapy",
        exact_protocol_text="SGLT2 inhibitors are reserved for eGFR ≥ 45 mL/min.",
        specific_difference="Guideline extends SGLT2 initiation to eGFR ≥ 20.",
    )
    brief = briefing_env["agent"].process_impact_record(impact_id)
    current = brief.structured_payload["current_protocol"]
    assert current["exact_protocol_text"] == "SGLT2 inhibitors are reserved for eGFR ≥ 45 mL/min."
    assert current["section_id"] == "SEC-3"
    assert current["section_heading"] == "Cardiorenal Therapy"
    assert brief.structured_payload["specific_difference"]["specific_difference"] == "Guideline extends SGLT2 initiation to eGFR ≥ 20."


def test_unresolvable_protocol_text_is_refused_not_invented(briefing_env):
    """No stored quote and no exact protocol/version/section on file -> no brief, no placeholder."""
    from app.agents.briefing_agent import BriefingError

    sf = briefing_env["session_factory"]
    # PROT-DM-001 exists on disk only as version "v1.0"; "9.9" must not fall back to it.
    impact_id = _create_full_pipeline_records(sf, matched_protocol_version="9.9")
    with pytest.raises(BriefingError, match="could not be resolved"):
        briefing_env["agent"].process_impact_record(impact_id)
    with sf() as session:
        assert session.query(ChangeBrief).count() == 0


def test_protocol_text_resolved_only_from_exact_version_and_section(briefing_env, tmp_path):
    import json
    from dataclasses import replace

    proto_dir = tmp_path / "protocols"
    proto_dir.mkdir()
    for version, text in (("v1.0", "OLD VERSION TEXT for SEC-3."), ("v2.0", "CURRENT v2 TEXT for SEC-3.")):
        (proto_dir / f"PROT-X_{version}.json").write_text(json.dumps({
            "protocol_id": "PROT-X", "protocol_version": version, "title": "X",
            "sections": [
                {"section_id": "SEC-1", "section_heading": "Other", "section_text": f"Unrelated {version} text."},
                {"section_id": "SEC-3", "section_heading": "Target", "section_text": text},
            ],
        }), encoding="utf-8")
    agent = briefing_env["agent"]
    agent.config = agent.config.model_copy(update={"protocol_dir": proto_dir})

    sf = briefing_env["session_factory"]
    impact_id = _create_full_pipeline_records(sf, matched_protocol_id="PROT-X", matched_protocol_version="v2.0")
    current = agent.process_impact_record(impact_id).structured_payload["current_protocol"]
    assert current["exact_protocol_text"] == "CURRENT v2 TEXT for SEC-3."
    assert current["section_heading"] == "Target"
    assert current["protocol_version"] == "v2.0"


def test_ui_renders_real_brief_payload_evidence(briefing_env):
    """The reviewer-facing view must read the real StructuredBriefPayload keys (previously it showed None)."""
    from unittest.mock import MagicMock, patch
    from app.ui.streamlit_app import render_brief_sections

    sf = briefing_env["session_factory"]
    impact_id = _create_full_pipeline_records(sf)
    _set_gap_evidence(
        sf, impact_id,
        matched_section_id="SEC-3",
        matched_section_heading="Cardiorenal Therapy",
        exact_protocol_text="SGLT2 inhibitors are reserved for eGFR ≥ 45 mL/min.",
        specific_difference="Guideline extends SGLT2 initiation to eGFR ≥ 20.",
    )
    payload = briefing_env["agent"].process_impact_record(impact_id).structured_payload

    with patch("app.ui.components.st") as mock_st:
        mock_st.expander.return_value = MagicMock()
        render_brief_sections(payload)
        shown = " ".join(
            str(arg)
            for method in ("write", "info", "warning", "markdown", "caption")
            for call in getattr(mock_st, method).call_args_list
            for arg in call.args
        )

    assert "Initiate SGLT2 inhibitor therapy for adults with T2D and established CKD." in shown
    assert "SGLT2 inhibitors are reserved for eGFR ≥ 45 mL/min." in shown
    assert "Guideline extends SGLT2 initiation to eGFR ≥ 20." in shown
    assert "PROT-DM-001" in shown
    assert "In adults with type 2 diabetes and established CKD" in shown
    assert "**Verbatim Recommendation:** None" not in shown
