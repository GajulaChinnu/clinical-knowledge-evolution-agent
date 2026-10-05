"""Unit tests for Phase 7 Deterministic Impact Assessment Agent and Scoring Engine.

Verifies:
1. YAML rules load correctly
2. Scoring version loads
3. Urgency scores 1 through 5
4. Evidence scores 1 through 5
5. Breadth scores 1 through 5
6. Total calculation
7. Tier boundaries (Critical=12, High=8, Standard=4, Low=3)
8. Incomplete status on missing or unmapped dimensions
9. No guessing or defaulting of scores
10. Rule IDs, YAML version, and written basis persistence
11. ImpactRecord database persistence and parent GapRecord relationship
12. Historical score version preservation
13. Automatic tier downgrade blocked
14. Manual tier downgrade requires reviewer and clinical rationale (logged to AuditLog)
15. Unresolved comparison findings cannot be scored
16. G3 confirmed no-match findings are scoreable
17. Idempotent re-scoring without duplicates
18. Zero LLM calls
"""

import uuid
from datetime import datetime, timezone
from pathlib import Path
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agents.impact_agent import ImpactAgent
from app.models.database import Base, get_session_factory
from app.models.entities import (
    AuditLog,
    ChangeRecord,
    GapRecord,
    ImpactRecord,
    IngestedDocument,
)
from app.schemas.changes import ChangeStatus
from app.schemas.gaps import DifferenceType, GapStatus
from app.schemas.impact import ImpactStatus, ImpactTier
from app.services.config_service import AppConfig
from app.services.scoring_engine import (
    InvalidRuleConfigError,
    ScoringEngine,
    TierDowngradeBlockedError,
    UnresolvedComparisonError,
)


@pytest.fixture
def scoring_engine():
    """Default ScoringEngine loading config/scoring.yaml."""
    return ScoringEngine()


@pytest.fixture
def impact_env(tmp_path):
    """Set up an isolated SQLite in-memory database and environment for impact tests."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_factory = get_session_factory(engine=engine)

    config = AppConfig(
        scoring_rules_path=Path("./config/scoring.yaml"),
    )
    engine_service = ScoringEngine(config=config)
    agent = ImpactAgent(session_factory=session_factory, scoring_engine=engine_service, config=config)

    yield {
        "engine": engine,
        "session_factory": session_factory,
        "scoring_engine": engine_service,
        "agent": agent,
        "config": config,
        "tmp_path": tmp_path,
    }

    engine.dispose()


def _create_sample_gap_record(
    session_factory,
    difference_type: str = DifferenceType.DOSAGE_CHANGE.value,
    evidence_grade: str = "Grade A",
    change_status: str = ChangeStatus.GAP_CONFIRMED.value,
    gap_status: str = GapStatus.MATCHED.value,
    is_match: bool = True,
    comparison_result: str = "gap",
) -> str:
    """Helper to persist a test ChangeRecord and GapRecord."""
    unique_id = uuid.uuid4().hex
    with session_factory() as session:
        doc = IngestedDocument(
            source_identifier=f"ada-{unique_id[:8]}.pdf",
            source_path=f"/data/sources/ada-{unique_id[:8]}.pdf",
            sha256_hash=(unique_id * 2)[:64],
            status="processed",
        )
        session.add(doc)
        session.flush()

        change = ChangeRecord(
            ingested_document_id=doc.id,
            verbatim_text="Start metformin 1000mg twice daily.",
            recommendation_type="pharmacotherapy",
            target_population="Adults with Type 2 Diabetes",
            intervention="Metformin 1000mg BID",
            evidence_grade=evidence_grade,
            page=12,
            section="Pharmacologic Therapy",
            extraction_model_version="openai/gpt-oss-20b",
            extraction_prompt_version="1.0",
            status=change_status,
            confidence=0.95,
        )
        session.add(change)
        session.flush()

        gap = GapRecord(
            change_record_id=change.id,
            candidate_protocol_section_ids=["PROT-DM-001__1.0__SEC-2"],
            similarity=0.88,
            comparison_result=comparison_result,
            comparison_confidence=0.92,
            difference_type=difference_type,
            matched_protocol_id="PROT-DM-001" if is_match else None,
            matched_protocol_version="1.0" if is_match else None,
            is_match=is_match,
            status=gap_status,
            schema_version="1.0",
        )
        session.add(gap)
        session.commit()
        return gap.id


# ==============================================================================
# TESTS 1 & 2: YAML AND VERSION LOADING
# ==============================================================================

def test_yaml_loads(scoring_engine):
    """Test 1: Verify YAML rules file loads cleanly into memory."""
    assert scoring_engine.dimensions is not None
    assert "clinical_urgency" in scoring_engine.dimensions
    assert "evidence_strength" in scoring_engine.dimensions
    assert "pathway_breadth" in scoring_engine.dimensions


def test_safe_yaml_loading_blocks_unsafe_code_execution(tmp_path, monkeypatch):
    """Verify YAML loading strictly uses safe_load and rejects unsafe arbitrary Python objects."""
    import yaml

    # 1. Verify yaml.safe_load was called
    called_safe_load = []
    original_safe_load = yaml.safe_load

    def spy_safe_load(stream):
        called_safe_load.append(True)
        return original_safe_load(stream)

    monkeypatch.setattr(yaml, "safe_load", spy_safe_load)
    ScoringEngine()
    assert len(called_safe_load) >= 1

    # 2. Verify that unsafe arbitrary Python object tags are safely rejected by yaml.safe_load
    unsafe_yaml = tmp_path / "unsafe_scoring.yaml"
    unsafe_yaml.write_text(
        'version: "1.0"\n'
        'exploit: !!python/object/apply:os.system ["echo unsafe"]\n'
        'dimensions: {}\n',
        encoding="utf-8",
    )
    with pytest.raises(InvalidRuleConfigError):
        ScoringEngine(yaml_path=unsafe_yaml)


def test_scoring_version_loads(scoring_engine):
    """Test 2: Verify scoring rules version is loaded."""
    assert scoring_engine.version == "1.0"


# ==============================================================================
# TESTS 3–7: CLINICAL URGENCY SCORES 1 THROUGH 5
# ==============================================================================

def test_urgency_score_1(scoring_engine):
    """Test 3: Clinical Urgency Score 1 (Safety notice without outcome evidence)."""
    score = scoring_engine.score_urgency("safety_notice")
    assert score is not None
    assert score.score == 1
    assert score.rule_id == "URGENCY-01"
    assert "Safety notice" in score.basis


def test_urgency_score_2(scoring_engine):
    """Test 4: Clinical Urgency Score 2 (Non-safety workflow or documentation change)."""
    score = scoring_engine.score_urgency("workflow_change")
    assert score is not None
    assert score.score == 2
    assert score.rule_id == "URGENCY-02"
    assert "Non-safety workflow" in score.basis


def test_urgency_score_3(scoring_engine):
    """Test 5: Clinical Urgency Score 3 (Monitoring frequency or diagnostic change)."""
    score = scoring_engine.score_urgency("monitoring_change")
    assert score is not None
    assert score.score == 3
    assert score.rule_id == "URGENCY-03"
    assert "Monitoring frequency" in score.basis


def test_urgency_score_4(scoring_engine):
    """Test 6: Clinical Urgency Score 4 (Time-sensitive treatment/dosage change)."""
    score = scoring_engine.score_urgency("dosage_change")
    assert score is not None
    assert score.score == 4
    assert score.rule_id == "URGENCY-04"
    assert "Time-sensitive treatment" in score.basis


def test_urgency_score_5(scoring_engine):
    """Test 7: Clinical Urgency Score 5 (Contraindication / harm risk)."""
    score = scoring_engine.score_urgency("contraindication")
    assert score is not None
    assert score.score == 5
    assert score.rule_id == "URGENCY-05"
    assert "Safety warning or contraindication" in score.basis


# ==============================================================================
# TESTS 8–12: EVIDENCE STRENGTH SCORES 1 THROUGH 5
# ==============================================================================

def test_evidence_score_1(scoring_engine):
    """Test 8: Evidence Strength Score 1 (Safety notice or manufacturer communication)."""
    score = scoring_engine.score_evidence("manufacturer_communication")
    assert score is not None
    assert score.score == 1
    assert score.rule_id == "EVIDENCE-01"
    assert "Safety notice" in score.basis


def test_evidence_score_2(scoring_engine):
    """Test 9: Evidence Strength Score 2 (Limited case-series, observational)."""
    score = scoring_engine.score_evidence("case_series")
    assert score is not None
    assert score.score == 2
    assert score.rule_id == "EVIDENCE-02"
    assert "Limited case-series" in score.basis


def test_evidence_score_3(scoring_engine):
    """Test 10: Evidence Strength Score 3 (Expert consensus supported by data)."""
    score = scoring_engine.score_evidence("expert_consensus")
    assert score is not None
    assert score.score == 3
    assert score.rule_id == "EVIDENCE-03"
    assert "Expert consensus" in score.basis


def test_evidence_score_4(scoring_engine):
    """Test 11: Evidence Strength Score 4 (High-quality observational or synthesis)."""
    score = scoring_engine.score_evidence("high_quality_observational")
    assert score is not None
    assert score.score == 4
    assert score.rule_id == "EVIDENCE-04"
    assert "High-quality observational" in score.basis


def test_evidence_score_5(scoring_engine):
    """Test 12: Evidence Strength Score 5 (Class I / Grade 1A systematic review or RCT)."""
    score = scoring_engine.score_evidence("Grade A")
    assert score is not None
    assert score.score == 5
    assert score.rule_id == "EVIDENCE-05"
    assert "Class I / Grade 1A" in score.basis


# ==============================================================================
# TESTS 13–17: PATHWAY BREADTH SCORES 1 THROUGH 5
# ==============================================================================

def test_breadth_score_1(scoring_engine):
    """Test 13: Pathway Breadth Score 1 (Sub-specialty or edge population only)."""
    score = scoring_engine.score_breadth("sub_specialty")
    assert score is not None
    assert score.score == 1
    assert score.rule_id == "BREADTH-01"
    assert "Sub-specialty" in score.basis


def test_breadth_score_2(scoring_engine):
    """Test 14: Pathway Breadth Score 2 (One care team or workflow)."""
    score = scoring_engine.score_breadth("one_care_team")
    assert score is not None
    assert score.score == 2
    assert score.rule_id == "BREADTH-02"
    assert "One care team" in score.basis


def test_breadth_score_3(scoring_engine):
    """Test 15: Pathway Breadth Score 3 (One specialty, multiple care pathways)."""
    score = scoring_engine.score_breadth("one_specialty")
    assert score is not None
    assert score.score == 3
    assert score.rule_id == "BREADTH-03"
    assert "One specialty" in score.basis


def test_breadth_score_4(scoring_engine):
    """Test 16: Pathway Breadth Score 4 (Two or three specialties)."""
    score = scoring_engine.score_breadth("two_or_three_specialties")
    assert score is not None
    assert score.score == 4
    assert score.rule_id == "BREADTH-04"
    assert "Two or three specialties" in score.basis


def test_breadth_score_5(scoring_engine):
    """Test 17: Pathway Breadth Score 5 (Four or more specialties or organisation-wide)."""
    score = scoring_engine.score_breadth("four_or_more_specialties")
    assert score is not None
    assert score.score == 5
    assert score.rule_id == "BREADTH-05"
    assert "Four or more specialties" in score.basis


# ==============================================================================
# TESTS 18–22: TOTAL CALCULATION AND TIER BOUNDARIES
# ==============================================================================

def test_total_calculation(scoring_engine):
    """Test 18: Total score equals sum of urgency + evidence + breadth."""
    result = scoring_engine.calculate_impact(
        urgency_input="dosage_change",      # 4
        evidence_input="Grade A",           # 5
        breadth_input="one_specialty",      # 3
    )
    assert result.is_complete is True
    assert result.total_score == 4 + 5 + 3
    assert result.total_score == 12


def test_critical_boundary_12(scoring_engine):
    """Test 19: Critical boundary = 12 (48 hours SLA), score 11 is High (7 days)."""
    # Total = 12 (4 + 5 + 3) -> Critical, 48 hours
    res_12 = scoring_engine.calculate_impact("dosage_change", "Grade A", "one_specialty")
    assert res_12.total_score == 12
    assert res_12.tier == ImpactTier.CRITICAL
    assert res_12.sla_hours == 48

    # Total = 11 (3 + 5 + 3) -> High, 168 hours (7 days)
    res_11 = scoring_engine.calculate_impact("monitoring_change", "Grade A", "one_specialty")
    assert res_11.total_score == 11
    assert res_11.tier == ImpactTier.HIGH
    assert res_11.sla_hours == 168


def test_high_boundary_8(scoring_engine):
    """Test 20: High boundary = 8 (7 days SLA), score 7 is Standard (30 days)."""
    # Total = 8 (3 + 3 + 2) -> High
    res_8 = scoring_engine.calculate_impact("monitoring_change", "expert_consensus", "one_care_team")
    assert res_8.total_score == 8
    assert res_8.tier == ImpactTier.HIGH
    assert res_8.sla_hours == 168

    # Total = 7 (2 + 3 + 2) -> Standard
    res_7 = scoring_engine.calculate_impact("workflow_change", "expert_consensus", "one_care_team")
    assert res_7.total_score == 7
    assert res_7.tier == ImpactTier.STANDARD
    assert res_7.sla_hours == 720


def test_standard_boundary_4(scoring_engine):
    """Test 21: Standard boundary = 4 (30 days SLA)."""
    # Total = 4 (2 + 1 + 1) -> Standard
    res_4 = scoring_engine.calculate_impact("workflow_change", "safety_notice", "sub_specialty")
    assert res_4.total_score == 4
    assert res_4.tier == ImpactTier.STANDARD
    assert res_4.sla_hours == 720


def test_low_boundary_3(scoring_engine):
    """Test 22: Low boundary = 3 (90 days SLA)."""
    # Total = 3 (1 + 1 + 1) -> Low
    res_3 = scoring_engine.calculate_impact("safety_notice", "safety_notice", "sub_specialty")
    assert res_3.total_score == 3
    assert res_3.tier == ImpactTier.LOW
    assert res_3.sla_hours == 2160


# ==============================================================================
# TESTS 23–26: MISSING / UNMAPPED VALUES AND NO GUESSING
# ==============================================================================

def test_incomplete_missing_dimension(scoring_engine):
    """Test 23: Missing dimension produces status=incomplete and no total."""
    result = scoring_engine.calculate_impact(
        urgency_input="dosage_change",
        evidence_input="Grade A",
        breadth_input=None,  # Missing breadth
    )
    assert result.is_complete is False
    assert result.status == ImpactStatus.INCOMPLETE
    assert result.total_score is None
    assert result.tier is None
    assert result.sla_deadline is None
    assert "Missing pathway breadth classification" in result.incomplete_reason


def test_unmapped_value(scoring_engine):
    """Test 24: Unmapped value is detected and preserved in incomplete_reason."""
    result = scoring_engine.calculate_impact(
        urgency_input="dosage_change",
        evidence_input="UNSUPPORTED_GRADE_XYZ",
        breadth_input="one_care_team",
    )
    assert result.is_complete is False
    assert result.status == ImpactStatus.INCOMPLETE
    assert result.total_score is None
    assert "Unsupported evidence grade 'UNSUPPORTED_GRADE_XYZ'" in result.incomplete_reason


def test_no_guessed_score(scoring_engine):
    """Test 25: Unmapped inputs are never defaulted to 1, 3, or 5."""
    score = scoring_engine.score_evidence("non_existent_evidence_type")
    assert score is None

    score_urg = scoring_engine.score_urgency("unrecognized_urgency")
    assert score_urg is None

    score_brd = scoring_engine.score_breadth("unrecognized_breadth")
    assert score_brd is None


def test_no_tier_for_incomplete_score(scoring_engine):
    """Test 26: No tier or SLA is ever calculated for incomplete impact scores."""
    result = scoring_engine.calculate_impact(
        urgency_input=None,
        evidence_input=None,
        breadth_input=None,
    )
    assert result.status == ImpactStatus.INCOMPLETE
    assert result.total_score is None
    assert result.tier is None
    assert result.sla_deadline is None
    assert result.sla_hours is None
    assert result.routing_target is None


# ==============================================================================
# TESTS 27–31: PERSISTENCE, RULE IDS, REASON, AND PARENT RELATIONSHIP
# ==============================================================================

def test_rule_id_persisted(impact_env):
    """Test 27: Verify stable rule IDs are persisted in ImpactRecord.rule_ids."""
    session_factory = impact_env["session_factory"]
    agent = impact_env["agent"]

    gap_id = _create_sample_gap_record(session_factory)
    impact = agent.process_gap_record(
        gap_record_id=gap_id,
        urgency_input="dosage_change",
        evidence_input="Grade A",
        breadth_input="one_specialty",
    )

    assert "URGENCY-04" in impact.rule_ids
    assert "EVIDENCE-05" in impact.rule_ids
    assert "BREADTH-03" in impact.rule_ids


def test_yaml_version_persisted(impact_env):
    """Test 28: Verify scoring_yaml_version is persisted."""
    session_factory = impact_env["session_factory"]
    agent = impact_env["agent"]

    gap_id = _create_sample_gap_record(session_factory)
    impact = agent.process_gap_record(
        gap_record_id=gap_id,
        urgency_input="dosage_change",
        evidence_input="Grade A",
        breadth_input="one_specialty",
    )

    assert impact.scoring_yaml_version == "1.0"


def test_written_basis_persisted(impact_env):
    """Test 29: Verify written clinical bases are persisted for all dimensions."""
    session_factory = impact_env["session_factory"]
    agent = impact_env["agent"]

    gap_id = _create_sample_gap_record(session_factory)
    impact = agent.process_gap_record(
        gap_record_id=gap_id,
        urgency_input="dosage_change",
        evidence_input="Grade A",
        breadth_input="one_specialty",
    )

    assert "Time-sensitive treatment" in impact.urgency_basis
    assert "Class I / Grade 1A" in impact.evidence_basis
    assert "One specialty" in impact.breadth_basis


def test_impact_record_persistence(impact_env):
    """Test 30: Verify complete ImpactRecord persistence in SQLite."""
    session_factory = impact_env["session_factory"]
    agent = impact_env["agent"]

    gap_id = _create_sample_gap_record(session_factory)
    impact = agent.process_gap_record(
        gap_record_id=gap_id,
        urgency_input="dosage_change",
        evidence_input="Grade A",
        breadth_input="one_specialty",
    )

    with session_factory() as session:
        queried = session.get(ImpactRecord, impact.id)
        assert queried is not None
        assert queried.total_score == 12
        assert queried.tier == "Critical"
        assert queried.status == "calculated"
        assert queried.routing_target == "Rapid Governance Committee"
        assert queried.sla_deadline is not None


def test_parent_gap_record_relationship(impact_env):
    """Test 31: Verify parent GapRecord relationship is navigable."""
    session_factory = impact_env["session_factory"]
    agent = impact_env["agent"]

    gap_id = _create_sample_gap_record(session_factory)
    impact = agent.process_gap_record(
        gap_record_id=gap_id,
        urgency_input="dosage_change",
        evidence_input="Grade A",
        breadth_input="one_specialty",
    )

    with session_factory() as session:
        gap = session.get(GapRecord, gap_id)
        assert gap.impact_record is not None
        assert gap.impact_record.id == impact.id


# ==============================================================================
# TESTS 32–34: HISTORICAL VERSIONING AND TIER IMMUTABILITY
# ==============================================================================

def test_historical_score_version_preservation(impact_env, tmp_path):
    """Test 32: Verify updating scoring YAML rules does not rewrite historical score snapshots."""
    session_factory = impact_env["session_factory"]
    agent = impact_env["agent"]

    # 1. Score record under version 1.0
    gap_id_1 = _create_sample_gap_record(session_factory)
    impact_1 = agent.process_gap_record(
        gap_record_id=gap_id_1,
        urgency_input="dosage_change",
        evidence_input="Grade A",
        breadth_input="one_specialty",
    )
    assert impact_1.scoring_yaml_version == "1.0"

    # 2. Create version 1.1 YAML in tmp_path
    v1_1_yaml = tmp_path / "scoring_v1_1.yaml"
    with open("./config/scoring.yaml", "r", encoding="utf-8") as f:
        content = f.read()
    v1_1_content = content.replace('version: "1.0"', 'version: "1.1"')
    v1_1_yaml.write_text(v1_1_content, encoding="utf-8")

    engine_v1_1 = ScoringEngine(yaml_path=v1_1_yaml)
    agent_v1_1 = ImpactAgent(session_factory=session_factory, scoring_engine=engine_v1_1)

    # 3. Score a new record under version 1.1
    gap_id_2 = _create_sample_gap_record(session_factory)
    impact_2 = agent_v1_1.process_gap_record(
        gap_record_id=gap_id_2,
        urgency_input="dosage_change",
        evidence_input="Grade A",
        breadth_input="one_specialty",
    )
    assert impact_2.scoring_yaml_version == "1.1"

    # 4. Verify historical record retains version 1.0 unchanged
    with session_factory() as session:
        historical = session.get(ImpactRecord, impact_1.id)
        assert historical.scoring_yaml_version == "1.0"


def test_automatic_tier_downgrade_blocked(impact_env):
    """Test 33: Verify automatic tier downgrade is strictly blocked."""
    session_factory = impact_env["session_factory"]
    agent = impact_env["agent"]

    # 1. Establish an existing High tier record (total 10: 4 + 3 + 3)
    gap_id = _create_sample_gap_record(session_factory)
    impact = agent.process_gap_record(
        gap_record_id=gap_id,
        urgency_input="dosage_change",     # 4
        evidence_input="expert_consensus",  # 3
        breadth_input="one_specialty",     # 3
    )
    assert impact.tier == "High"

    # 2. Re-scoring with lower inputs (total 6: Standard) without reviewer authorization
    with pytest.raises(TierDowngradeBlockedError) as exc_info:
        agent.process_gap_record(
            gap_record_id=gap_id,
            urgency_input="workflow_change",   # 2
            evidence_input="case_series",       # 2
            breadth_input="one_care_team",     # 2
        )

    assert "Automatic tier downgrade from 'High' to 'Standard' is blocked" in str(exc_info.value)

    # Verify database tier was NOT downgraded
    with session_factory() as session:
        refreshed = session.get(ImpactRecord, impact.id)
        assert refreshed.tier == "High"


def test_correction_requires_reviewer_and_rationale(impact_env):
    """Test 34: Authorized tier downgrade succeeds with named reviewer and rationale, logged to AuditLog."""
    session_factory = impact_env["session_factory"]
    agent = impact_env["agent"]

    # 1. Establish existing High tier record
    gap_id = _create_sample_gap_record(session_factory)
    impact = agent.process_gap_record(
        gap_record_id=gap_id,
        urgency_input="dosage_change",     # 4
        evidence_input="expert_consensus",  # 3
        breadth_input="one_specialty",     # 3
    )
    assert impact.tier == "High"

    # 2. Perform authorized downgrade with named reviewer and clinical rationale
    downgraded = agent.process_gap_record(
        gap_record_id=gap_id,
        urgency_input="workflow_change",   # 2
        evidence_input="case_series",       # 2
        breadth_input="one_care_team",     # 2
        reviewer="Dr. Eleanor Vance",
        correction_rationale="Multidisciplinary review confirmed workflow change only with limited cohort impact.",
    )

    assert downgraded.tier == "Standard"
    assert downgraded.total_score == 6

    # 3. Verify an append-only AuditLog record was persisted
    with session_factory() as session:
        audit_records = (
            session.query(AuditLog)
            .filter_by(entity_id=downgraded.id, entity_type="ImpactRecord")
            .all()
        )
        assert len(audit_records) == 1
        audit = audit_records[0]
        assert audit.actor == "Dr. Eleanor Vance"
        assert audit.previous_status == "High"
        assert audit.new_status == "Standard"
        assert "Multidisciplinary review" in audit.reason


# ==============================================================================
# TESTS 35–38: UNRESOLVED COMPARISON, G3 NO-MATCH, IDEMPOTENCY, ZERO LLM CALLS
# ==============================================================================

def test_unresolved_comparison_finding_cannot_be_scored(impact_env):
    """Test 35: Verify unresolved G2 comparison findings raise UnresolvedComparisonError."""
    session_factory = impact_env["session_factory"]
    agent = impact_env["agent"]

    unresolved_gap_id = _create_sample_gap_record(
        session_factory=session_factory,
        change_status=ChangeStatus.HELD_FOR_G2.value,
        gap_status=GapStatus.REVIEW_REQUIRED.value,
    )

    with pytest.raises(UnresolvedComparisonError) as exc_info:
        agent.process_gap_record(
            gap_record_id=unresolved_gap_id,
            urgency_input="dosage_change",
            evidence_input="Grade A",
            breadth_input="one_specialty",
        )

    assert "held for G2 review" in str(exc_info.value)


def test_g3_no_match_finding_is_scoreable(impact_env):
    """Test 36: Verify G3 confirmed no-match finding is scoreable without inventing a protocol match."""
    session_factory = impact_env["session_factory"]
    agent = impact_env["agent"]

    g3_gap_id = _create_sample_gap_record(
        session_factory=session_factory,
        difference_type=DifferenceType.NO_MATCH.value,
        change_status=ChangeStatus.HELD_FOR_G3.value,
        gap_status=GapStatus.NO_MATCH.value,
        is_match=False,
        comparison_result="no_match",
    )

    impact = agent.process_gap_record(
        gap_record_id=g3_gap_id,
        urgency_input="time_sensitive_treatment",  # 4
        evidence_input="Class I",                  # 5
        breadth_input="two_or_three_specialties",  # 4
    )

    assert impact.total_score == 13
    assert impact.tier == "Critical"
    assert impact.status == "calculated"

    # Verify no protocol was invented on the parent GapRecord
    with session_factory() as session:
        gap = session.get(GapRecord, g3_gap_id)
        assert gap.is_match is False
        assert gap.matched_protocol_id is None


def test_idempotent_impact_scoring_updates_existing_record(impact_env):
    """Test 37: Verify re-running impact scoring on the same GapRecord updates in-place without duplicates."""
    session_factory = impact_env["session_factory"]
    agent = impact_env["agent"]

    gap_id = _create_sample_gap_record(session_factory)

    impact1 = agent.process_gap_record(
        gap_record_id=gap_id,
        urgency_input="monitoring_change",
        evidence_input="expert_consensus",
        breadth_input="one_care_team",
    )

    # Re-score with an upgraded breadth (same or higher tier)
    impact2 = agent.process_gap_record(
        gap_record_id=gap_id,
        urgency_input="monitoring_change",
        evidence_input="expert_consensus",
        breadth_input="two_or_three_specialties",
    )

    assert impact1.id == impact2.id

    with session_factory() as session:
        all_impacts = session.query(ImpactRecord).filter_by(gap_record_id=gap_id).all()
        assert len(all_impacts) == 1


def test_zero_llm_calls_made(impact_env, monkeypatch):
    """Test 38: Verify ImpactAgent executes deterministically without any LLM calls."""
    session_factory = impact_env["session_factory"]
    agent = impact_env["agent"]

    # If any OpenAI/Groq call were attempted, fail immediately
    import openai
    def fail_llm(*args, **kwargs):
        raise AssertionError("LLM call attempted inside deterministic Impact Agent!")

    monkeypatch.setattr(openai.resources.chat.completions.Completions, "create", fail_llm)

    gap_id = _create_sample_gap_record(session_factory)
    impact = agent.process_gap_record(
        gap_record_id=gap_id,
        urgency_input="dosage_change",
        evidence_input="Grade A",
        breadth_input="one_specialty",
    )

    assert impact.total_score == 12
    assert impact.tier == "Critical"
