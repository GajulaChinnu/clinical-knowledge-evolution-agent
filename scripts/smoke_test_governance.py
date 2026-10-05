"""Smoke test for Phase 9 Governance Agent and Human Gates.

Demonstrates:
1. Existing draft ChangeBrief
2. Reviewer assignment (draft -> assigned)
3. Review start (assigned -> in_review)
4. Defer with mandatory future follow-up date (in_review -> deferred)
5. Re-review (deferred -> in_review)
6. Explicit approve with clinical rationale (in_review -> decided)
7. Closure (decided -> closed)
8. Audit history verification
9. Zero LLM calls verification
10. Protocol data unchanged
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import openai
from sqlalchemy import create_engine

from app.agents.governance_agent import GovernanceAgent
from app.models.database import Base, get_session_factory
from app.models.entities import (
    AuditLog,
    ChangeBrief,
    ChangeRecord,
    GapRecord,
    ImpactRecord,
    IngestedDocument,
    Notification,
    ReviewAssignment,
)
from app.schemas.briefs import BriefStatus
from app.schemas.governance import ReviewDecision
from app.schemas.impact import ImpactStatus, ImpactTier
from app.services.reviewer_authorization import ReviewerAuthorizationService


def run_smoke_test():
    # 0. Monkeypatch OpenAI to prove ZERO LLM calls occur
    def fail_llm(*args, **kwargs):
        raise AssertionError("CRITICAL SAFETY BREACH: LLM call attempted in GovernanceAgent!")

    openai.resources.chat.completions.Completions.create = fail_llm

    # Setup in-memory SQLite database
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_factory = get_session_factory(engine=engine)

    auth_service = ReviewerAuthorizationService()
    agent = GovernanceAgent(session_factory=session_factory, auth_service=auth_service)

    unique_id = uuid.uuid4().hex
    initial_protocol_text = "Metformin 500mg BID is standard initial therapy. Target HbA1c < 7.0%."

    # 1. Existing draft ChangeBrief
    with session_factory() as session:
        doc = IngestedDocument(
            source_identifier=f"ADA-Guidelines-2026.pdf",
            source_path="/data/sources/ADA-Guidelines-2026.pdf",
            sha256_hash=(unique_id * 2)[:64],
            status="processed",
        )
        session.add(doc)
        session.flush()

        change = ChangeRecord(
            ingested_document_id=doc.id,
            verbatim_text="In adults with type 2 diabetes and high CVD risk, initiate SGLT2i or GLP-1 RA.",
            recommendation_type="pharmacotherapy",
            target_population="Adults with T2D and high CVD risk",
            intervention="SGLT2 inhibitor or GLP-1 RA",
            evidence_grade="Grade A",
            page=24,
            section="Pharmacologic Therapy",
            source_excerpt="In adults with type 2 diabetes and high CVD risk, initiate SGLT2i or GLP-1 RA.",
            extraction_model_version="openai/gpt-oss-20b",
            extraction_prompt_version="1.0",
            status="gap_confirmed",
            confidence=0.96,
        )
        session.add(change)
        session.flush()

        gap = GapRecord(
            change_record_id=change.id,
            candidate_protocol_section_ids=["PROT-DM-001__1.0__SEC-3"],
            similarity=0.91,
            comparison_result="gap",
            comparison_confidence=0.95,
            difference_type="dosage_change",
            matched_protocol_id="PROT-DM-001",
            matched_protocol_version="1.0",
            is_match=True,
            status="matched",
            schema_version="1.0",
        )
        session.add(gap)
        session.flush()

        deadline = datetime.now(timezone.utc) + timedelta(hours=48)
        impact = ImpactRecord(
            gap_record_id=gap.id,
            clinical_urgency=4,
            evidence_strength=5,
            pathway_breadth=3,
            rule_ids={"clinical_urgency": "URGENCY-04", "evidence_strength": "EVIDENCE-05", "pathway_breadth": "BREADTH-03"},
            scoring_yaml_version="1.0",
            total_score=12,
            tier=ImpactTier.CRITICAL.value,
            routing_target="Rapid Governance Committee",
            sla_deadline=deadline,
            urgency_basis="Time-sensitive treatment change",
            evidence_basis="Grade A systematic review / RCT",
            breadth_basis="One specialty, multiple care pathways",
            status=ImpactStatus.CALCULATED.value,
            schema_version="1.0",
        )
        session.add(impact)
        session.flush()

        brief = ChangeBrief(
            id=str(uuid.uuid4()),
            impact_record_id=impact.id,
            status=BriefStatus.DRAFT.value,
            rendered_file_path=f"/data/output/brief_{unique_id[:8]}.md",
            rendered_file_hash=(unique_id * 2)[:64],
            structured_payload={"protocol_text": initial_protocol_text},
            schema_version="1.0",
        )
        session.add(brief)
        session.commit()
        brief_id = brief.id

    print("=== PHASE 9 GOVERNANCE SMOKE TEST ===")
    print(f"1. Created initial ChangeBrief: {brief_id} (Status: {BriefStatus.DRAFT.value})")

    # 2. Reviewer assignment (draft -> assigned)
    assignment = agent.assign_reviewer(
        change_brief_id=brief_id,
        reviewer_id="dr_smith",
        reviewer_role="Clinical Governance Lead",
        actor="governance_coordinator",
    )
    assert assignment.status == "assigned"
    print(f"2. Assigned reviewer '{assignment.reviewer_id}' ({assignment.reviewer_role}). Status: assigned")

    # 3. Review start (assigned -> in_review)
    brief = agent.start_review(change_brief_id=brief_id, reviewer_id="dr_smith")
    assert brief.status == "in_review"
    print("3. Review started by clinician. Brief status: in_review")

    # 4. Defer (in_review -> deferred)
    future_date = datetime.now(timezone.utc) + timedelta(days=60)
    brief = agent.defer(
        change_brief_id=brief_id,
        reviewer_id="dr_smith",
        rationale="Awaiting published trial sub-group analysis for renal safety profile.",
        defer_follow_up_date=future_date,
    )
    assert brief.status == "deferred"
    print(f"4. Brief deferred with future due date ({future_date.strftime('%Y-%m-%d')}). Status: deferred (remains open)")

    # 5. Re-review (deferred -> in_review)
    brief = agent.start_review(change_brief_id=brief_id, reviewer_id="dr_smith")
    assert brief.status == "in_review"
    print("5. Re-entered review following sub-group analysis release. Status: in_review")

    # 6. Explicit approve (in_review -> decided)
    brief = agent.decide(
        change_brief_id=brief_id,
        reviewer_id="dr_smith",
        decision=ReviewDecision.APPROVE,
        rationale="Sub-group analysis confirms robust renal protection and cardiovascular risk reduction.",
    )
    assert brief.status == "decided"
    print("6. Human decision recorded: APPROVE. Status: decided")

    # 7. Closure (decided -> closed)
    brief = agent.close_after_decision(
        change_brief_id=brief_id,
        actor="governance_chair",
        closure_note="Approved protocol modification authorized for order set deployment.",
    )
    assert brief.status == "closed"
    print("7. ChangeBrief closed following explicit governance decision. Status: closed")

    # 8. Audit history
    with session_factory() as session:
        audits = (
            session.query(AuditLog)
            .filter_by(entity_id=brief_id)
            .order_by(AuditLog.timestamp.asc())
            .all()
        )
        print(f"8. Audit History ({len(audits)} immutable log entries recorded):")
        for idx, a in enumerate(audits, 1):
            action = a.audit_metadata.get("action", "unknown") if a.audit_metadata else "transition"
            print(f"   [{idx}] Action: {action:12} | Status: {a.previous_status} -> {a.new_status} | Actor: {a.actor} | Reason: {a.reason[:50]}...")

    # 9. Verify zero LLM calls
    print("9. Zero LLM/API calls verified: YES (OpenAI client monkeypatched to fail on invocation)")

    # 10. Verify protocol data unchanged
    with session_factory() as session:
        final_brief = session.query(ChangeBrief).filter_by(id=brief_id).first()
        assert final_brief.structured_payload["protocol_text"] == initial_protocol_text
        print("10. Protocol data immutability verified: YES (verbatim protocol text unchanged)")

    print("\n=== GOVERNANCE SMOKE TEST COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    run_smoke_test()
