"""Smoke test for Phase 11 SLA Scheduler and G5 Escalation.

Demonstrates:
1. Create an unresolved review assignment with expired SLA deadline
2. Run single SLA check (run_once)
3. Verify G5 escalation detection
4. Verify escalation Notification persisted
5. Verify AuditLog event recorded
6. Verify governance decision remains empty (no auto-decision)
7. Run second SLA check to prove idempotency (no duplicate escalation)
8. Resolve brief explicitly via GovernanceAgent (start review -> decide -> close)
9. Run third SLA check to verify resolved brief is skipped and not escalated
10. Confirm institutional protocol data remains strictly immutable
11. Confirm ZERO direct LLM / API calls occurred
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import uuid

# Ensure project root is in sys.path
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
from app.schemas.governance import ReviewAssignmentStatus, ReviewDecision
from app.schemas.impact import ImpactStatus, ImpactTier
from app.services.config_service import AppConfig
from app.services.reviewer_authorization import ReviewerAuthorizationService
from app.services.sla_scheduler import SLAScheduler


def run_smoke_test():
    print("=" * 75)
    print(" CKEA PHASE 11: SLA SCHEDULER & G5 ESCALATION SMOKE TEST")
    print("=" * 75)

    # 11. Verify zero LLM calls by failing on any invocation
    def fail_llm_call(*args, **kwargs):
        raise AssertionError("CRITICAL SAFETY BREACH: LLM API call attempted during SLA Scheduler execution!")

    openai.resources.chat.completions.Completions.create = fail_llm_call

    # Set up clean in-memory database
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_factory = get_session_factory(engine=engine)

    # Baseline synthetic protocol text for immutability check
    baseline_protocol_text = (
        "PROT-CARD-001 v1.0 Section 2.1: SGLT2 inhibitors are recommended in patients with heart failure."
    )

    config = AppConfig(
        database_url="sqlite:///:memory:",
        sla_scheduler_enabled=True,
        sla_check_interval_minutes=60,
        sla_timezone="UTC",
    )

    auth_service = ReviewerAuthorizationService()
    gov_agent = GovernanceAgent(session_factory=session_factory, auth_service=auth_service, config=config)
    scheduler = SLAScheduler(
        session_factory=session_factory,
        governance_agent=gov_agent,
        config=config,
    )

    now_utc = datetime.now(timezone.utc)
    expired_deadline = now_utc - timedelta(hours=4)  # 4 hours overdue

    # 1. Create entities with an already-expired SLA deadline
    print("\n1. Setting up synthetic unresolved ChangeBrief with an expired SLA deadline...")
    with session_factory() as session:
        doc = IngestedDocument(
            source_identifier="ACC-Guideline-2026.pdf",
            source_path="/data/sources/ACC-Guideline-2026.pdf",
            sha256_hash="c" * 64,
            status="processed",
        )
        session.add(doc)
        session.flush()

        change = ChangeRecord(
            ingested_document_id=doc.id,
            verbatim_text="Initiate dapagliflozin 10mg once daily in symptomatic heart failure.",
            recommendation_type="pharmacotherapy",
            target_population="Adults with HFrEF",
            intervention="Dapagliflozin 10mg daily",
            evidence_grade="Grade A",
            page=14,
            section="Pharmacotherapy",
            source_excerpt="Initiate dapagliflozin 10mg once daily in symptomatic heart failure.",
            extraction_model_version="openai/gpt-oss-20b",
            extraction_prompt_version="1.0",
            confidence=0.96,
            status="gap_confirmed",
        )
        session.add(change)
        session.flush()

        gap = GapRecord(
            change_record_id=change.id,
            candidate_protocol_section_ids=["PROT-CARD-001__1.0__SEC-2"],
            similarity=0.92,
            comparison_result="gap",
            comparison_confidence=0.95,
            difference_type="intervention_change",
            matched_protocol_id="PROT-CARD-001",
            matched_protocol_version="1.0",
            is_match=True,
            status="matched",
            schema_version="1.0",
        )
        session.add(gap)
        session.flush()

        impact = ImpactRecord(
            gap_record_id=gap.id,
            clinical_urgency=4,
            evidence_strength=5,
            pathway_breadth=3,
            rule_ids={"clinical_urgency": "URGENCY-04"},
            scoring_yaml_version="1.0",
            total_score=12,
            tier=ImpactTier.CRITICAL.value,
            routing_target="Rapid Cardiology Committee",
            sla_deadline=expired_deadline,
            urgency_basis="Time-sensitive cardiovascular intervention",
            evidence_basis="Grade A RCT evidence",
            breadth_basis="Multiple cardiology care pathways",
            status=ImpactStatus.CALCULATED.value,
            schema_version="1.0",
        )
        session.add(impact)
        session.flush()

        brief = ChangeBrief(
            id=str(uuid.uuid4()),
            impact_record_id=impact.id,
            status=BriefStatus.ASSIGNED.value,
            rendered_file_path="/data/output/brief_card_01.md",
            rendered_file_hash="d" * 64,
            structured_payload={"protocol_text": baseline_protocol_text},
            schema_version="1.0",
        )
        session.add(brief)
        session.flush()

        assignment = ReviewAssignment(
            id=str(uuid.uuid4()),
            change_brief_id=brief.id,
            reviewer_role="Clinical Governance Lead",
            reviewer_id="dr_smith",
            status=ReviewAssignmentStatus.ASSIGNED.value,
            due_date=expired_deadline,
            decision=None,
            rationale=None,
            schema_version="1.0",
        )
        session.add(assignment)
        session.commit()

        brief_id = brief.id
        assignment_id = assignment.id

    print(f"   ChangeBrief ID:   {brief_id} (Status: assigned)")
    print(f"   Assignment ID:    {assignment_id} (Reviewer: dr_smith)")
    print(f"   SLA Deadline:     {expired_deadline.isoformat()} (OVERDUE by 4 hours)")

    # 2. Run first SLA check
    print("\n2. Executing first SLA check cycle (run_once)...")
    summary1 = scheduler.run_once(reference_time=now_utc)

    print(f"   Cycle Outcome:    Evaluated: {summary1.evaluated_count} | Overdue: {summary1.overdue_count} | Escalated: {summary1.escalated_count}")
    assert summary1.evaluated_count == 1
    assert summary1.overdue_count == 1
    assert summary1.escalated_count == 1
    assert summary1.already_escalated_count == 0

    # 3-6. Verify Escalation state, Notification, AuditLog, and No Auto-Decision
    print("\n3-6. Verifying G5 Escalation artifacts and human gate boundaries...")
    with session_factory() as session:
        assignment_db = session.get(ReviewAssignment, assignment_id)
        brief_db = session.get(ChangeBrief, brief_id)
        notif = session.query(Notification).filter_by(related_entity_id=brief_id).first()
        audits = session.query(AuditLog).filter_by(entity_id=brief_id).all()

        print(f"   Assignment Status:   {assignment_db.status} (Expected: escalated)")
        print(f"   Decision on Brief:   {assignment_db.decision} (Expected: None / Unset)")
        print(f"   Brief Status:        {brief_db.status} (Expected: assigned, NOT auto-closed)")
        print(f"   Escalation Notif:    Recipient='{notif.recipient}' | Type='{notif.notification_type}'")
        print(f"   Audit Log Count:     {len(audits)} entry/entries recorded")

        assert assignment_db.status == ReviewAssignmentStatus.ESCALATED.value
        assert assignment_db.decision is None, "CRITICAL ERROR: SLA breach automatically made a decision!"
        assert brief_db.status == BriefStatus.ASSIGNED.value
        assert notif is not None
        assert notif.notification_type == "escalation"
        assert notif.recipient == "Rapid Cardiology Committee"
        assert len(audits) >= 1

    print("   Confirmed: SLA breach triggered G5 notification without inferring any decision.")

    # 7-9. Run second SLA check (Verify Idempotency)
    print("\n7-9. Executing second SLA check cycle to verify idempotency...")
    summary2 = scheduler.run_once(reference_time=now_utc)

    print(f"   Cycle Outcome:    Evaluated: {summary2.evaluated_count} | Overdue: {summary2.overdue_count} | Newly Escalated: {summary2.escalated_count} | Already Escalated: {summary2.already_escalated_count}")
    assert summary2.evaluated_count == 1
    assert summary2.overdue_count == 1
    assert summary2.escalated_count == 0
    assert summary2.already_escalated_count == 1

    with session_factory() as session:
        notif_count = session.query(Notification).filter_by(related_entity_id=brief_id).count()
        assert notif_count == 1, f"Expected 1 notification, found {notif_count} (duplicate created!)"

    print("   Confirmed: Repeated SLA check produced zero duplicate notifications or escalations.")

    # 10. Explicit human resolution
    print("\n10. Explicit human review and decision by authorized clinician (dr_smith)...")
    gov_agent.start_review(change_brief_id=brief_id, reviewer_id="dr_smith")
    gov_agent.decide(
        change_brief_id=brief_id,
        reviewer_id="dr_smith",
        decision=ReviewDecision.APPROVE,
        rationale="Urgent HFrEF guideline evidence reviewed and approved for clinical protocol update.",
    )
    gov_agent.close_after_decision(
        change_brief_id=brief_id,
        actor="governance_chair",
        closure_note="Approved protocol update deployed to institutional EHR order sets.",
    )

    with session_factory() as session:
        resolved_brief = session.get(ChangeBrief, brief_id)
        assert resolved_brief.status == BriefStatus.CLOSED.value
    print(f"    Brief status following explicit resolution: {resolved_brief.status}")

    # 11-12. Run third SLA check (Verify Resolved Work Skipped)
    print("\n11-12. Executing third SLA check cycle on resolved brief...")
    summary3 = scheduler.run_once(reference_time=now_utc)

    print(f"    Cycle Outcome:    Evaluated: {summary3.evaluated_count} | Skipped: {summary3.skipped_count} | Escalated: {summary3.escalated_count}")
    assert summary3.evaluated_count == 0
    assert summary3.skipped_count == 1
    assert summary3.escalated_count == 0
    print("    Confirmed: Resolved brief is safely skipped and no longer subject to SLA escalation.")

    # 13. Protocol Immutability Check
    print("\n13. Protocol Immutability & Zero LLM Calls Check:")
    with session_factory() as session:
        final_brief = session.get(ChangeBrief, brief_id)
        assert final_brief.structured_payload["protocol_text"] == baseline_protocol_text
    print("    Baseline Protocol Text: Unchanged and unmutated.")
    print("    Zero LLM / Groq API Calls: Verified (assertion guard on openai API).")

    print("\n" + "=" * 75)
    print(" ALL PHASE 11 SMOKE TEST DEMONSTRATIONS COMPLETED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    run_smoke_test()
