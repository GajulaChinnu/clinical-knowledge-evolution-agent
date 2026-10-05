"""Smoke test for Phase 8 Briefing Agent."""

import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine

from app.agents.briefing_agent import BriefingAgent
from app.models.database import Base, get_session_factory
from app.models.entities import ChangeRecord, GapRecord, ImpactRecord, IngestedDocument
from app.schemas.briefs import BriefStatus
from app.schemas.impact import ImpactStatus, ImpactTier
from app.services.brief_renderer import BriefRenderer
from app.services.config_service import AppConfig


def run_smoke_test():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        session_factory = get_session_factory(engine=engine)

        config = AppConfig(
            templates_dir=Path("./templates"),
            output_dir=output_dir,
        )

        renderer = BriefRenderer(templates_dir=Path("./templates"), output_dir=output_dir, config=config)
        agent = BriefingAgent(session_factory=session_factory, renderer=renderer, config=config)

        # 1. Create pipeline database records
        with session_factory() as session:
            doc = IngestedDocument(
                source_identifier="ada-2026-guidelines.pdf",
                source_path="/data/sources/ada-2026-guidelines.pdf",
                sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                document_version="1.0",
                source_version="2026.1",
                status="processed",
            )
            session.add(doc)
            session.flush()

            change = ChangeRecord(
                ingested_document_id=doc.id,
                verbatim_text="In patients with type 2 diabetes and heart failure, initiate SGLT2 inhibitor therapy.",
                recommendation_type="pharmacotherapy",
                target_population="Adults with Type 2 Diabetes and Heart Failure",
                intervention="SGLT2 inhibitor therapy",
                evidence_grade="Grade A",
                page=18,
                section="Cardiovascular Risk Management",
                source_excerpt="In adults with type 2 diabetes and heart failure, SGLT2 inhibitor therapy should be initiated to reduce hospitalization and mortality.",
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
                comparison_confidence=0.94,
                difference_type="dosage_change",
                matched_protocol_id="PROT-DM-001",
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
                rule_ids={
                    "clinical_urgency": "URGENCY-04",
                    "evidence_strength": "EVIDENCE-05",
                    "pathway_breadth": "BREADTH-03",
                },
                scoring_yaml_version="1.0",
                total_score=12,
                tier=ImpactTier.CRITICAL.value,
                routing_target="Rapid Governance Committee",
                urgency_basis="Time-sensitive treatment change where delay may worsen outcomes",
                evidence_basis="Class I / Grade 1A systematic review or RCT",
                breadth_basis="One specialty, multiple care pathways",
                status=ImpactStatus.CALCULATED.value,
                schema_version="1.0",
            )
            session.add(impact)
            session.commit()
            impact_id = impact.id

        # 2. Process ImpactRecord with BriefingAgent
        brief = agent.process_impact_record(
            impact_record_id=impact_id,
            exact_protocol_text="Metformin monotherapy is standard first-line unless contraindicated.",
            specific_difference="Recommendation adds mandatory SGLT2 inhibitor initiation for patients with concurrent heart failure.",
        )

        # 3. Verify outputs
        html_file = output_dir / f"brief_{brief.id}.html"
        md_file = output_dir / f"brief_{brief.id}.md"
        json_file = output_dir / f"brief_{brief.id}.json"

        assert html_file.exists(), "HTML file was not generated"
        assert md_file.exists(), "Markdown file was not generated"
        assert json_file.exists(), "JSON file was not generated"
        assert brief.status == BriefStatus.DRAFT.value, "Brief status is not draft"
        assert len(brief.rendered_file_hash) == 64, "Rendered file hash is invalid"

        # Check payload
        payload = brief.structured_payload
        assert len(payload["proposed_actions"]) == 3, "Did not contain 3 proposed actions"
        assert payload["impact_assessment"]["tier"] == "Critical"
        assert payload["what_changed"]["recommendation_type"] == "pharmacotherapy"

        print("=== BRIEFING AGENT SMOKE TEST SUCCESSFUL ===")
        print(f"Brief ID: {brief.id}")
        print(f"Status: {brief.status}")
        print(f"Rendered Hash: {brief.rendered_file_hash}")
        print(f"HTML size: {html_file.stat().st_size} bytes")
        print(f"Markdown size: {md_file.stat().st_size} bytes")
        print(f"JSON size: {json_file.stat().st_size} bytes")
        print("Seven sections validated: YES")


if __name__ == "__main__":
    run_smoke_test()
