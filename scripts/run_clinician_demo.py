"""End-to-end CKEA demo on the SYNTHETIC corpus (problem statement + clinician Treatment Check).

1. Background surveillance: the watchlist is checked while only v1 of each guideline is published,
   then v2 is "published"; the Monitoring Agent detects it, changes are categorised and ranked.
2. Clinician Treatment Check: one query per verdict (consistent / updated / conflicts / insufficient).
3. Governance: the out-of-date protocol's change brief is routed to the department specialist, who
   records an explicit decision (a human action, performed here by a named demo reviewer).
4. The labelled evaluation runs and its report is written to data/evaluation/reports/.

Uses its own database in a temporary directory - never data/ckea.db.
Run:  .venv\\Scripts\\python.exe scripts/run_clinician_demo.py
"""

from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agents.briefing_agent import BriefingAgent  # noqa: E402
from app.agents.comparison_agent import ComparisonAgent  # noqa: E402
from app.agents.extraction_agent import ExtractionAgent  # noqa: E402
from app.agents.governance_agent import GovernanceAgent  # noqa: E402
from app.agents.impact_agent import ImpactAgent  # noqa: E402
from app.agents.monitoring_agent import MonitoringAgent  # noqa: E402
from app.models.database import get_engine, get_session_factory, init_db  # noqa: E402
from app.orchestration.clinician_query import ClinicianQueryWorkflow  # noqa: E402
from app.schemas.clinician_query import ClinicianQueryInput, PatientContext  # noqa: E402
from app.services.clinician_query_evaluation import run_clinician_evaluation, write_report  # noqa: E402
from app.services.config_service import load_config  # noqa: E402
from app.services.taxonomy import get_taxonomy  # noqa: E402
from app.services.watchlist import load_watchlist  # noqa: E402


def banner(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def main() -> int:
    # Fresh working directory outside the project (synced folders can lock directories).
    work = Path(tempfile.mkdtemp(prefix="ckea_clinician_demo_"))
    print(f"Demo working directory: {work}")
    (work / "sources").mkdir(parents=True)
    corpus = work / "corpus"
    shutil.copytree(ROOT / "data" / "corpus", corpus)
    held_back = [p for p in corpus.rglob("v2.0.md")]
    for p in held_back:  # v2 not yet "published"
        p.rename(p.with_suffix(".unpublished"))

    base = load_config()
    config = base.model_copy(update={"database_url": f"sqlite:///{work / 'demo.db'}", "source_dir": work / "sources",
                                     "output_dir": work / "briefs"})
    engine = get_engine(db_url=config.database_url)
    init_db(engine=engine)
    sf = get_session_factory(engine=engine)
    taxonomy = get_taxonomy()

    monitoring = MonitoringAgent(source_dir=config.source_dir, session_factory=sf, config=config)
    extraction = ExtractionAgent(session_factory=sf, config=config)
    impact = ImpactAgent(session_factory=sf, config=config)
    governance = GovernanceAgent(session_factory=sf, config=config)

    def surveillance(label: str) -> None:
        watchlist = load_watchlist(taxonomy=taxonomy, corpus_root=corpus)
        checks = monitoring.check_watchlist(watchlist)
        extraction.index_guidance_statements([v.document_id for c in checks for v in c.versions])
        impact.assess_guidance_changes()
        new = sum(len(c.new_document_ids) for c in checks)
        print(f"{label}: checked {len(checks)} sources, {new} new version(s) ingested")

    banner("1. Background surveillance")
    surveillance("Initial check (v1 published)")
    for p in held_back:
        p.with_suffix(".unpublished").rename(p)
    surveillance("Publishers release v2")
    print("\nTop of the ranked feed (latest versions only):")
    for c in impact.ranked_feed()[:8]:
        print(f"  {c.priority_score:5.1f}  {c.change_category:24s} {c.watchlist_id:28s} {', '.join(c.treatments or [])}")
    filtered = [c for c in impact.ranked_feed(include_filtered=True) if c.relevance_status != "active"]
    print(f"Filtered as not practice-changing (kept, recoverable): {len(filtered)}")

    banner("2. Clinician Treatment Check (one query per verdict)")
    workflow = ClinicianQueryWorkflow(
        session_factory=sf, config=config, monitoring_agent=monitoring, extraction_agent=extraction,
        comparison_agent=ComparisonAgent(session_factory=sf, config=config), impact_agent=impact,
        briefing_agent=BriefingAgent(session_factory=sf, config=config), governance_agent=governance,
        watchlist=load_watchlist(taxonomy=taxonomy, corpus_root=corpus), taxonomy=taxonomy,
    )
    demo_queries = [
        ("Endocrinology", "metformin 500 mg twice daily", None, {"egfr_band": ">=60"}),
        ("Endocrinology", "metformin 500 mg once daily", None, {"egfr_band": ">=60"}),
        ("Infectious Diseases", "levofloxacin 500 mg once daily", "low-severity community-acquired pneumonia", {}),
        ("Cardiology", "unicorn extract 5 mg", None, {}),
    ]
    updated_answer = None
    for dept, treatment, condition, ctx in demo_queries:
        answer = workflow.run(ClinicianQueryInput(department=dept, treatment=treatment, condition=condition,
                                                  context=PatientContext(**ctx)), actor="demo_clinician")
        print(f"\n[{dept}] {treatment}\n  -> {answer.verdict_label}\n     {answer.verdict_basis}")
        for v in answer.what_changed:
            if v.plan_matches_previous:
                print(f"     previous v{v.previous.version} ({v.previous.published_date}): {v.previous.excerpt}")
                print(f"     latest   v{v.latest.version} ({v.latest.published_date}): {v.latest.excerpt}")
        for g in answer.governance:
            print(f"     governance: {g.message}")
        print(f"     {answer.disclaimer}")
        if answer.verdict == "guidance_updated_follow_new_version":
            updated_answer = answer

    banner("3. Governance: specialist decision (human action)")
    brief_id = next(g.brief_id for g in updated_answer.governance if g.brief_id)
    governance.start_review(change_brief_id=brief_id, reviewer_id="dr_wilson")
    governance.decide(change_brief_id=brief_id, reviewer_id="dr_wilson", decision="approve",
                      rationale="Demo: adopt the v2.0 twice-daily starting dose; protocol owner to update PROT-DM-001.")
    after = workflow.run(ClinicianQueryInput(department="Endocrinology", treatment="metformin 500 mg once daily"),
                         actor="demo_clinician")
    print(f"Clinician now sees: {after.governance[0].message}")
    print("Note: CKEA never edits the protocol; the protocol owner publishes the new version.")

    banner("4. Evaluation")
    report = run_clinician_evaluation(workflow, sf, impact)
    path = write_report(report, ROOT / "data" / "evaluation" / "reports")
    print(f"Cases passed {report['cases_passed']}/{report['cases']}; citations verified "
          f"{report['citations_verified_rate']:.0%}; ungrounded answers {report['ungrounded_answers']}")
    print(f"Report: {path}")
    engine.dispose()
    return 0 if report["cases_passed"] == report["cases"] else 1


if __name__ == "__main__":
    sys.exit(main())
