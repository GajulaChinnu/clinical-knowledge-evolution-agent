"""Evaluation of the clinician Treatment Check and background surveillance on the SYNTHETIC corpus.

Measures: verdict accuracy per class, citation verification, ungrounded answers, latest-version
identification, abstention, identifier rejection, protocol position, specialist routing,
change-category detection, relevance filtering, duplicate linking and feed ordering.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.models.entities import ReviewAssignment
from app.schemas.clinician_query import ClinicianQueryInput, PatientContext, PatientIdentifierError

DEFAULT_CASES = Path("data/evaluation/clinician_queries.json")

# Expected source-evolution categories into each guideline's latest version (from data/corpus).
EXPECTED_CHANGE_CATEGORIES = {
    "endo-t2d-guideline": ["contraindication_added", "contraindication_added", "dose_change", "new_recommendation",
                           "threshold_change", "withdrawn"],
    "cardio-af-guideline": ["contraindication_added", "threshold_change", "withdrawn"],
    "neph-ckd-guideline": ["contraindication_added", "threshold_change", "withdrawn"],
    "id-cap-guideline": ["contraindication_added", "dose_change", "dose_change", "withdrawn"],
    "em-anaphylaxis-guideline": ["contraindication_added", "dose_change", "withdrawn"],
}


@dataclass
class CaseResult:
    case_id: str
    expected: Optional[str]
    actual: Optional[str]
    passed: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def run_clinician_evaluation(workflow, session_factory, impact_agent, cases_path: Path = DEFAULT_CASES) -> Dict[str, Any]:
    data = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    results: List[CaseResult] = []
    citations_total = citations_verified = ungrounded = latest_ok = latest_total = 0

    for case in data["cases"]:
        raw = ClinicianQueryInput(department=case["department"], treatment=case["treatment"],
                                  condition=case.get("condition"), context=PatientContext(**case.get("context", {})))
        if case.get("expect_rejected"):
            try:
                workflow.run(raw, actor="evaluation")
                results.append(CaseResult(case["id"], "rejected", "accepted", False, {"rejected": False}))
            except PatientIdentifierError:
                results.append(CaseResult(case["id"], "rejected", "rejected", True, {"rejected": True}))
            continue

        answer = workflow.run(raw, actor="evaluation")
        checks: Dict[str, bool] = {"verdict": answer.verdict == case["expected_verdict"]}
        citations_total += len(answer.citations)
        citations_verified += sum(1 for c in answer.citations if c.verified)
        if answer.verdict != "insufficient_grounded_evidence" and not any(c.verified for c in answer.citations):
            ungrounded += 1
        if answer.verdict == "insufficient_grounded_evidence":
            checks["abstained_without_citations"] = not answer.latest_guidance
        if case.get("expect_citation"):
            checks["citation"] = any(case["expect_citation"] in c.excerpt for c in answer.citations)
        if case.get("expect_previous_version"):
            checks["previous_version"] = any(v.previous and v.previous.version == case["expect_previous_version"]
                                             and v.plan_matches_previous for v in answer.what_changed)
        for source in answer.sources_checked:
            if source.latest_version:
                latest_total += 1
                latest_ok += int(source.error is None)
        for protocol_id, status in (case.get("expect_protocol") or {}).items():
            checks[f"protocol:{protocol_id}"] = any(p.protocol_id == protocol_id and p.status == status
                                                    for p in answer.protocol_positions)
        if case.get("expect_specialist"):
            brief_ids = [g.brief_id for g in answer.governance if g.brief_id]
            with session_factory() as session:
                reviewers = {a.reviewer_id for a in session.query(ReviewAssignment)
                             .filter(ReviewAssignment.change_brief_id.in_(brief_ids))}
            checks["specialist"] = case["expect_specialist"] in reviewers
        if case.get("expect_duplicate_linked"):
            checks["duplicate_linked"] = any(f.novelty == "duplicate_of_existing" and f.duplicate_of for f in answer.findings)
        results.append(CaseResult(case["id"], case["expected_verdict"], answer.verdict, all(checks.values()), checks))

    per_class: Dict[str, Counter] = defaultdict(Counter)
    for r in results:
        per_class[r.expected or "?"]["total"] += 1
        per_class[r.expected or "?"]["correct"] += int(r.checks.get("verdict", r.passed))

    surveillance = _surveillance_metrics(session_factory, impact_agent)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases": len(results),
        "cases_passed": sum(r.passed for r in results),
        "verdict_accuracy_by_class": {k: {"correct": v["correct"], "total": v["total"],
                                          "accuracy": round(v["correct"] / v["total"], 3)} for k, v in per_class.items()},
        "citations_total": citations_total,
        "citations_verified_rate": round(citations_verified / citations_total, 3) if citations_total else 1.0,
        "ungrounded_answers": ungrounded,
        "latest_version_identified_rate": round(latest_ok / latest_total, 3) if latest_total else 1.0,
        "surveillance": surveillance,
        "results": [r.__dict__ for r in results],
    }
    return report


def _surveillance_metrics(session_factory, impact_agent) -> Dict[str, Any]:
    from app.models.entities import GuidanceChange

    feed = impact_agent.ranked_feed(include_filtered=True)
    by_source: Dict[str, List[str]] = defaultdict(list)
    for c in feed:
        by_source[c.watchlist_id].append(c.change_category)
    category = {sid: sorted(by_source.get(sid, [])) == sorted(exp) for sid, exp in EXPECTED_CHANGE_CATEGORIES.items()}

    # Precision: everything filtered is genuinely not practice-changing (no treatment content).
    # Recall: the editorial (no practice change by construction) is entirely filtered.
    filtered = [c for c in feed if c.relevance_status == "filtered_not_practice_changing"]
    correctly_filtered = [c for c in filtered if not c.treatments]
    should_filter = [c for c in feed if c.watchlist_id == "pub-ckd-editorial"]
    precision = len(correctly_filtered) / len(filtered) if filtered else 1.0
    recall = (len([c for c in should_filter if c.relevance_status == "filtered_not_practice_changing"]) / len(should_filter)
              if should_filter else 1.0)

    active = [c for c in feed if c.relevance_status != "filtered_not_practice_changing"]
    position = {c.id: i for i, c in enumerate(active)}
    case_series = [c for c in active if c.watchlist_id == "pub-metformin-case-series"]
    ordering_ok = bool(case_series) and all(
        position[c.id] < position[case_series[0].id]
        for c in active if c.change_category in ("contraindication_added", "safety_warning", "threshold_change")
    )
    duplicates = [c for c in active if c.watchlist_id == "bulletin-antimicrobial"]
    with session_factory() as session:
        dup_ok = bool(duplicates) and all(
            c.duplicate_of_change_id and session.get(GuidanceChange, c.duplicate_of_change_id).watchlist_id == "id-cap-guideline"
            for c in duplicates
        )
    return {
        "change_category_accuracy": round(sum(category.values()) / len(category), 3),
        "change_category_by_source": category,
        "filter_precision": round(precision, 3),
        "filter_recall": round(recall, 3),
        "filtered_items_retained": len(filtered),
        "duplicate_linked_to_higher_quality_source": dup_ok,
        "safety_and_threshold_ranked_above_case_series": ordering_ok,
    }


def write_report(report: Dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "clinician_query_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    lines = ["# Clinician Treatment Check evaluation (SYNTHETIC corpus)", "",
             f"Generated: {report['generated_at']}", "",
             f"- Cases passed: {report['cases_passed']} / {report['cases']}",
             f"- Citations verified: {report['citations_verified_rate']:.0%} of {report['citations_total']}",
             f"- Answers without a verified citation: {report['ungrounded_answers']}",
             f"- Latest version identified: {report['latest_version_identified_rate']:.0%}", "",
             "## Verdict accuracy by class", "", "| Verdict | Correct | Total |", "|---|---|---|"]
    lines += [f"| {k} | {v['correct']} | {v['total']} |" for k, v in report["verdict_accuracy_by_class"].items()]
    s = report["surveillance"]
    lines += ["", "## Surveillance", "",
              f"- Change category accuracy: {s['change_category_accuracy']:.0%}",
              f"- Relevance filter precision / recall: {s['filter_precision']:.0%} / {s['filter_recall']:.0%}",
              f"- Filtered items retained: {s['filtered_items_retained']}",
              f"- Duplicate linked to higher-quality source: {s['duplicate_linked_to_higher_quality_source']}",
              f"- Safety/threshold changes ranked above case series: {s['safety_and_threshold_ranked_above_case_series']}",
              "", "## Failed cases", ""]
    failed = [r for r in report["results"] if not r["passed"]]
    lines += [f"- {r['case_id']}: expected {r['expected']}, got {r['actual']}; checks {r['checks']}" for r in failed] or ["None."]
    path = out_dir / "clinician_query_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
