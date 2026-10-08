"""Phase 4: relevance filtering, evidence grading (source quality, novelty) and ranking."""

import pytest
import yaml

from app.agents.impact_agent import ImpactAgent
from app.models.entities import AuditLog, ChangeRecord, GapRecord, GuidanceChange, IngestedDocument
from app.services.ranking import RankingConfigError, load_ranking_rules, priority, source_quality
from app.services.reviewer_authorization import ReviewerAuthorizationService
from tests.corpus_env import build_corpus_env

RULES = load_ranking_rules()


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    env = build_corpus_env(tmp_path_factory.mktemp("rank"))
    env.impact = ImpactAgent(session_factory=env.session_factory, config=env.config)
    env.impact.assess_guidance_changes()
    yield env
    env.engine.dispose()


# ------------------------------------------------------------------------------ rules
def test_rules_carry_ids_and_bases():
    assert RULES.relevance_rule("REL-01").score == 5 and RULES.relevance_rule("REL-01").basis
    assert RULES.urgency_rule("URG-01").score == 5


@pytest.mark.parametrize("patch, match", [
    ({"priority_weights": {"urgency": 0.5, "relevance": 0.5, "source_quality": 0.5}}, "summing to 1.0"),
    ({"relevance": {"REL-01": {"score": 9}}}, "1-5"),
])
def test_invalid_rules_rejected(tmp_path, patch, match):
    data = yaml.safe_load(open("config/ranking.yaml", encoding="utf-8"))
    data.update(patch)
    path = tmp_path / "r.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(RankingConfigError, match=match):
        load_ranking_rules(path)


def test_evidence_level_caps_but_never_raises_quality():
    assert source_quality(RULES, 5, "D")[0] == 2
    assert source_quality(RULES, 1, "A")[0] == 1
    assert source_quality(RULES, None, None)[0] == RULES.default_tier


def test_duplicates_are_heavily_down_weighted():
    assert priority(RULES, 4, 5, 5, "duplicate_of_existing") < priority(RULES, 4, 5, 5, "new") / 2


# ------------------------------------------------------------------------------ background mode
def _changes(env, **filters):
    with env.session_factory() as session:
        rows = session.query(GuidanceChange).filter_by(**filters).all()
        for r in rows:
            session.expunge(r)
        return rows


def test_editorial_is_filtered_with_reason_and_kept(corpus):
    rows = _changes(corpus, watchlist_id="pub-ckd-editorial")
    assert rows
    assert all(r.relevance_status == "filtered_not_practice_changing" for r in rows)
    assert all(r.filter_reason.startswith("FLT-01") for r in rows)
    feed_ids = {c.id for c in corpus.impact.ranked_feed()}
    assert not feed_ids & {r.id for r in rows}
    assert {r.id for r in rows} <= {c.id for c in corpus.impact.ranked_feed(include_filtered=True)}


def test_feed_ranks_safety_and_threshold_changes_above_case_series(corpus):
    feed = corpus.impact.ranked_feed()
    position = {c.id: i for i, c in enumerate(feed)}
    case_series = next(c for c in feed if c.watchlist_id == "pub-metformin-case-series" and c.change_category == "new_evidence")
    notice_ci = next(c for c in feed if c.watchlist_id == "sn-fluoroquinolone" and c.change_category == "contraindication_added")
    threshold = next(c for c in feed if c.change_category == "threshold_change")
    assert position[notice_ci.id] < position[case_series.id]
    assert position[threshold.id] < position[case_series.id]
    assert case_series.source_quality == 1


def test_feed_contains_only_latest_versions(corpus):
    with corpus.session_factory() as session:
        superseded = {d.previous_source_version_id for d in session.query(IngestedDocument) if d.previous_source_version_id}
    assert all(c.to_document_id not in superseded for c in corpus.impact.ranked_feed())


def test_duplicate_is_linked_to_higher_quality_source(corpus):
    bulletin = next(c for c in _changes(corpus, watchlist_id="bulletin-antimicrobial") if "amoxicillin" in (c.treatments or []))
    assert bulletin.novelty == "duplicate_of_existing"
    with corpus.session_factory() as session:
        canonical = session.get(GuidanceChange, bulletin.duplicate_of_change_id)
        assert canonical.watchlist_id == "id-cap-guideline"


def test_every_ranked_change_records_rule_ids(corpus):
    for c in corpus.impact.ranked_feed(include_filtered=True):
        assert c.ranking_basis["relevance"]["rule_id"].startswith("REL-")
        assert c.ranking_basis["urgency"]["rule_id"].startswith("URG-")
        assert c.ranking_basis["rules_version"] == RULES.version


def test_department_filter(corpus):
    feed = corpus.impact.ranked_feed(department="cardiology")
    assert feed and all("cardiology" in c.departments for c in feed)


def test_restore_filtered_change_requires_authorised_reviewer_and_is_audited(corpus):
    auth = ReviewerAuthorizationService()
    filtered = _changes(corpus, watchlist_id="pub-ckd-editorial")[0]
    with pytest.raises(PermissionError):
        corpus.impact.restore_filtered_change(filtered.id, "dr_random", "looks relevant", auth)
    with pytest.raises(ValueError):
        corpus.impact.restore_filtered_change(filtered.id, "dr_smith", "  ", auth)
    restored = corpus.impact.restore_filtered_change(filtered.id, "dr_smith", "Relevant to our CKD awareness work.", auth)
    assert restored.relevance_status == "restored_by_reviewer"
    with corpus.session_factory() as session:
        audit = session.query(AuditLog).filter_by(entity_id=filtered.id).one()
        assert audit.actor == "dr_smith" and audit.new_status == "restored_by_reviewer"


# ------------------------------------------------------------------------------ query mode
def test_query_findings_ranked_with_contraindication_first(corpus):
    query, outcome = corpus.compare("Infectious Diseases", "levofloxacin 500 mg once daily",
                                    condition="low-severity community-acquired pneumonia")
    ranked = corpus.impact.score_findings(query.department, query.treatment_ids, outcome.findings)
    assert ranked[0].kind == "contraindication" and ranked[0].relation == "applies"
    assert all(f.priority_score is not None and f.ranking_basis["urgency"]["rule_id"] for f in ranked)


def test_query_duplicate_finding_linked(corpus):
    query, outcome = corpus.compare("Infectious Diseases", "amoxicillin 500 mg three times daily for 5 days")
    ranked = corpus.impact.score_findings(query.department, query.treatment_ids, outcome.findings)
    bulletin = next(f for f in ranked if f.citation.watchlist_id == "bulletin-antimicrobial")
    guideline = next(f for f in ranked if f.citation.watchlist_id == "id-cap-guideline" and f.citation.excerpt.startswith("Give amoxicillin"))
    assert bulletin.novelty == "duplicate_of_existing" and bulletin.duplicate_of == guideline.citation.statement_id
    assert guideline.priority_score > bulletin.priority_score


# ------------------------------------------------------------------------------ impact records
def test_impact_record_gets_ranking_without_changing_tier(corpus):
    with corpus.session_factory() as session:
        doc = session.query(IngestedDocument).filter_by(source_identifier="wl:endo-t2d-guideline").order_by(IngestedDocument.created_at.desc()).first()
        change = ChangeRecord(
            ingested_document_id=doc.id, verbatim_text="Start metformin 500 mg twice daily with meals in adults with newly diagnosed type 2 diabetes.",
            recommendation_type="treatment", target_population="adults with type 2 diabetes", intervention="metformin",
            evidence_grade="Grade A", confidence=0.95, page=1, section="2. First-line therapy",
            source_excerpt="Start metformin 500 mg twice daily with meals.", extraction_model_version="m",
            extraction_prompt_version="1", status="gap_confirmed", change_category="dose_change",
            departments=["endocrinology"], treatments=["metformin"],
        )
        session.add(change)
        session.flush()
        gap = GapRecord(change_record_id=change.id, comparison_result="gap", comparison_confidence=0.9,
                        difference_type="dosage_change", matched_protocol_id="PROT-DM-001", matched_protocol_version="v1.0",
                        is_match=True, status="matched", similarity=0.9)
        session.add(gap)
        session.commit()
        gap_id = gap.id
    impact = corpus.impact.process_gap_record(gap_id, breadth_input="one_specialty")
    assert impact.tier is not None
    assert impact.priority_score and impact.affected_departments == ["endocrinology"]
    assert "t2d-glycaemic" in impact.affected_pathways
    assert impact.ranking_basis["note"].startswith("Priority orders the governance queue only")
