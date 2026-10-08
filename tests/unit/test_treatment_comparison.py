"""Phase 3: three-way comparison (plan vs latest, previous vs latest, protocol vs latest)."""

from pathlib import Path

import pytest

from tests.corpus_env import build_corpus_env


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    env = build_corpus_env(tmp_path_factory.mktemp("corpus"))
    yield env
    env.engine.dispose()


def _by_kind(outcome, kind):
    return [f for f in outcome.findings if f.kind == kind]


def _governing(outcome):
    found = _by_kind(outcome, "governing_recommendation")
    assert len(found) == 1
    return found[0]


def test_plan_following_superseded_version(corpus):
    _, outcome = corpus.compare("Endocrinology", "metformin 500 mg once daily")
    gov = _governing(outcome)
    assert gov.relation == "differs_from_plan"
    assert gov.citation.excerpt.startswith("Start metformin 500 mg twice daily")
    assert gov.citation.version == "2.0" and gov.citation.published_date == "2026-02-01"
    assert gov.previous_citation.excerpt.startswith("Start metformin 500 mg once daily")
    assert gov.previous_citation.version == "1.0"
    dose = next(v for v in outcome.version_changes if v.change_category == "dose_change")
    assert dose.plan_matches_previous
    assert [(d.kind, d.before, d.after) for d in dose.differences] == [("frequency", "once_daily", "twice_daily")]


def test_plan_matching_latest_version(corpus):
    _, outcome = corpus.compare("Endocrinology", "metformin 500 mg twice daily", egfr_band=">=60")
    assert _governing(outcome).relation == "matches_plan"
    contraindications = _by_kind(outcome, "contraindication")
    assert contraindications and all(f.relation == "not_applicable" for f in contraindications)


def test_egfr_contraindication_applies(corpus):
    _, outcome = corpus.compare("Endocrinology", "metformin 500 mg twice daily", egfr_band="15-29")
    applies = [f for f in _by_kind(outcome, "contraindication") if f.relation == "applies"]
    assert any("below 30" in f.citation.excerpt for f in applies)


def test_extra_condition_needs_confirmation(corpus):
    _, outcome = corpus.compare("Endocrinology", "metformin 500 mg twice daily", egfr_band="30-44")
    contrast = next(f for f in _by_kind(outcome, "contraindication") if "contrast" in f.citation.excerpt)
    assert contrast.relation == "check_applicability"
    reduce = next(f for f in outcome.findings if f.citation.excerpt.startswith("Reduce metformin"))
    assert reduce.relation != "not_applicable"


def test_withdrawn_recommendation_matches_plan(corpus):
    _, outcome = corpus.compare("Endocrinology", "gliclazide 40 mg once daily")
    withdrawal = _by_kind(outcome, "withdrawal")[0]
    assert withdrawal.relation == "withdrawn_matches_plan"
    assert withdrawal.previous_citation.excerpt.startswith("Gliclazide 40 mg once daily")
    assert any(v.change_category == "withdrawn" and v.plan_matches_previous for v in outcome.version_changes)


def test_comorbidity_contraindication(corpus):
    _, outcome = corpus.compare("Cardiology", "apixaban 5 mg twice daily", comorbidities=["mechanical heart valve"])
    valve = next(f for f in _by_kind(outcome, "contraindication") if "mechanical heart valve" in f.citation.excerpt)
    assert valve.relation == "applies"
    _, outcome2 = corpus.compare("Cardiology", "apixaban 5 mg twice daily")
    valve2 = next(f for f in _by_kind(outcome2, "contraindication") if "mechanical heart valve" in f.citation.excerpt)
    assert valve2.relation == "check_applicability"


def test_safety_notice_applies_to_indication(corpus):
    _, outcome = corpus.compare("Infectious Diseases", "levofloxacin 500 mg once daily",
                                condition="low-severity community-acquired pneumonia")
    notice = next(f for f in _by_kind(outcome, "contraindication") if f.citation.source_type == "safety_notice")
    assert notice.relation == "applies"
    assert any(f.kind == "safety_warning" for f in outcome.findings)


@pytest.mark.parametrize("meds, expected", [(["simvastatin 40 mg"], "applies"), (["amoxicillin"], "not_applicable"), ([], "check_applicability")])
def test_drug_interaction_uses_medication_list(corpus, meds, expected):
    _, outcome = corpus.compare("Infectious Diseases", "clarithromycin 500 mg twice daily for 5 days", current_medications=meds)
    interaction = next(f for f in _by_kind(outcome, "contraindication") if "simvastatin" in f.citation.excerpt)
    assert interaction.relation == expected


def test_cross_department_safety_notice(corpus):
    _, outcome = corpus.compare("Nephrology", "dabigatran 110 mg twice daily", egfr_band="15-29")
    notice = next(f for f in _by_kind(outcome, "contraindication") if f.citation.source_type == "safety_notice")
    assert notice.relation == "applies"


def test_route_specific_restriction(corpus):
    _, outcome = corpus.compare("Emergency Medicine", "intramuscular adrenaline 0.5 mg")
    iv = next(f for f in _by_kind(outcome, "contraindication") if "Intravenous adrenaline" in f.citation.excerpt)
    assert iv.relation == "not_applicable"


@pytest.mark.parametrize("department, treatment, protocol, status", [
    ("Endocrinology", "metformin 500 mg once daily", "PROT-DM-001", "out_of_date"),
    ("Cardiology", "apixaban 5 mg twice daily", "PROT-CARD-001", "aligned"),
    ("Nephrology", "ramipril 2.5 mg once daily", "PROT-NEPH-001", "out_of_date"),
    ("Infectious Diseases", "amoxicillin 500 mg three times daily for 5 days", "PROT-ID-001", "aligned"),
    ("Emergency Medicine", "intramuscular adrenaline 0.5 mg", "PROT-EM-001", "out_of_date"),
])
def test_protocol_position(corpus, department, treatment, protocol, status):
    _, outcome = corpus.compare(department, treatment)
    position = next(p for p in outcome.protocol_positions if p.protocol_id == protocol)
    assert position.status == status, [i.reason for i in position.items]


def test_out_of_date_protocol_points_at_superseded_version(corpus):
    _, outcome = corpus.compare("Endocrinology", "metformin 500 mg once daily")
    position = next(p for p in outcome.protocol_positions if p.protocol_id == "PROT-DM-001")
    item = next(i for i in position.items if i.status == "out_of_date")
    assert item.previous_citation.version == "1.0" and item.latest_citation.version == "2.0"
    assert "Metformin 500mg once daily" in item.section_text


def test_missing_contraindication_is_a_protocol_gap(corpus):
    _, outcome = corpus.compare("Nephrology", "ramipril 2.5 mg once daily")
    position = next(p for p in outcome.protocol_positions if p.protocol_id == "PROT-NEPH-001")
    assert any(i.status == "missing_from_protocol" and "losartan" in i.latest_citation.excerpt for i in position.items)


def test_unknown_treatment_has_no_grounded_findings(corpus):
    _, outcome = corpus.compare("Cardiology", "unicorn extract 5 mg")
    assert outcome.findings == [] and outcome.retrieval == "none"


def test_every_citation_is_verbatim_and_verified(corpus):
    for dept, treatment in [("Endocrinology", "metformin 500 mg once daily"), ("Cardiology", "apixaban 5 mg twice daily"),
                            ("Infectious Diseases", "levofloxacin 500 mg once daily")]:
        _, outcome = corpus.compare(dept, treatment)
        cites = [f.citation for f in outcome.findings] + [f.previous_citation for f in outcome.findings if f.previous_citation]
        assert cites
        for c in cites:
            assert c.verified
            with corpus.session_factory() as session:
                from app.models.entities import IngestedDocument
                doc = session.get(IngestedDocument, c.document_id)
                assert Path(doc.source_path).read_text(encoding="utf-8")[c.char_start:c.char_end] == c.excerpt
