"""Phase 2: statement parsing, watchlist monitoring, statement index, change detection, query input."""

from pathlib import Path

import pytest

from app.agents.extraction_agent import ExtractionAgent
from app.agents.monitoring_agent import MonitoringAgent
from app.models.database import get_engine, get_session_factory, init_db
from app.models.entities import GuidanceChange, GuidanceStatement, IngestedDocument, IngestionFailure, ImmutableEntityError
from app.schemas.clinician_query import (
    ClinicianQueryInput,
    PatientContext,
    PatientIdentifierError,
    build_clinician_query,
    find_patient_identifiers,
)
from app.services.clinical_statements import parse_statement, split_statements
from app.services.config_service import AppConfig
from app.services.taxonomy import TaxonomyError, load_taxonomy
from app.services.watchlist import WatchlistEntry, load_watchlist

TAXONOMY = load_taxonomy()
WATCHLIST = load_watchlist(taxonomy=TAXONOMY)


@pytest.fixture
def env(tmp_path: Path):
    engine = get_engine(db_url=f"sqlite:///{tmp_path / 'g.db'}")
    init_db(engine=engine)
    sf = get_session_factory(engine=engine)
    cfg = AppConfig(database_url=f"sqlite:///{tmp_path / 'g.db'}", source_dir=tmp_path / "sources", groq_api_key="k")
    (tmp_path / "sources").mkdir()
    agent = MonitoringAgent(source_dir=cfg.source_dir, session_factory=sf, config=cfg)
    yield {"sf": sf, "cfg": cfg, "monitoring": agent, "extraction": ExtractionAgent(session_factory=sf, config=cfg)}
    engine.dispose()


def _ingest_and_index(env, entry_id):
    result = env["monitoring"].check_watchlist_entry(WATCHLIST.get(entry_id), WATCHLIST)
    assert result.error is None, result.error
    env["extraction"].index_guidance_statements([v.document_id for v in result.versions])
    return result


# ------------------------------------------------------------------------------ parsing
@pytest.mark.parametrize("text, expect", [
    ("Start metformin 500 mg twice daily with meals. (Evidence level A)",
     dict(doses=[(500.0, "mg")], frequency="twice_daily", type="recommendation", evidence="A")),
    ("Give amoxicillin 500 mg three times daily for 5 days.",
     dict(doses=[(500.0, "mg")], frequency="three_times_daily", duration=(5.0, "day"))),
    ("Repeat intramuscular adrenaline after 5 minutes if there is no improvement.",
     dict(interval=(5.0, "minute"), route="intramuscular", type="recommendation")),
    ("Metformin may be continued when eGFR is 45 mL/min/1.73m2 or above.",
     dict(doses=[], thresholds=[("egfr", 45.0)], type="recommendation")),
    ("Dabigatran is contraindicated in patients with eGFR below 30 mL/min/1.73m2.",
     dict(type="contraindication", thresholds=[("egfr", 30.0)])),
    ("The previous recommendation to add gliclazide as routine second-line therapy has been withdrawn.",
     dict(type="withdrawal")),
    ("metformin 500mg OD orally", dict(doses=[(500.0, "mg")], frequency="once_daily", route="oral")),
])
def test_parse_statement(text, expect):
    p = parse_statement(text, TAXONOMY)
    if "doses" in expect:
        assert [(d.value, d.unit) for d in p.doses] == expect["doses"]
    for key, attr in (("frequency", "frequency"), ("interval", "interval"), ("duration", "duration"), ("route", "route")):
        if key in expect:
            assert getattr(p, attr) == expect[key]
    if "thresholds" in expect:
        assert [(t.measure, t.value) for t in p.thresholds] == expect["thresholds"]
    if "type" in expect:
        assert p.statement_type == expect["type"]
    if "evidence" in expect:
        assert p.evidence_level == expect["evidence"]


def test_concentration_units_are_not_doses():
    p = parse_statement("Start ramipril 2.5 mg once daily when the albumin-to-creatinine ratio is 3 mg/mmol or above.", TAXONOMY)
    assert [(d.value, d.unit) for d in p.doses] == [(2.5, "mg")]
    assert [(t.measure, t.value) for t in p.thresholds] == [("acr", 3.0)]


def test_split_statements_offsets_are_exact():
    text = "# Title\n\n## 2. Therapy\nStart metformin 500 mg. Review in 2 weeks.\n> banner\n"
    spans = split_statements(text)
    assert [s.text for s in spans] == ["Start metformin 500 mg.", "Review in 2 weeks."]
    assert all(text[s.char_start:s.char_end] == s.text for s in spans)


# ------------------------------------------------------------------------------ monitoring
def test_watchlist_check_ingests_versions_in_order_and_is_idempotent(env):
    entry = WATCHLIST.get("endo-t2d-guideline")
    first = env["monitoring"].check_watchlist_entry(entry, WATCHLIST)
    assert first.error is None
    assert len(first.new_document_ids) == 2
    assert [v.publisher_version for v in first.versions] == ["1.0", "2.0"]
    assert [v.published_date for v in first.versions] == ["2025-01-15", "2026-02-01"]

    second = env["monitoring"].check_watchlist_entry(entry, WATCHLIST)
    assert second.new_document_ids == [] and second.error is None

    with env["sf"]() as session:
        v1, v2 = (session.get(IngestedDocument, v.document_id) for v in first.versions)
        assert v2.previous_source_version_id == v1.id
        assert v2.source_identifier == "wl:endo-t2d-guideline"
        assert v2.doc_metadata["quality_tier"] == 5 and v2.doc_metadata["departments"] == ["endocrinology"]


def test_new_published_version_is_detected(env, tmp_path):
    entry = WATCHLIST.get("sn-dabigatran-renal")
    env["monitoring"].check_watchlist_entry(entry, WATCHLIST)
    # Simulate the publisher releasing a new version in a copied corpus.
    import shutil
    corpus = tmp_path / "corpus"
    shutil.copytree("data/corpus", corpus)
    v1 = corpus / "safety-notices/dabigatran-renal-notice/v1.0.md"
    v2 = v1.with_name("v2.0.md")
    v2.write_text(v1.read_text(encoding="utf-8").replace('version: "1.0"', 'version: "2.0"').replace("below 30", "below 35"), encoding="utf-8")
    moved = load_watchlist(taxonomy=TAXONOMY, corpus_root=corpus)
    result = env["monitoring"].check_watchlist_entry(moved.get("sn-dabigatran-renal"), moved)
    assert len(result.new_document_ids) == 1
    assert result.latest.publisher_version == "2.0"


def test_url_entry_without_service_fails_visibly(env):
    entry = WatchlistEntry("web-x", "X", "https://example.org/x", "guideline", "P", 3, "", ("cardiology",), ())
    result = env["monitoring"].check_watchlist_entry(entry, WATCHLIST)
    assert result.error and "URL" in result.error
    with env["sf"]() as session:
        assert session.query(IngestionFailure).filter(IngestionFailure.source_path.like("watchlist:web-x%")).count() == 1


# ------------------------------------------------------------------------------ statement index
def test_statements_are_verbatim_substrings_of_the_artifact(env):
    result = _ingest_and_index(env, "endo-t2d-guideline")
    with env["sf"]() as session:
        for version in result.versions:
            doc = session.get(IngestedDocument, version.document_id)
            text = Path(doc.source_path).read_text(encoding="utf-8")
            stmts = session.query(GuidanceStatement).filter_by(ingested_document_id=doc.id).all()
            assert stmts
            for s in stmts:
                assert text[s.char_start:s.char_end] == s.verbatim_text
                assert s.publisher_version == version.publisher_version


def test_change_detection_categories_for_guideline_update(env):
    result = _ingest_and_index(env, "endo-t2d-guideline")
    v2_id = result.versions[-1].document_id
    with env["sf"]() as session:
        changes = session.query(GuidanceChange).filter_by(to_document_id=v2_id).all()
        cats = sorted(c.change_category for c in changes)
        assert cats == ["contraindication_added", "contraindication_added", "dose_change",
                        "new_recommendation", "threshold_change", "withdrawn"]
        withdrawn = next(c for c in changes if c.change_category == "withdrawn")
        old = session.get(GuidanceStatement, withdrawn.from_statement_id)
        new = session.get(GuidanceStatement, withdrawn.to_statement_id)
        assert old.verbatim_text.startswith("Gliclazide 40 mg") and old.publisher_version == "1.0"
        assert "withdrawn" in new.verbatim_text and new.publisher_version == "2.0"
        dose = next(c for c in changes if c.change_category == "dose_change")
        assert dose.attribute_changes == [{"kind": "frequency", "before": "once_daily", "after": "twice_daily"}]
        assert dose.treatments == ["metformin"] and dose.departments == ["endocrinology"]
        assert "t2d-glycaemic" in dose.pathways


@pytest.mark.parametrize("entry_id, expected", [
    ("cardio-af-guideline", {"threshold_change", "contraindication_added", "withdrawn"}),
    ("neph-ckd-guideline", {"threshold_change", "contraindication_added", "withdrawn"}),
    ("id-cap-guideline", {"dose_change", "contraindication_added", "withdrawn"}),
    ("em-anaphylaxis-guideline", {"dose_change", "contraindication_added", "withdrawn"}),
])
def test_every_department_update_is_categorised(env, entry_id, expected):
    result = _ingest_and_index(env, entry_id)
    with env["sf"]() as session:
        cats = {c.change_category for c in session.query(GuidanceChange).filter_by(to_document_id=result.versions[-1].document_id)}
    assert cats == expected


def test_indexing_is_idempotent(env):
    result = _ingest_and_index(env, "cardio-af-guideline")
    again = env["extraction"].index_guidance_statements([result.versions[-1].document_id])
    assert again[0].skipped and again[0].changes_created == 0


def test_statements_are_immutable(env):
    _ingest_and_index(env, "sn-fluoroquinolone")
    with env["sf"]() as session:
        stmt = session.query(GuidanceStatement).first()
        stmt.verbatim_text = "tampered"
        with pytest.raises(ImmutableEntityError):
            session.commit()


# ------------------------------------------------------------------------------ query input
@pytest.mark.parametrize("text, kind", [
    ("metformin for jane.doe@example.org", "email address"),
    ("MRN 12345678 metformin", "medical record / patient number"),
    ("DOB 01/02/1960", "date of birth"),
    ("call 0207 946 0000", "phone or national health number"),
    ("Mr Smith metformin", "personal name"),
    ("lives at 12 Baker Street", "street address"),
])
def test_identifier_guard_detects(text, kind):
    assert kind in find_patient_identifiers(text)


@pytest.mark.parametrize("text", [
    "metformin 500 mg twice daily", "MS flare", "patient is pregnant", "eGFR 30 to 44", "CHA2DS2-VASc 3",
    "apixaban 2.5 mg BD", "low-severity community-acquired pneumonia",
])
def test_identifier_guard_allows_clinical_text(text):
    assert find_patient_identifiers(text) == []


def test_build_query_normalises_and_guards():
    q = build_clinician_query(ClinicianQueryInput(
        department="Diabetes", treatment="Metformin 500mg OD",
        context=PatientContext(egfr_band="30-44", current_medications=["Zocor 20 mg"]),
    ), TAXONOMY)
    assert q.department == "endocrinology"
    assert q.treatment_ids == ["metformin"]
    assert q.plan.parsed.frequency == "once_daily"
    assert q.medication_treatments == ["simvastatin"]

    with pytest.raises(PatientIdentifierError) as exc:
        build_clinician_query(ClinicianQueryInput(department="cardiology", treatment="apixaban for Mrs Jones"), TAXONOMY)
    assert "Jones" not in str(exc.value)

    with pytest.raises(TaxonomyError):
        build_clinician_query(ClinicianQueryInput(department="astrology", treatment="apixaban"), TAXONOMY)


def test_context_rejects_unknown_fields():
    with pytest.raises(ValueError):
        PatientContext(name="Jane")
