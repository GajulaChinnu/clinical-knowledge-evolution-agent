"""Phase 1: taxonomy, watchlist, SYNTHETIC corpus and protocol repository."""

import json
from pathlib import Path

import pytest

from app.services.protocol_repository import load_protocols
from app.services.taxonomy import TaxonomyError, load_taxonomy
from app.services.watchlist import (
    WatchlistError,
    load_watchlist,
    split_front_matter,
)

TAXONOMY = load_taxonomy()
WATCHLIST = load_watchlist(taxonomy=TAXONOMY)
DEPARTMENTS = ["endocrinology", "cardiology", "nephrology", "infectious-diseases", "emergency-medicine"]


# ------------------------------------------------------------------------------ taxonomy
@pytest.mark.parametrize("value, expected", [
    ("Endocrinology", "endocrinology"), ("diabetes", "endocrinology"), ("Renal", "nephrology"),
    ("A&E", "emergency-medicine"), ("infectious disease", "infectious-diseases"), ("  CARDIOLOGY ", "cardiology"),
])
def test_department_resolution(value, expected):
    assert TAXONOMY.resolve_department(value) == expected


def test_unknown_department_rejected():
    with pytest.raises(TaxonomyError, match="Unknown department"):
        TAXONOMY.resolve_department("Astrology")


@pytest.mark.parametrize("text, treatments, classes", [
    ("Start metformin 500 mg", ["metformin"], []),
    ("Glucophage XR", ["metformin"], []),
    ("Give EPINEPHRINE 0.5 mg IM", ["adrenaline"], []),
    ("Offer sodium bicarbonate 500 mg", ["sodium-bicarbonate"], []),
    ("Fluoroquinolones such as levofloxacin and ciprofloxacin", ["ciprofloxacin", "levofloxacin"], ["fluoroquinolone"]),
    ("Consider an SGLT2 inhibitor", [], ["sglt2-inhibitor"]),
    ("metformingly unrelated wording", [], []),
])
def test_treatment_matching(text, treatments, classes):
    found = TAXONOMY.find_treatments(text)
    assert found.treatments == treatments
    assert found.classes == classes


def test_class_membership():
    assert TAXONOMY.members_of_class("fluoroquinolone") == ["ciprofloxacin", "levofloxacin"]
    assert TAXONOMY.class_of("apixaban") == "doac"


def test_malformed_taxonomy_rejected(tmp_path):
    bad = tmp_path / "t.yaml"
    bad.write_text("departments:\n  - id: a\n    name: A\ntreatments:\n  - {id: x, name: X, class: nope}\n", encoding="utf-8")
    with pytest.raises(TaxonomyError, match="unknown class"):
        load_taxonomy(bad)


# ------------------------------------------------------------------------------ watchlist
def test_watchlist_covers_every_department_and_source_type():
    entries = list(WATCHLIST)
    assert {d for e in entries for d in e.departments} >= set(DEPARTMENTS)
    assert {e.source_type for e in entries} == {"guideline", "safety_notice", "publication"}
    assert all(e.source_identity == f"wl:{e.id}" for e in entries)


def test_every_department_has_a_guideline_with_two_versions():
    for dept in DEPARTMENTS:
        guidelines = [e for e in WATCHLIST if e.source_type == "guideline" and dept in e.departments]
        assert guidelines, dept
        versions = WATCHLIST.corpus_versions(guidelines[0])
        assert [v.version for v in versions] == ["1.0", "2.0"]
        assert versions[0].published_date < versions[1].published_date


def test_every_corpus_document_is_labelled_synthetic():
    for entry in WATCHLIST:
        for version in WATCHLIST.corpus_versions(entry):
            assert version.front_matter["synthetic"] is True
            assert "SYNTHETIC" in version.body
            assert version.front_matter["source_id"] == entry.id


def test_safety_notices_are_selected_across_departments():
    selected = {e.id for e in WATCHLIST.select("endocrinology", ["dabigatran"], TAXONOMY)}
    assert "sn-dabigatran-renal" in selected
    # Publications outside the treatment are not pulled in.
    assert "pub-apixaban-rct" not in {e.id for e in WATCHLIST.select("cardiology", ["dabigatran"], TAXONOMY)}


def test_class_level_selection():
    selected = {e.id for e in WATCHLIST.select("infectious-diseases", ["levofloxacin"], TAXONOMY)}
    assert "sn-fluoroquinolone" in selected and "id-cap-guideline" in selected


@pytest.mark.parametrize("patch, match", [
    ({"source_type": "blog"}, "source_type"),
    ({"quality_tier": 9}, "quality_tier"),
    ({"departments": ["astrology"]}, "Unknown department"),
    ({"location": "ftp://x"}, "location"),
])
def test_invalid_watchlist_entries_rejected(tmp_path, patch, match):
    entry = {"id": "x", "title": "X", "location": "corpus:x", "source_type": "guideline",
             "publisher": "P", "quality_tier": 3, "departments": ["cardiology"], "treatments": []}
    entry.update(patch)
    import yaml
    path = tmp_path / "w.yaml"
    path.write_text(yaml.safe_dump({"sources": [entry]}), encoding="utf-8")
    with pytest.raises(WatchlistError, match=match):
        load_watchlist(path, taxonomy=TAXONOMY)


def test_corpus_location_cannot_escape_root(tmp_path):
    import yaml
    path = tmp_path / "w.yaml"
    path.write_text(yaml.safe_dump({"sources": [{
        "id": "x", "title": "X", "location": "corpus:../../etc", "source_type": "guideline",
        "publisher": "P", "quality_tier": 3, "departments": ["cardiology"]}]}), encoding="utf-8")
    watchlist = load_watchlist(path, taxonomy=TAXONOMY, corpus_root=tmp_path / "corpus")
    with pytest.raises(WatchlistError, match="escapes"):
        watchlist.corpus_dir(watchlist.get("x"))


def test_front_matter_parsing():
    meta, body = split_front_matter("---\nversion: '2.0'\nsynthetic: true\n---\n\n# Title\n")
    assert meta == {"version": "2.0", "synthetic": True}
    assert body == "# Title\n"


# ------------------------------------------------------------------------------ protocols
def test_protocol_repository_has_departments_for_every_protocol():
    protocols = load_protocols("data/protocols", TAXONOMY)
    assert len(protocols) >= 5
    assert {p.department for p in protocols} == set(DEPARTMENTS)
    legacy = next(p for p in protocols if p.protocol_id == "PROT-DM-001")
    assert legacy.department == "endocrinology" and legacy.treatments == ["metformin"]


def test_legacy_protocol_file_is_unchanged():
    data = json.loads(Path("data/protocols/PROT-DM-001_v1.0.json").read_text(encoding="utf-8"))
    assert "department" not in data
    assert data["sections"][1]["section_text"].startswith("Metformin 500mg once daily")
