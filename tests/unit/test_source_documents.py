"""Tests for unified source normalization, identity, section loading and source diffing."""

from pathlib import Path

import pytest

from app.services.evaluation_corpus import make_multipage_pdf_bytes
from app.services.pdf_parser import DocumentSection, PageText, detect_document_sections, split_oversized_sections
from app.services.source_documents import (
    NormalizedSource,
    UnsupportedSourceFormatError,
    canonical_text,
    canonicalize_url,
    content_sha256,
    detect_markdown_sections,
    load_source_sections,
    load_source_text,
    url_source_identity,
)
from app.services.source_verifier import verify_recommendation_provenance
from app.services.source_version_diff import SourceVersionDiffService


# ------------------------------------------------------------------------------
# URL canonicalization & identity
# ------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "a, b",
    [
        ("https://Example.org/guide", "https://example.org/guide"),
        ("https://example.org/guide/", "https://example.org/guide"),
        ("https://example.org:443/guide", "https://example.org/guide"),
        ("http://example.org:80/guide", "http://example.org/guide"),
        ("https://example.org/guide#section-2", "https://example.org/guide"),
        ("https://example.org/guide?utm_source=mail&utm_medium=x", "https://example.org/guide"),
        ("https://example.org/guide?b=2&a=1", "https://example.org/guide?a=1&b=2"),
        ("https://example.org//a//b", "https://example.org/a/b"),
        ("https://example.org/guide?fbclid=abc&id=7", "https://example.org/guide?id=7"),
        ("HTTPS://EXAMPLE.ORG", "https://example.org/"),
    ],
)
def test_equivalent_urls_share_identity(a, b):
    assert canonicalize_url(a) == canonicalize_url(b)
    assert url_source_identity(a) == url_source_identity(b)


@pytest.mark.parametrize(
    "a, b",
    [
        ("https://example.org/guide", "https://example.org/guide2"),
        ("https://example.org/guide?id=1", "https://example.org/guide?id=2"),
        ("http://example.org/guide", "https://example.org/guide"),
        ("https://example.org:8443/guide", "https://example.org/guide"),
        ("https://a.example.org/guide", "https://b.example.org/guide"),
    ],
)
def test_distinct_urls_have_distinct_identity(a, b):
    assert url_source_identity(a) != url_source_identity(b)


def test_canonicalize_is_idempotent():
    for url in ("https://Example.org:443//x/?utm_campaign=q&z=1&a=2#f", "http://[::1]:8080/a/"):
        once = canonicalize_url(url)
        assert canonicalize_url(once) == once


def test_url_with_credentials_rejected():
    with pytest.raises(ValueError):
        canonicalize_url("https://user:pass@example.org/guide")


# ------------------------------------------------------------------------------
# Canonical text & hashing
# ------------------------------------------------------------------------------

def test_canonical_text_preserves_clinical_symbols():
    text = "HbA1c ≤ 7.0 %; eGFR ≥ 30; 500 µg ± 10 % – titrate"
    assert canonical_text(text) == text


def test_whitespace_and_tracking_noise_do_not_change_hash():
    a = "# Title\r\n\r\n\r\nAdults should take metformin.  \nSee https://x.org/p?utm_source=a&id=3\n![logo](https://x.org/l.png)"
    b = "# Title\n\nAdults should take metformin.\nSee https://x.org/p?id=3\n"
    assert content_sha256(a) == content_sha256(b)


def test_meaningful_change_alters_hash():
    assert content_sha256("metformin 500 mg") != content_sha256("metformin 1000 mg")
    assert content_sha256("eGFR ≥ 30") != content_sha256("eGFR ≤ 30")


# ------------------------------------------------------------------------------
# Markdown sections & provenance offsets
# ------------------------------------------------------------------------------

MARKDOWN = (
    "Intro paragraph without heading.\n\n"
    "# Recommendations\n\n"
    "Adults with T2D should start metformin 1000 mg when eGFR ≥ 45.\n\n"
    "Setext Heading\n"
    "--------------\n"
    "Monitor renal function every 12 months.\n"
)


def test_markdown_sections_with_offsets():
    sections = detect_markdown_sections(MARKDOWN)
    headings = [s.section_heading for s in sections]
    assert headings == ["General", "Recommendations", "Setext Heading"]
    for sec in sections:
        assert sec.page_number == 1
        assert MARKDOWN[sec.char_start:sec.char_end].strip() == sec.text


def test_markdown_loaded_sections_pass_provenance_verification(tmp_path: Path):
    path = tmp_path / "source.md"
    path.write_text(MARKDOWN, encoding="utf-8")
    rec = next(s for s in load_source_sections(path) if s.section_heading == "Recommendations")
    verbatim = "Adults with T2D should start metformin 1000 mg when eGFR ≥ 45."
    result = verify_recommendation_provenance(verbatim, verbatim, rec.text, page=1, expected_page=rec.page_number)
    assert result.is_verified


def test_unsupported_artifact_rejected(tmp_path: Path):
    path = tmp_path / "source.docx"
    path.write_bytes(b"x")
    with pytest.raises(UnsupportedSourceFormatError):
        load_source_sections(path)


def test_pdf_and_markdown_with_equivalent_content_yield_same_section_text(tmp_path: Path):
    pdf = tmp_path / "s.pdf"
    pdf.write_bytes(make_multipage_pdf_bytes([["1. Recommendations", "Adults should take metformin daily."]]))
    md = tmp_path / "s.md"
    md.write_text("## 1. Recommendations\n\nAdults should take metformin daily.\n", encoding="utf-8")

    pdf_secs = load_source_sections(pdf)
    md_secs = load_source_sections(md)
    assert [(s.section_heading.lstrip("# "), s.text) for s in pdf_secs] == [
        (s.section_heading, s.text) for s in md_secs
    ]
    assert content_sha256(load_source_text(md)) != ""


# ------------------------------------------------------------------------------
# PDF section detection
# ------------------------------------------------------------------------------

def test_pdf_heading_carries_across_page_break():
    pages = [
        PageText(1, "SECTION 4: RECOMMENDATIONS\nAdults should receive an SGLT2 inhibitor."),
        PageText(2, "and continue therapy unless eGFR falls below 20."),
    ]
    sections = detect_document_sections(pages)
    assert [s.section_heading for s in sections] == ["SECTION 4: RECOMMENDATIONS"] * 2
    assert sections[1].page_number == 2


def test_font_size_heading_detected():
    pages = [PageText(1, "Key Practice Points for adults\nAdults should be screened annually.",
                      heading_lines=frozenset({"Key Practice Points for adults"}))]
    sections = detect_document_sections(pages)
    assert sections[0].section_heading == "Key Practice Points for adults"


def test_oversized_section_split_without_losing_text():
    paragraphs = [f"Paragraph {i}: adults should be monitored. " * 20 for i in range(12)]
    sec = DocumentSection(page_number=3, section_heading="Monitoring", text="\n\n".join(paragraphs))
    parts = split_oversized_sections([sec], max_chars=2500)

    assert len(parts) > 1
    assert all(len(p.text) <= 2500 for p in parts)
    assert all(p.page_number == 3 for p in parts)
    assert parts[0].section_heading == "Monitoring (part 1)"
    assert "".join(p.text.replace("\n", "") for p in parts) == sec.text.replace("\n", "")


# ------------------------------------------------------------------------------
# Source evolution diff
# ------------------------------------------------------------------------------

def _sec(heading, text):
    return DocumentSection(page_number=1, section_heading=heading, text=text)


def test_renamed_heading_is_modified_not_removed_plus_added():
    old = [_sec("3. Pharmacotherapy", "Start metformin 500 mg daily.")]
    new = [_sec("3.1 Pharmacotherapy", "Start metformin 1000 mg daily.")]
    changes = SourceVersionDiffService().diff_sections(old, new)
    assert [(c.change_type, c.heading, c.old_heading) for c in changes] == [
        ("modified", "3.1 Pharmacotherapy", "3. Pharmacotherapy")
    ]


def test_renamed_heading_with_identical_text_is_no_change():
    text = "Monitor eGFR every 12 months in all adults receiving metformin therapy."
    changes = SourceVersionDiffService().diff_sections([_sec("Renal Monitoring", text)], [_sec("Kidney Monitoring", text)])
    assert changes == []


def test_added_and_removed_sections_reported():
    old = [_sec("Dosing", "Metformin 500 mg."), _sec("Legacy Advice", "Use sulfonylureas first-line in all adults.")]
    new = [_sec("Dosing", "Metformin 500 mg."), _sec("Cardiorenal Protection", "Add an SGLT2 inhibitor when CKD present.")]
    kinds = {(c.change_type, c.heading) for c in SourceVersionDiffService().diff_sections(old, new)}
    assert kinds == {("added", "Cardiorenal Protection"), ("removed", "Legacy Advice")}


def test_normalized_source_rejects_bad_hash():
    from datetime import datetime, timezone

    with pytest.raises(ValueError):
        NormalizedSource(
            source_identity="url_x", canonical_source_ref="https://x.org/", input_type="web_url",
            retrieval_provider="Jina Reader", routing_decision="jina", artifact_path="a.md",
            artifact_format="markdown", normalized_content_sha256="short", original_bytes_sha256="a" * 64,
            normalized_text_chars=1, retrieved_at=datetime.now(timezone.utc),
        )


def test_single_overlong_line_is_split_by_sentence_without_losing_text():
    line = " ".join(f"Sentence {i} says adults should be monitored closely." for i in range(400))
    sec = DocumentSection(page_number=1, section_heading="Table", text=line)
    parts = split_oversized_sections([sec], max_chars=1000)

    assert all(len(p.text) <= 1000 for p in parts)
    assert " ".join(" ".join(p.text.split()) for p in parts) == " ".join(line.split())
