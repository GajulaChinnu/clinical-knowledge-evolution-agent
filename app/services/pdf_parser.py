"""Local deterministic PDF text extraction, section detection, and candidate filtering."""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
import statistics
from typing import FrozenSet, List, Optional, Sequence, Union
import pdfplumber

logger = logging.getLogger("ckea.services.pdf_parser")

# Standard clinical recommendation indicator terms
RECOMMENDATION_KEYWORDS = (
    "recommend",
    "recommended",
    "should",
    "should not",
    "contraindicated",
    "contraindication",
    "indication",
    "indications",
    "threshold",
    "treatment",
    "monitoring",
    "target",
    "dose",
    "dosage",
    "therapy",
)


@dataclass
class PageText:
    """Represents text extracted from a single PDF page."""
    page_number: int
    text: str
    # Normalized lines typeset noticeably larger than body text (font-size heading signal).
    heading_lines: FrozenSet[str] = field(default_factory=frozenset)


@dataclass
class DocumentSection:
    """Represents a discrete section identified within a source document.

    Provenance: PDFs use real page numbers; text/Markdown sources use page 1 plus
    character offsets (char_start/char_end) into the stored artifact.
    """
    page_number: int
    section_heading: str
    text: str
    candidate: bool = False
    char_start: Optional[int] = None
    char_end: Optional[int] = None


# Sections longer than this are split into paragraph windows before extraction.
MAX_SECTION_CHARS = 6000
HEADING_FONT_RATIO = 1.2


def _normalize_line(line: str) -> str:
    return " ".join(line.split())


def _large_font_lines(page, body_size: Optional[float]) -> FrozenSet[str]:
    """Return lines on a page whose largest glyph is >= HEADING_FONT_RATIO x body size."""
    if not body_size:
        return frozenset()
    try:
        words = page.extract_words(extra_attrs=["size"])
    except Exception:  # pdfplumber raises assorted errors on malformed pages; the signal is optional
        logger.debug("Font-size heading detection skipped for a page", exc_info=True)
        return frozenset()
    lines: dict = {}
    for w in words:
        lines.setdefault(round(float(w["top"])), []).append(w)
    out = set()
    for ws in lines.values():
        if max(float(w["size"]) for w in ws) >= body_size * HEADING_FONT_RATIO:
            text = _normalize_line(" ".join(w["text"] for w in sorted(ws, key=lambda w: w["x0"])))
            if text and len(text) <= 80:
                out.add(text)
    return frozenset(out)


def extract_page_texts(file_path: Union[str, Path]) -> List[PageText]:
    """Extract raw text page by page from a local PDF using pdfplumber.

    Args:
        file_path: Path to the target PDF file.

    Returns:
        List of PageText objects containing 1-indexed page numbers and extracted text.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the PDF cannot be opened or contains no pages.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    pages_out: List[PageText] = []
    with pdfplumber.open(path) as pdf:
        if len(pdf.pages) == 0:
            raise ValueError(f"PDF file contains zero pages: {path}")

        sizes: List[float] = []
        for page in pdf.pages:
            sizes.extend(float(c["size"]) for c in page.chars if c.get("text", "").strip())
        body_size = statistics.median(sizes) if sizes else None

        for idx, page in enumerate(pdf.pages, start=1):
            extracted = page.extract_text() or ""
            pages_out.append(PageText(
                page_number=idx,
                text=extracted,
                heading_lines=_large_font_lines(page, body_size),
            ))

    return pages_out


_HEADING_REGEX = re.compile(
    r"^(?:(?:Section|Chapter|Part)\s+\d+[:.]?|\d+(?:\.\d+)*\.?)\s+[A-Z].*$",
    re.MULTILINE,
)


def _is_heading_line(line: str, font_headings: FrozenSet[str] = frozenset()) -> bool:
    """Deterministic check to identify heading lines in clinical documents."""
    stripped = line.strip()
    if not stripped:
        return False

    # Heading length constraint
    if len(stripped) > 80:
        return False

    # Typeset larger than body text
    if _normalize_line(stripped) in font_headings and not stripped.endswith((".", ";", ",")):
        return True

    # Numbered heading pattern: e.g. "1. Recommendations", "Section 2: Treatment"
    if _HEADING_REGEX.match(stripped):
        return True

    # Standalone uppercase or Title Case line without trailing punctuation
    if not stripped.endswith((".", ";", ":", ",")):
        if stripped.isupper() and len(stripped.split()) <= 8:
            return True
        if stripped.istitle() and len(stripped.split()) <= 8:
            return True

    return False


def detect_document_sections(page_texts: Sequence[PageText]) -> List[DocumentSection]:
    """Detect headings and group lines into discrete DocumentSections deterministically.

    Args:
        page_texts: Sequence of PageText instances.

    Returns:
        List of DocumentSection objects with page numbers and headings.
    """
    sections: List[DocumentSection] = []
    # The active heading carries across page breaks so a section continuing on the
    # next page keeps its heading instead of falling back to "General".
    current_heading = "General"

    for pt in page_texts:
        lines = pt.text.splitlines()
        current_lines: List[str] = []

        for line in lines:
            if _is_heading_line(line, pt.heading_lines):
                # Save previous section if it contained text
                if current_lines:
                    sec_text = "\n".join(current_lines).strip()
                    if sec_text:
                        sections.append(
                            DocumentSection(
                                page_number=pt.page_number,
                                section_heading=current_heading,
                                text=sec_text,
                            )
                        )
                    current_lines = []
                current_heading = line.strip()
            else:
                current_lines.append(line)

        # Append remaining lines of the page
        if current_lines:
            sec_text = "\n".join(current_lines).strip()
            if sec_text:
                sections.append(
                    DocumentSection(
                        page_number=pt.page_number,
                        section_heading=current_heading,
                        text=sec_text,
                    )
                )

    # If document had text but no sections were recorded, create a fallback section
    if not sections and page_texts:
        for pt in page_texts:
            clean = pt.text.strip()
            if clean:
                sections.append(
                    DocumentSection(
                        page_number=pt.page_number,
                        section_heading="General",
                        text=clean,
                    )
                )

    return sections


def split_oversized_sections(
    sections: Sequence[DocumentSection],
    max_chars: int = MAX_SECTION_CHARS,
) -> List[DocumentSection]:
    """Split sections longer than max_chars into paragraph windows ("Heading (part n)").

    No text is dropped: windows partition the section by paragraph (or by line when a
    single paragraph is itself too long). Character offsets are carried when known.
    """
    out: List[DocumentSection] = []
    for sec in sections:
        if len(sec.text) <= max_chars:
            out.append(sec)
            continue
        units = [u for u in re.split(r"\n\s*\n", sec.text) if u.strip()]
        expanded: List[str] = []
        for unit in units:
            if len(unit) <= max_chars:
                expanded.append(unit)
                continue
            for line in (ln for ln in unit.splitlines() if ln.strip()):
                if len(line) <= max_chars:
                    expanded.append(line)
                    continue
                for sentence in re.split(r"(?<=[.!?;])\s+", line):
                    while len(sentence) > max_chars:
                        cut = sentence.rfind(" ", 0, max_chars)
                        cut = cut if cut > 0 else max_chars
                        expanded.append(sentence[:cut])
                        sentence = sentence[cut:].lstrip()
                    if sentence:
                        expanded.append(sentence)
        windows: List[str] = []
        current: List[str] = []
        size = 0
        for unit in expanded:
            if current and size + len(unit) > max_chars:
                windows.append("\n\n".join(current))
                current, size = [], 0
            current.append(unit)
            size += len(unit) + 2
        if current:
            windows.append("\n\n".join(current))
        search_from = 0
        for n, window in enumerate(windows, start=1):
            start = end = None
            if sec.char_start is not None:
                rel = sec.text.find(window.split("\n", 1)[0], search_from)
                if rel >= 0:
                    search_from = rel
                    start = sec.char_start + rel
                    end = start + len(window)
            out.append(DocumentSection(
                page_number=sec.page_number,
                section_heading=f"{sec.section_heading} (part {n})",
                text=window,
                candidate=sec.candidate,
                char_start=start,
                char_end=end,
            ))
    return out


def filter_candidate_sections(
    sections: Sequence[DocumentSection],
    keywords: Sequence[str] = RECOMMENDATION_KEYWORDS,
) -> List[DocumentSection]:
    """Filter document sections down to candidates likely to contain recommendations.

    Args:
        sections: Sequence of DocumentSection items to inspect.
        keywords: Sequence of lower-case indicator terms to match.

    Returns:
        List of candidate DocumentSection items with candidate=True.
    """
    candidate_sections: List[DocumentSection] = []

    for sec in sections:
        content_lower = f"{sec.section_heading} {sec.text}".lower()
        if any(kw in content_lower for kw in keywords):
            candidate_sec = DocumentSection(
                page_number=sec.page_number,
                section_heading=sec.section_heading,
                text=sec.text,
                candidate=True,
                char_start=sec.char_start,
                char_end=sec.char_end,
            )
            candidate_sections.append(candidate_sec)

    return candidate_sections
