"""Local deterministic PDF text extraction, section detection, and candidate filtering."""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
from typing import List, Optional, Sequence, Union
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


@dataclass
class DocumentSection:
    """Represents a discrete section identified within a source document."""
    page_number: int
    section_heading: str
    text: str
    candidate: bool = False


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

        for idx, page in enumerate(pdf.pages, start=1):
            extracted = page.extract_text() or ""
            pages_out.append(PageText(page_number=idx, text=extracted))

    return pages_out


_HEADING_REGEX = re.compile(
    r"^(?:(?:Section|Chapter|Part)\s+\d+[:.]?|\d+(?:\.\d+)*\.?)\s+[A-Z].*$",
    re.MULTILINE,
)


def _is_heading_line(line: str) -> bool:
    """Deterministic check to identify heading lines in clinical documents."""
    stripped = line.strip()
    if not stripped:
        return False

    # Heading length constraint
    if len(stripped) > 80:
        return False

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

    for pt in page_texts:
        lines = pt.text.splitlines()
        current_heading = "General"
        current_lines: List[str] = []

        for line in lines:
            if _is_heading_line(line):
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
            )
            candidate_sections.append(candidate_sec)

    return candidate_sections
