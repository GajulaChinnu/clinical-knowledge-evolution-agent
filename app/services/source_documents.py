"""Unified source normalization for every CKEA input type.

All three entry points (PDF upload, PDF URL, web page URL) converge on the helpers here:
- canonical URL and stable source identity (never derived from filenames or timestamps)
- canonical normalized text and its SHA-256 (drives versioning / no-op detection)
- one section loader for stored artifacts (.pdf via pdfplumber, .md/.txt as UTF-8 text)

Downstream agents call `load_source_sections` and never branch on the input type.
"""

from datetime import datetime
import hashlib
from pathlib import Path
import re
from typing import List, Literal, Optional, Union
import unicodedata
import urllib.parse

from pydantic import BaseModel, ConfigDict, Field

from app.services.pdf_parser import (
    DocumentSection,
    detect_document_sections,
    extract_page_texts,
    split_oversized_sections,
)

TEXT_SOURCE_SUFFIXES = (".md", ".txt")
# Fewer alphanumeric characters than this across the whole PDF => no text layer (needs OCR).
MIN_READABLE_CHARS = 10


class UnreadableSourceError(ValueError):
    """Raised when a source has no extractable text (e.g. scanned/image-only PDF needing OCR)."""


class UnsupportedSourceFormatError(ValueError):
    """Raised when a stored source artifact has an unsupported file format."""


InputType = Literal["pdf_upload", "pdf_url", "web_url", "local_scan"]


class NormalizedSource(BaseModel):
    """Input-type-independent description of a normalized clinical source."""

    model_config = ConfigDict(extra="forbid")

    source_identity: str = Field(..., min_length=1)
    canonical_source_ref: str = Field(..., min_length=1)
    input_type: InputType
    retrieval_provider: str
    routing_decision: str
    artifact_path: str
    artifact_format: Literal["pdf", "markdown", "text"]
    normalized_content_sha256: str = Field(..., min_length=64, max_length=64)
    original_bytes_sha256: str = Field(..., min_length=64, max_length=64)
    normalized_text_chars: int = Field(..., ge=0)
    title: Optional[str] = None
    retrieved_at: datetime
    warnings: List[str] = Field(default_factory=list)

    def to_metadata(self) -> dict:
        """Fields persisted into IngestedDocument.doc_metadata."""
        data = self.model_dump(mode="json")
        data.pop("artifact_path", None)
        data.pop("source_identity", None)
        return data


# ==============================================================================
# URL CANONICALIZATION & IDENTITY
# ==============================================================================

_TRACKING_PARAM_PREFIXES = ("utm_",)
_TRACKING_PARAMS = {"fbclid", "gclid", "mc_cid", "mc_eid", "_ga", "_gl", "ref_src"}
_DEFAULT_PORTS = {"http": 80, "https": 443}


def _is_tracking_param(name: str) -> bool:
    lowered = name.lower()
    return lowered in _TRACKING_PARAMS or lowered.startswith(_TRACKING_PARAM_PREFIXES)


def canonicalize_url(url: str) -> str:
    """Return a canonical form of an HTTP(S) URL for stable source identity.

    Lowercases scheme and host, removes default ports, fragments and tracking parameters,
    sorts the query, collapses duplicate slashes and strips a trailing slash (except root).
    """
    parsed = urllib.parse.urlsplit(url.strip())
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower().rstrip(".")
    port = parsed.port
    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are not accepted as clinical sources.")
    netloc = f"[{host}]" if ":" in host else host
    if port and _DEFAULT_PORTS.get(scheme) != port:
        netloc = f"{netloc}:{port}"

    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    path = urllib.parse.quote(urllib.parse.unquote(path), safe="/:@!$&'()*+,;=-._~%")
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/") or "/"

    query_pairs = [
        (k, v)
        for k, v in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        if not _is_tracking_param(k)
    ]
    query = urllib.parse.urlencode(sorted(query_pairs))
    return urllib.parse.urlunsplit((scheme, netloc, path, query, ""))


def url_source_identity(url: str) -> str:
    """Stable source identity for a URL: hash of its canonical form."""
    canonical = canonicalize_url(url)
    return f"url_{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]}"


def content_source_identity(normalized_content_sha256: str) -> str:
    """Identity for a newly uploaded PDF that is not attached to an existing source."""
    return f"pdf_{normalized_content_sha256[:16]}"


# ==============================================================================
# CANONICAL TEXT & HASHING
# ==============================================================================

_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_URL_IN_TEXT = re.compile(r"https?://[^\s)\]>\"']+")


def _strip_tracking_from_url(match: "re.Match[str]") -> str:
    raw = match.group(0)
    try:
        parts = urllib.parse.urlsplit(raw)
    except ValueError:
        return raw
    if not parts.query:
        return raw
    kept = [(k, v) for k, v in urllib.parse.parse_qsl(parts.query, keep_blank_values=True) if not _is_tracking_param(k)]
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, urllib.parse.urlencode(kept), parts.fragment))


def canonical_text(text: str) -> str:
    """Canonicalize normalized source text so only meaningful changes alter its hash.

    Unicode NFC (clinical symbols such as ≥ ≤ µ ± are preserved), consistent line endings,
    trimmed whitespace, collapsed blank lines, image embeds and tracking parameters removed.
    """
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MD_IMAGE.sub("", text)
    text = _URL_IN_TEXT.sub(_strip_tracking_from_url, text)
    lines = [re.sub(r"[ \t ]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def content_sha256(text: str) -> str:
    """SHA-256 of the canonical form of a source's normalized text."""
    return hashlib.sha256(canonical_text(text).encode("utf-8")).hexdigest()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ==============================================================================
# MARKDOWN SECTION DETECTION (web sources)
# ==============================================================================

_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SETEXT_UNDERLINE = re.compile(r"^(=+|-+)\s*$")


def detect_markdown_sections(text: str) -> List[DocumentSection]:
    """Split Markdown/plain text into sections by headings, with character-offset provenance.

    Web sources have no pages: every section is page 1 and carries char_start/char_end
    offsets into the stored artifact. Text without headings falls back to a single
    "General" section, which `split_oversized_sections` then windows by paragraph.
    """
    lines = text.split("\n")
    offsets: List[int] = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1

    sections: List[DocumentSection] = []
    heading = "General"
    body_start: Optional[int] = None
    body_lines: List[str] = []

    def flush(end_offset: int) -> None:
        body = "\n".join(body_lines).strip()
        if body and body_start is not None:
            sections.append(DocumentSection(
                page_number=1,
                section_heading=heading,
                text=body,
                char_start=body_start,
                char_end=end_offset,
            ))

    i = 0
    while i < len(lines):
        line = lines[i]
        atx = _ATX_HEADING.match(line.strip())
        is_setext = (
            i + 1 < len(lines)
            and line.strip()
            and _SETEXT_UNDERLINE.match(lines[i + 1].strip() or "x")
            and len(line.strip()) <= 120
        )
        if atx or is_setext:
            flush(offsets[i])
            heading = (atx.group(2) if atx else line).strip().strip("*_ ")[:200] or "General"
            body_lines = []
            i += 1 if atx else 2
            body_start = offsets[i] if i < len(lines) else pos
            continue
        if body_start is None:
            body_start = offsets[i]
        body_lines.append(line)
        i += 1
    flush(pos)
    return sections


# ==============================================================================
# ARTIFACT LOADING
# ==============================================================================

def artifact_format(path: Union[str, Path]) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".md":
        return "markdown"
    if suffix == ".txt":
        return "text"
    raise UnsupportedSourceFormatError(f"Unsupported source artifact format: '{suffix}'")


def load_source_text(path: Union[str, Path]) -> str:
    """Return the full normalized text of a stored source artifact.

    Raises:
        UnreadableSourceError: PDF with no extractable text (needs OCR) or empty text file.
    """
    path = Path(path)
    if artifact_format(path) == "pdf":
        text = "\n\n".join(p.text for p in extract_page_texts(path))
        if sum(ch.isalnum() for ch in text) < MIN_READABLE_CHARS:
            raise UnreadableSourceError(
                f"unreadable_needs_ocr: PDF '{path.name}' has no extractable text "
                "(scanned or image-only). OCR is not enabled; manual review required."
            )
        return text
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise UnreadableSourceError(f"Source text artifact '{path.name}' is empty.")
    return text


def load_source_sections(path: Union[str, Path]) -> List[DocumentSection]:
    """Load sections from any stored source artifact. Callers never branch on input type."""
    path = Path(path)
    if artifact_format(path) == "pdf":
        sections = detect_document_sections(extract_page_texts(path))
    else:
        sections = detect_markdown_sections(path.read_text(encoding="utf-8"))
    return split_oversized_sections(sections)
