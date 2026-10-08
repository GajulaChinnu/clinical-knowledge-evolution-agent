"""CKEA URL Ingestion Service for Clinical Documents.

Supports:
1. HTTP and HTTPS clinical source URLs.
2. Direct PDF URLs (downloaded and saved directly under data/sources/).
3. HTML clinical web pages (normalized into local PDF representations under data/sources/).
4. Converges with the existing MonitoringAgent and ClinicalKnowledgePipeline.

Guarantees:
- Strict URL validation (HTTP/HTTPS only, malformed rejected).
- Safe filename generation and path traversal blocking.
- Output restricted exclusively to data/sources/.
- Complete provenance tracking (original URL, timestamp, SHA-256, filename).
- Zero direct LLM calls in the ingestion service.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import html as html_lib
import html.parser
import io
import logging
from pathlib import Path
import re
import socket
import textwrap
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from app.services.evaluation_corpus import make_multipage_pdf_bytes
from app.services.file_hash import compute_sha256
from app.services.jina_reader_service import JinaReaderService

logger = logging.getLogger("ckea.services.url_ingestion")


class URLIngestionError(Exception):
    """Base exception for all URL ingestion errors."""
    pass


class InvalidURLError(URLIngestionError):
    """Raised when URL is malformed, empty, or has an unsupported protocol."""
    pass


class HTTPFetchError(URLIngestionError):
    """Raised when HTTP request fails, returns 4xx/5xx, or times out."""
    pass


class EmptyContentError(URLIngestionError):
    """Raised when URL returns empty or unparseable content."""
    pass


class ContentChallengeError(URLIngestionError):
    """Raised when server returns an anti-bot challenge, cookie wall, captcha, or login block instead of clinical content."""
    pass


@dataclass
class URLIngestionResult:
    """Operational outcome of a successful URL ingestion."""
    source_url: str
    saved_path: Path
    filename: str
    content_sha256: str
    retrieval_timestamp: datetime
    is_direct_pdf: bool
    file_size_bytes: int
    content_type: str
    page_title: Optional[str] = None
    extracted_text_size: int = 0
    content_classification: str = "Web Page"
    resolved_source_url: Optional[str] = None
    retrieval_provider: str = "Direct HTTP"


# ==============================================================================
# URL VALIDATION & FILENAME SANITIZATION
# ==============================================================================

def validate_source_url(url: str) -> urllib.parse.ParseResult:
    """Validate that the provided string is a well-formed HTTP or HTTPS URL.

    Args:
        url: URL string to inspect.

    Returns:
        urllib.parse.ParseResult if valid.

    Raises:
        InvalidURLError: If empty, malformed, or using an unsupported scheme.
    """
    if not url or not isinstance(url, str):
        raise InvalidURLError("Source URL cannot be empty.")

    clean_url = url.strip()
    if not clean_url:
        raise InvalidURLError("Source URL cannot be empty.")

    # Disallow internal whitespace
    if re.search(r"\s", clean_url):
        raise InvalidURLError(f"Malformed URL containing whitespace: '{clean_url}'")

    try:
        parsed = urllib.parse.urlparse(clean_url)
    except Exception as e:
        raise InvalidURLError(f"Malformed URL '{clean_url}': {e}")

    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise InvalidURLError(
            f"Unsupported URL scheme '{parsed.scheme}'. Only 'http://' and 'https://' are supported."
        )

    if not parsed.netloc or not parsed.netloc.strip():
        raise InvalidURLError(f"Malformed URL missing network domain/host: '{clean_url}'")

    return parsed


def derive_safe_filename_from_url(
    url: str,
    content_hash: str,
    is_pdf: bool = False,
    max_slug_len: int = 40,
) -> str:
    """Derive a safe, traversal-free filename from a URL.

    Args:
        url: Target URL.
        content_hash: SHA-256 hash of the content.
        is_pdf: Whether the resource is known to be a direct PDF.
        max_slug_len: Maximum length of the slug prefix.

    Returns:
        Safe filename ending in '.pdf'.
    """
    parsed = urllib.parse.urlparse(url.strip())
    path_name = Path(parsed.path).name.strip()

    # If URL path ends with .pdf, use clean stem from that filename
    if path_name.lower().endswith(".pdf"):
        raw_stem = path_name[:-4]
        clean_stem = re.sub(r"[^a-zA-Z0-9_\-]", "_", raw_stem).strip("._-")
        clean_stem = re.sub(r"_+", "_", clean_stem)[:max_slug_len]
        if not clean_stem:
            clean_stem = "source_guideline"
        return f"{clean_stem}_{content_hash[:8]}.pdf"

    # Otherwise, generate a slug from domain and path
    slug_base = f"{parsed.netloc}_{parsed.path.strip('/')}"
    clean_slug = re.sub(r"[^a-zA-Z0-9_\-]", "_", slug_base).strip("._-")
    clean_slug = re.sub(r"_+", "_", clean_slug)[:max_slug_len]
    if not clean_slug:
        clean_slug = "clinical_source"

    return f"{clean_slug}_{content_hash[:8]}.pdf"


# ==============================================================================
# CONTENT VALIDATION & CHALLENGE / BOT-CHECK DETECTION
# ==============================================================================

CHALLENGE_PATTERNS = [
    re.compile(r"cookies\s+must\s+be\s+enabled", re.IGNORECASE),
    re.compile(r"enable\s+cookies", re.IGNORECASE),
    re.compile(r"checking\s+your\s+browser", re.IGNORECASE),
    re.compile(r"verify\s+you\s+are\s+human", re.IGNORECASE),
    re.compile(r"captcha", re.IGNORECASE),
    re.compile(r"access\s+denied", re.IGNORECASE),
    re.compile(r"forbidden", re.IGNORECASE),
    re.compile(r"login\s+required", re.IGNORECASE),
    re.compile(r"attention\s+required!\s*\|\s*cloudflare", re.IGNORECASE),
    re.compile(r"just\s+a\s+moment\.\.\.", re.IGNORECASE),
    re.compile(r"please\s+enable\s+javascript", re.IGNORECASE),
]


def detect_content_challenge(html_or_text: str, title: Optional[str] = None) -> Optional[str]:
    """Inspect content and title for automated bot-checks, cookie walls, or access denial.

    Args:
        html_or_text: Raw or extracted body text.
        title: Optional page title.

    Returns:
        Explanation string if challenge detected, None otherwise.
    """
    check_text = ((title or "") + "\n" + html_or_text[:12000]).lower()
    for pat in CHALLENGE_PATTERNS:
        match = pat.search(check_text)
        if match:
            matched_phrase = match.group(0)
            return f"Source returned browser/access challenge ('{matched_phrase}') instead of article content."
    return None





# ==============================================================================
# HTML PARSER & NORMALIZATION INTO PDF
# ==============================================================================

class HTMLClinicalTextExtractor(html.parser.HTMLParser):
    """Extracts structured text lines from clinical guideline HTML pages."""

    _HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
    _BLOCK_TAGS = {"p", "li", "div", "article", "section", "blockquote"}
    _IGNORE_TAGS = {"script", "style", "noscript", "svg", "nav", "footer", "header"}

    def __init__(self) -> None:
        super().__init__()
        self._ignore_depth = 0
        self._in_title = False
        self._current_tag: Optional[str] = None
        self._buffer: List[str] = []
        self._title_buffer: List[str] = []
        self.sections: List[Tuple[bool, str]] = []  # (is_heading, text)
        self.page_title: Optional[str] = None

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag_lower = tag.lower()
        if tag_lower == "title":
            self._in_title = True
            return
        if tag_lower in self._IGNORE_TAGS:
            self._ignore_depth += 1
            return
        if self._ignore_depth > 0:
            return

        if tag_lower in self._HEADING_TAGS or tag_lower in self._BLOCK_TAGS:
            self._flush_buffer()
            self._current_tag = tag_lower

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower == "title":
            self._in_title = False
            if self._title_buffer:
                clean_title = html_lib.unescape(" ".join(self._title_buffer)).strip()
                self.page_title = clean_title if clean_title else None
            return
        if tag_lower in self._IGNORE_TAGS:
            self._ignore_depth = max(0, self._ignore_depth - 1)
            return
        if self._ignore_depth > 0:
            return

        if tag_lower in self._HEADING_TAGS or tag_lower in self._BLOCK_TAGS:
            self._flush_buffer()
            self._current_tag = None

    def handle_data(self, data: str) -> None:
        if self._in_title:
            clean = data.strip()
            if clean:
                self._title_buffer.append(clean)
            return
        if self._ignore_depth > 0:
            return
        clean = data.strip()
        if clean:
            self._buffer.append(clean)

    def _flush_buffer(self) -> None:
        if not self._buffer:
            return
        full_text = html_lib.unescape(" ".join(self._buffer)).strip()
        self._buffer = []
        if not full_text:
            return

        is_heading = self._current_tag in self._HEADING_TAGS
        self.sections.append((is_heading, full_text))

    @property
    def total_text_chars(self) -> int:
        return sum(len(text) for _, text in self.sections)

    @property
    def content_classification(self) -> str:
        all_text = " ".join(t.lower() for _, t in self.sections)
        title_lower = (self.page_title or "").lower()
        if "abstract" in all_text or "abstract" in title_lower or "pmid" in all_text:
            return "Abstract / Citation Summary"
        if any(h in all_text for h in ["recommendation", "guideline", "treatment", "protocol", "pharmacotherapy"]):
            return "Clinical Guideline / Article"
        return "Web Page Content"


def normalize_html_to_pdf_bytes(
    html_content: str,
    source_url: str,
    retrieval_timestamp: datetime,
    max_lines_per_page: int = 36,
    max_pages: int = 25,
) -> Tuple[bytes, HTMLClinicalTextExtractor]:
    """Normalize HTML content into a standard PDF representation with clinical provenance.

    Args:
        html_content: Raw HTML text.
        source_url: Original source URL.
        retrieval_timestamp: Timestamp when the page was retrieved.
        max_lines_per_page: Line limit per PDF page.
        max_pages: Maximum pages to generate from web HTML to prevent unbounded PDFs.

    Returns:
        Tuple of (pdf_bytes, extractor_instance).

    Raises:
        EmptyContentError: If no readable text blocks could be extracted.
    """
    extractor = HTMLClinicalTextExtractor()
    try:
        extractor.feed(html_content)
        extractor.close()
    except Exception as e:
        logger.warning("HTML parsing warning for '%s': %s", source_url, e)

    if not extractor.sections:
        # Fallback regex strip if parser returned no sections
        clean_text = re.sub(r"<[^>]+>", " ", html_content)
        clean_text = html_lib.unescape(clean_text).strip()
        if not clean_text:
            raise EmptyContentError(f"No readable clinical text could be extracted from HTML at '{source_url}'.")
        raw_paras = [p.strip() for p in clean_text.splitlines() if p.strip()]
        extractor.sections = [(False, p) for p in raw_paras]

    if not extractor.sections:
        raise EmptyContentError(f"No readable text blocks found in HTML from '{source_url}'.")

    # Build line stream with provenance header on Page 1
    lines_pool: List[str] = [
        "1. Clinical Guideline Source Document (Retrieved via Web URL)",
        f"Source URL: {source_url}",
        f"Retrieved: {retrieval_timestamp.isoformat()}",
        f"Title: {extractor.page_title or 'Untitled Source'}",
        "=" * 70,
        "",
    ]

    heading_idx = 1
    for is_heading, text_block in extractor.sections:
        if is_heading:
            heading_idx += 1
            # Format as deterministic heading line recognized by detect_document_sections
            heading_line = f"Section {heading_idx}: {text_block[:70]}"
            lines_pool.append("")
            lines_pool.append(heading_line)
            lines_pool.append("-" * min(len(heading_line), 70))
        else:
            wrapped = textwrap.wrap(text_block, width=72)
            lines_pool.extend(wrapped)
            lines_pool.append("")

    # Paginate into pages
    pages: List[List[str]] = []
    current_page: List[str] = []
    for line in lines_pool:
        current_page.append(line)
        if len(current_page) >= max_lines_per_page:
            pages.append(current_page)
            current_page = []

    if current_page:
        pages.append(current_page)

    if not pages:
        raise EmptyContentError(f"Constructed 0 pages for HTML document from '{source_url}'.")

    # Bound maximum pages for web HTML documents
    if len(pages) > max_pages:
        pages = pages[:max_pages]
        pages[-1].append("[Notice: Source document content truncated to 25 pages for clinical processing]")

    return make_multipage_pdf_bytes(pages), extractor


# ==============================================================================
# URL FETCHER & INGESTION SERVICE
# ==============================================================================

def fetch_url_content(
    url: str,
    timeout_seconds: float = 10.0,
) -> Tuple[bytes, str, Dict[str, str]]:
    """Retrieve raw content and headers from an HTTP/HTTPS URL.

    Args:
        url: Validated URL string.
        timeout_seconds: Request timeout in seconds.

    Returns:
        Tuple of (content_bytes, content_type, headers_dict).

    Raises:
        InvalidURLError: If URL is invalid.
        HTTPFetchError: If network error, timeout, or HTTP 4xx/5xx occurs.
        EmptyContentError: If server returns 0 bytes.
    """
    validate_source_url(url)

    req = urllib.request.Request(
        url.strip(),
        headers={
            "User-Agent": "CKEA-Clinical-Knowledge-Evolution-Agent/1.0",
            "Accept": "application/pdf, text/html, application/xhtml+xml, */*",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            status_code = getattr(response, "status", 200)
            if status_code >= 400:
                raise HTTPFetchError(f"HTTP error {status_code} fetching URL: '{url}'")

            content_bytes = response.read()
            content_type = response.headers.get("Content-Type", "").lower()
            headers = dict(response.headers.items())

    except urllib.error.HTTPError as e:
        raise HTTPFetchError(f"HTTP {e.code} error fetching URL '{url}': {e.reason}")
    except urllib.error.URLError as e:
        if isinstance(e.reason, socket.timeout):
            raise HTTPFetchError(f"Request timed out after {timeout_seconds}s fetching URL '{url}'")
        raise HTTPFetchError(f"Network error fetching URL '{url}': {e.reason}")
    except (socket.timeout, TimeoutError):
        raise HTTPFetchError(f"Request timed out after {timeout_seconds}s fetching URL '{url}'")
    except HTTPFetchError:
        raise
    except Exception as e:
        raise HTTPFetchError(f"Unexpected error fetching URL '{url}': {e}")

    if not content_bytes or len(content_bytes.strip()) == 0:
        raise EmptyContentError(f"URL returned empty content (0 bytes): '{url}'")

    return content_bytes, content_type, headers


class URLIngestionService:
    """Deterministic clinical source URL ingestion service.

    Downloads direct PDFs or extracts and converts HTML clinical web pages into
    standard local PDFs in data/sources/.
    """

    def __init__(self, timeout_seconds: float = 10.0) -> None:
        self.timeout_seconds = timeout_seconds

    def ingest_url(
        self,
        url: str,
        source_dir: Union[str, Path],
    ) -> URLIngestionResult:
        """Fetch, normalize, and safely persist a clinical source from a URL.

        Args:
            url: Target HTTP or HTTPS URL.
            source_dir: Authorized destination directory (must be data/sources/).

        Returns:
            URLIngestionResult containing metadata and path to the saved PDF.

        Raises:
            InvalidURLError: On invalid URL scheme or format.
            HTTPFetchError: On HTTP error, connection error, or timeout.
            EmptyContentError: On empty payload or empty extracted text.
            ValueError: On security violation or path traversal attempt.
        """
        validate_source_url(url)

        target_dir = Path(source_dir).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

        retrieval_ts = datetime.now(timezone.utc)

        # Standard Web/PDF Ingestion Flow
        content_bytes, content_type, _ = fetch_url_content(url, timeout_seconds=self.timeout_seconds)

        # Detect direct PDF
        is_direct_pdf = (
            content_type.startswith("application/pdf")
            or content_bytes.startswith(b"%PDF-")
            or urllib.parse.urlparse(url).path.lower().endswith(".pdf")
        )

        page_title = None
        extracted_text_size = 0
        content_classification = "Direct PDF Document" if is_direct_pdf else "Web Page Content"
        retrieval_provider = "Direct HTTP"

        if is_direct_pdf:
            if not content_bytes.startswith(b"%PDF"):
                if b"%PDF" in content_bytes[:1024]:
                    pdf_start = content_bytes.find(b"%PDF")
                    pdf_bytes = content_bytes[pdf_start:]
                else:
                    raise EmptyContentError(f"URL was identified as PDF but does not contain valid PDF data: '{url}'")
            else:
                pdf_bytes = content_bytes
            extracted_text_size = len(pdf_bytes)
        else:
            # HTML page normalization with challenge detection
            html_text = content_bytes.decode("utf-8", errors="replace")

            # Validate against bot challenge, captcha, or cookie wall
            challenge = detect_content_challenge(html_text)
            if challenge:
                raise ContentChallengeError(challenge)

            # Route to Jina Reader for web content
            try:
                jina_service = JinaReaderService()
                jina_content, jina_title, jina_url = jina_service.retrieve_webpage(url, timeout_seconds=self.timeout_seconds)
            except Exception as e:
                if "Authentication error" in str(e) or "401" in str(e) or "403" in str(e):
                    raise HTTPFetchError(f"Jina API Authentication/Authorization error: {e}")
                raise HTTPFetchError(f"Jina Reader failed to retrieve URL: {e}")

            if not jina_content.strip():
                raise EmptyContentError(f"Jina Reader returned empty content for '{url}'")

            if jina_title:
                title_challenge = detect_content_challenge("", title=jina_title)
                if title_challenge:
                    raise ContentChallengeError(title_challenge)
            
            page_title = jina_title
            retrieval_provider = "Jina Reader"
            
            # Normalize Jina's markdown to PDF bytes
            # Build line stream with provenance header on Page 1
            lines_pool: List[str] = [
                "1. Clinical Guideline Source Document (Retrieved via Jina Reader)",
                f"Source URL: {url}",
                f"Title: {page_title or 'Untitled Source'}",
                "=" * 70,
                "",
            ]
            
            for line in jina_content.splitlines():
                if not line.strip():
                    lines_pool.append("")
                else:
                    lines_pool.extend(textwrap.wrap(line, width=72))
                    
            pages: List[List[str]] = []
            current_page: List[str] = []
            for line in lines_pool:
                current_page.append(line)
                if len(current_page) >= 36:
                    pages.append(current_page)
                    current_page = []
            if current_page:
                pages.append(current_page)
                
            if not pages:
                raise EmptyContentError(f"Constructed 0 pages for HTML document from '{url}'.")
                
            if len(pages) > 25:
                pages = pages[:25]
                pages[-1].append("[Notice: Source document content truncated to 25 pages for clinical processing]")
                
            pdf_bytes = make_multipage_pdf_bytes(pages)
            extracted_text_size = len(jina_content)
            content_classification = "Web Page Content"

        file_hash = hashlib.sha256(pdf_bytes).hexdigest()
        # Generate safe filename and write to source_dir
        filename = derive_safe_filename_from_url(url, content_hash=file_hash, is_pdf=is_direct_pdf)
        dest_path = (target_dir / filename).resolve()

        # Strict security check: destination must reside directly in target_dir
        if dest_path.parent != target_dir:
            raise ValueError(f"Security error: path traversal detected for URL '{url}' (target: {dest_path})")

        # Save bytes
        dest_path.write_bytes(pdf_bytes)

        logger.info(
            "URL ingested: %s -> %s (direct_pdf=%s, bytes=%d, hash=%s...)",
            url,
            dest_path.name,
            is_direct_pdf,
            len(pdf_bytes),
            file_hash[:8],
        )

        return URLIngestionResult(
            source_url=url,
            saved_path=dest_path,
            filename=dest_path.name,
            content_sha256=file_hash,
            retrieval_timestamp=retrieval_ts,
            is_direct_pdf=is_direct_pdf,
            file_size_bytes=len(pdf_bytes),
            content_type=content_type,
            page_title=page_title,
            extracted_text_size=extracted_text_size,
            content_classification=content_classification,
            resolved_source_url=None,
            retrieval_provider=retrieval_provider,
        )
