"""CKEA URL Ingestion Service for Clinical Documents.

Routing (source-agnostic; no site-specific handling):
1. Validate the URL (HTTP/HTTPS only) and block private/loopback/link-local targets
   unless ALLOW_PRIVATE_HOSTS is explicitly enabled (local demo only).
2. Probe the content type (HEAD, falling back to a small ranged GET).
3. PDF   -> direct download (size-capped) -> existing PDF path.
   Other -> Jina Reader ONLY. Web pages are never fetched directly, and bot-challenge
   detection runs only on the retrieval provider's output.
   Private host + ALLOW_PRIVATE_HOSTS -> direct local fetch (Jina cannot reach it).
4. Normalized content is persisted under data/sources/ with full retrieval provenance.

Walled sources (login, CAPTCHA, bot-check) fail safely; protection is never bypassed.
Zero LLM calls in the ingestion service.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import html as html_lib
import html.parser
import ipaddress
import logging
from pathlib import Path
import re
import socket
import textwrap
from typing import Callable, Iterable, List, Optional, Tuple, Union
import urllib.parse

import httpx

from app.services.config_service import AppConfig, load_config
from app.services.evaluation_corpus import make_multipage_pdf_bytes
from app.services.jina_reader_service import JinaReaderError, JinaReaderService

logger = logging.getLogger("ckea.services.url_ingestion")


class URLIngestionError(Exception):
    """Base exception for all URL ingestion errors."""
    pass


class InvalidURLError(URLIngestionError):
    """Raised when URL is malformed, empty, or has an unsupported protocol."""
    pass


class BlockedURLError(InvalidURLError):
    """Raised when a URL targets a private, loopback, link-local or otherwise non-public host."""
    pass


class HTTPFetchError(URLIngestionError):
    """Raised when HTTP request fails, returns 4xx/5xx, or times out."""
    pass


class RetrievalProviderError(HTTPFetchError):
    """Raised when the web retrieval provider (Jina Reader) fails or is not configured."""
    pass


class EmptyContentError(URLIngestionError):
    """Raised when URL returns empty or unparseable content."""
    pass


class ContentTooLargeError(URLIngestionError):
    """Raised when a download exceeds the configured maximum size."""
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
    routing_decision: str = "pdf_direct"
    probe_status_code: Optional[int] = None
    probe_content_type: Optional[str] = None
    probe_error: Optional[str] = None
    provider_status_code: Optional[int] = None
    provider_attempts: int = 0
    provider_warnings: List[str] = field(default_factory=list)

    def provenance_metadata(self) -> dict:
        """Retrieval provenance for IngestedDocument.doc_metadata."""
        return {
            "retrieval_provider": self.retrieval_provider,
            "routing_decision": self.routing_decision,
            "resolved_source_url": self.resolved_source_url,
            "probe_status_code": self.probe_status_code,
            "probe_content_type": self.probe_content_type,
            "probe_error": self.probe_error,
            "provider_status_code": self.provider_status_code,
            "provider_attempts": self.provider_attempts,
            "provider_warnings": list(self.provider_warnings),
        }


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

# Challenge detection runs ONLY on retrieval-provider output (never on direct-HTTP HTML).
# Broad phrases are only meaningful on short bodies: a full clinical article can legitimately
# mention "forbidden", "access denied" or "captcha". Long bodies are judged by title only.
SHORT_BODY_CHARS = 2000

TITLE_CHALLENGE_PATTERNS = [
    re.compile(r"just\s+a\s+moment", re.IGNORECASE),
    re.compile(r"attention\s+required", re.IGNORECASE),
    re.compile(r"access\s+denied", re.IGNORECASE),
    re.compile(r"are\s+you\s+a\s+robot", re.IGNORECASE),
    re.compile(r"captcha", re.IGNORECASE),
    re.compile(r"security\s+check", re.IGNORECASE),
    re.compile(r"verify(ing)?\s+(that\s+)?you\s+are\s+(a\s+)?human", re.IGNORECASE),
    re.compile(r"bot\s+(verification|check|detection)", re.IGNORECASE),
    re.compile(r"ddos\s+protection", re.IGNORECASE),
]

SHORT_BODY_CHALLENGE_PATTERNS = [
    re.compile(r"cookies\s+must\s+be\s+enabled", re.IGNORECASE),
    re.compile(r"enable\s+cookies", re.IGNORECASE),
    re.compile(r"checking\s+your\s+browser", re.IGNORECASE),
    re.compile(r"verify\s+(that\s+)?you\s+are\s+(a\s+)?human", re.IGNORECASE),
    re.compile(r"captcha", re.IGNORECASE),
    re.compile(r"access\s+denied", re.IGNORECASE),
    re.compile(r"\bforbidden\b", re.IGNORECASE),
    re.compile(r"login\s+required", re.IGNORECASE),
    re.compile(r"attention\s+required!?\s*\|\s*cloudflare", re.IGNORECASE),
    re.compile(r"just\s+a\s+moment\.\.\.", re.IGNORECASE),
    re.compile(r"please\s+enable\s+javascript", re.IGNORECASE),
]

PROVIDER_WARNING_PATTERN = re.compile(
    r"\b(401|403|429|503)\b|captcha|blocked|challenge|forbidden", re.IGNORECASE
)


def _compact(text: str) -> str:
    return " ".join((text or "").split())


def detect_content_challenge(
    html_or_text: str,
    title: Optional[str] = None,
    warnings: Optional[Iterable[str]] = None,
) -> Optional[str]:
    """Decide whether retrieved content is an access wall rather than source content.

    Args:
        html_or_text: Retrieved body text (provider output).
        title: Optional page title.
        warnings: Optional provider warnings (e.g. "Target URL returned error 403").

    Returns:
        Explanation string if a challenge is detected, None otherwise.
    """
    for pat in TITLE_CHALLENGE_PATTERNS:
        match = pat.search(title or "")
        if match:
            return f"Source returned browser/access challenge ('{match.group(0)}') instead of article content."

    body = _compact(html_or_text)
    if len(body) > SHORT_BODY_CHARS:
        return None

    for pat in SHORT_BODY_CHALLENGE_PATTERNS:
        match = pat.search(body)
        if match:
            return f"Source returned browser/access challenge ('{match.group(0)}') instead of article content."

    for warning in warnings or ():
        match = PROVIDER_WARNING_PATTERN.search(warning or "")
        if match:
            return (
                f"Retrieval provider reported the source as blocked ('{match.group(0)}') "
                "and only a short body was returned."
            )

    return None


# ==============================================================================
# HTML PARSER & NORMALIZATION INTO PDF (local demo path only)
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


def normalize_text_to_pdf_bytes(
    text: str,
    source_url: str,
    title: Optional[str],
    provider_label: str,
    max_lines_per_page: int = 36,
    max_pages: int = 25,
) -> bytes:
    """Render provider-normalized text into the PDF representation consumed downstream.

    The header deliberately excludes the retrieval timestamp so unchanged content
    produces an identical SHA-256 (idempotent versioning).
    """
    lines_pool: List[str] = [
        f"1. Clinical Guideline Source Document (Retrieved via {provider_label})",
        f"Source URL: {source_url}",
        f"Title: {title or 'Untitled Source'}",
        "=" * 70,
        "",
    ]
    for line in text.splitlines():
        if not line.strip():
            lines_pool.append("")
        else:
            lines_pool.extend(textwrap.wrap(line, width=72))

    pages: List[List[str]] = [
        lines_pool[i:i + max_lines_per_page] for i in range(0, len(lines_pool), max_lines_per_page)
    ]
    if not pages:
        raise EmptyContentError(f"Constructed 0 pages for document from '{source_url}'.")
    if len(pages) > max_pages:
        pages = pages[:max_pages]
        pages[-1].append(f"[Notice: Source document content truncated to {max_pages} pages for clinical processing]")
    return make_multipage_pdf_bytes(pages)


# ==============================================================================
# HOST GUARD, PROBE & DOWNLOAD
# ==============================================================================

HostResolver = Callable[[str], List[str]]

PROBE_MAX_BYTES = 64 * 1024
MAX_REDIRECTS = 5
_USER_AGENT = "CKEA-Clinical-Knowledge-Evolution-Agent/1.0"


def resolve_host_addresses(host: str) -> List[str]:
    """Resolve a hostname to its IP addresses."""
    infos = socket.getaddrinfo(host, None)
    return sorted({info[4][0] for info in infos})


def _is_non_public_address(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def is_private_host(host: str, resolver: Optional[HostResolver] = None) -> bool:
    """Return True when the host is, or resolves to, a non-public address.

    Raises:
        InvalidURLError: If the host is empty.
        HTTPFetchError: If the host cannot be resolved.
    """
    if not host:
        raise InvalidURLError("URL has no host.")
    host = host.strip("[]").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return _is_non_public_address(host)
    except ValueError:
        pass
    try:
        addresses = (resolver or resolve_host_addresses)(host)
    except (OSError, UnicodeError) as e:
        raise HTTPFetchError(f"Could not resolve host '{host}': {e}")
    if not addresses:
        raise HTTPFetchError(f"Could not resolve host '{host}'.")
    return any(_is_non_public_address(a) for a in addresses)


def guard_url_host(url: str, allow_private_hosts: bool, resolver: Optional[HostResolver] = None) -> bool:
    """Validate a URL and enforce the private-host policy.

    Returns:
        True if the host is private (only possible when allow_private_hosts is set).

    Raises:
        InvalidURLError, BlockedURLError, HTTPFetchError.
    """
    parsed = validate_source_url(url)
    private = is_private_host(parsed.hostname or "", resolver)
    if private and not allow_private_hosts:
        raise BlockedURLError(
            f"URL host '{parsed.hostname}' is private, loopback or link-local and is blocked. "
            "Set ALLOW_PRIVATE_HOSTS=true only for local demos."
        )
    return private


@dataclass
class ProbeResult:
    """Outcome of the content-type probe."""
    final_url: str
    is_pdf: bool
    status_code: Optional[int] = None
    content_type: str = ""
    method: str = "none"
    error: Optional[str] = None


def _guarded_request(
    client: httpx.Client,
    method: str,
    url: str,
    headers: dict,
    timeout: float,
    allow_private_hosts: bool,
    resolver: Optional[HostResolver],
    max_bytes: int,
    truncate: bool,
) -> Tuple[httpx.Response, bytes, str]:
    """Issue a request, following redirects manually so every hop passes the host guard.

    Returns (response, body_bytes, final_url). The body is read up to max_bytes; beyond that
    it is truncated (truncate=True) or ContentTooLargeError is raised.
    """
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        guard_url_host(current, allow_private_hosts, resolver)
        with client.stream(method, current, headers=headers, timeout=timeout, follow_redirects=False) as resp:
            if resp.is_redirect and resp.headers.get("location"):
                current = urllib.parse.urljoin(current, resp.headers["location"])
                continue
            chunks: List[bytes] = []
            total = 0
            if method != "HEAD":
                for chunk in resp.iter_bytes():
                    chunks.append(chunk)
                    total += len(chunk)
                    if total > max_bytes:
                        if truncate:
                            break
                        raise ContentTooLargeError(
                            f"Download from '{url}' exceeded the {max_bytes}-byte limit."
                        )
            body = b"".join(chunks)
            return resp, (body[:max_bytes] if truncate else body), current
    raise HTTPFetchError(f"Too many redirects fetching URL '{url}'.")


def _looks_like_pdf(content_type: str, body: bytes, url: str) -> bool:
    return (
        content_type.lower().startswith("application/pdf")
        or body.startswith(b"%PDF-")
        or urllib.parse.urlparse(url).path.lower().endswith(".pdf")
    )


def probe_url(
    url: str,
    client: httpx.Client,
    timeout_seconds: float,
    allow_private_hosts: bool = False,
    resolver: Optional[HostResolver] = None,
) -> ProbeResult:
    """Determine whether a URL serves a PDF without inspecting page HTML.

    A failed probe (e.g. a bot wall answering 403 to direct requests) is not fatal: the URL
    is routed to Jina Reader unless its path clearly names a PDF.
    """
    path_is_pdf = urllib.parse.urlparse(url).path.lower().endswith(".pdf")
    errors: List[str] = []
    headers = {"User-Agent": _USER_AGENT, "Accept": "application/pdf, text/html;q=0.9, */*;q=0.8"}

    attempts = (
        ("HEAD", "HEAD", headers),
        ("GET-range", "GET", {**headers, "Range": f"bytes=0-{PROBE_MAX_BYTES - 1}"}),
    )
    for label, method, hdrs in attempts:
        try:
            resp, body, final_url = _guarded_request(
                client, method, url, hdrs, timeout_seconds,
                allow_private_hosts, resolver, PROBE_MAX_BYTES, truncate=True,
            )
        except InvalidURLError:
            raise
        except (httpx.HTTPError, HTTPFetchError) as e:
            errors.append(f"{label}: {type(e).__name__}")
            continue
        content_type = resp.headers.get("content-type", "").lower()
        if resp.status_code >= 400:
            errors.append(f"{label}: HTTP {resp.status_code}")
            continue
        if label == "HEAD" and not content_type:
            errors.append("HEAD: no content-type")
            continue
        return ProbeResult(
            final_url=final_url,
            is_pdf=_looks_like_pdf(content_type, body, final_url),
            status_code=resp.status_code,
            content_type=content_type,
            method=label,
        )

    return ProbeResult(
        final_url=url,
        is_pdf=path_is_pdf,
        method="none",
        error="; ".join(errors) or None,
    )


def fetch_url_content(
    url: str,
    timeout_seconds: float = 30.0,
    max_bytes: int = 50 * 1024 * 1024,
    client: Optional[httpx.Client] = None,
    allow_private_hosts: bool = False,
    resolver: Optional[HostResolver] = None,
) -> Tuple[bytes, str, dict]:
    """Download a URL directly (PDFs, or local demo pages) with host guard and size cap.

    Returns:
        Tuple of (content_bytes, content_type, headers_dict).

    Raises:
        InvalidURLError, BlockedURLError, HTTPFetchError, ContentTooLargeError, EmptyContentError.
    """
    validate_source_url(url)
    owns_client = client is None
    client = client or httpx.Client()
    headers = {"User-Agent": _USER_AGENT, "Accept": "application/pdf, text/html, */*"}
    try:
        resp, body, _ = _guarded_request(
            client, "GET", url.strip(), headers, timeout_seconds,
            allow_private_hosts, resolver, max_bytes, truncate=False,
        )
    except httpx.TimeoutException:
        raise HTTPFetchError(f"Request timed out after {timeout_seconds}s fetching URL '{url}'")
    except httpx.HTTPError as e:
        raise HTTPFetchError(f"Network error fetching URL '{url}': {type(e).__name__}: {e}")
    finally:
        if owns_client:
            client.close()

    if resp.status_code >= 400:
        raise HTTPFetchError(f"HTTP {resp.status_code} error fetching URL '{url}': {resp.reason_phrase}")
    if not body or not body.strip():
        raise EmptyContentError(f"URL returned empty content (0 bytes): '{url}'")
    return body, resp.headers.get("content-type", "").lower(), dict(resp.headers.items())


# ==============================================================================
# INGESTION SERVICE
# ==============================================================================

class URLIngestionService:
    """Deterministic, source-agnostic clinical source URL ingestion service."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        timeout_seconds: Optional[float] = None,
        jina_service: Optional[JinaReaderService] = None,
        http_client: Optional[httpx.Client] = None,
        host_resolver: Optional[HostResolver] = None,
    ) -> None:
        self.config = config or load_config()
        self.probe_timeout_seconds = timeout_seconds or self.config.url_probe_timeout_seconds
        self.download_timeout_seconds = timeout_seconds or self.config.url_download_timeout_seconds
        self.jina_service = jina_service or JinaReaderService.from_config(self.config)
        self._http_client = http_client
        self.host_resolver = host_resolver

    def ingest_url(
        self,
        url: str,
        source_dir: Union[str, Path],
    ) -> URLIngestionResult:
        """Probe, retrieve, normalize and persist a clinical source from a URL.

        Raises:
            InvalidURLError / BlockedURLError: Invalid or disallowed URL.
            HTTPFetchError / RetrievalProviderError: Network, HTTP or provider failure.
            ContentChallengeError: Source is behind a login/CAPTCHA/bot wall.
            EmptyContentError / ContentTooLargeError: Unusable payload.
            ValueError: Path traversal attempt.
        """
        validate_source_url(url)
        url = url.strip()
        allow_private = self.config.allow_private_hosts
        host_is_private = guard_url_host(url, allow_private, self.host_resolver)

        target_dir = Path(source_dir).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        retrieval_ts = datetime.now(timezone.utc)

        client = self._http_client or httpx.Client()
        try:
            probe = probe_url(url, client, self.probe_timeout_seconds, allow_private, self.host_resolver)
            logger.info(
                "URL probe: %s -> is_pdf=%s status=%s content_type=%s method=%s error=%s",
                url, probe.is_pdf, probe.status_code, probe.content_type or "-", probe.method, probe.error,
            )

            result = URLIngestionResult(
                source_url=url,
                saved_path=Path(),
                filename="",
                content_sha256="",
                retrieval_timestamp=retrieval_ts,
                is_direct_pdf=probe.is_pdf,
                file_size_bytes=0,
                content_type=probe.content_type,
                probe_status_code=probe.status_code,
                probe_content_type=probe.content_type or None,
                probe_error=probe.error,
            )

            if probe.is_pdf:
                pdf_bytes = self._download_pdf(probe.final_url, client, result)
            elif host_is_private:
                pdf_bytes = self._retrieve_local_html(url, client, retrieval_ts, result)
            else:
                pdf_bytes = self._retrieve_via_jina(url, result)
        finally:
            if self._http_client is None:
                client.close()

        file_hash = hashlib.sha256(pdf_bytes).hexdigest()
        filename = derive_safe_filename_from_url(url, content_hash=file_hash, is_pdf=probe.is_pdf)
        dest_path = (target_dir / filename).resolve()
        if dest_path.parent != target_dir:
            raise ValueError(f"Security error: path traversal detected for URL '{url}' (target: {dest_path})")
        dest_path.write_bytes(pdf_bytes)

        result.saved_path = dest_path
        result.filename = dest_path.name
        result.content_sha256 = file_hash
        result.file_size_bytes = len(pdf_bytes)

        logger.info(
            "URL ingested: %s -> %s (routing=%s, provider=%s, bytes=%d, hash=%s...)",
            url, dest_path.name, result.routing_decision, result.retrieval_provider,
            len(pdf_bytes), file_hash[:8],
        )
        return result

    def _download_pdf(self, url: str, client: httpx.Client, result: URLIngestionResult) -> bytes:
        content_bytes, content_type, _ = fetch_url_content(
            url,
            timeout_seconds=self.download_timeout_seconds,
            max_bytes=self.config.max_download_bytes,
            client=client,
            allow_private_hosts=self.config.allow_private_hosts,
            resolver=self.host_resolver,
        )
        pdf_start = content_bytes.find(b"%PDF", 0, 1024)
        if pdf_start < 0:
            raise EmptyContentError(f"URL was identified as PDF but does not contain valid PDF data: '{url}'")
        pdf_bytes = content_bytes[pdf_start:]
        result.routing_decision = "pdf_direct"
        result.retrieval_provider = "Direct HTTP"
        result.content_type = content_type or result.content_type
        result.content_classification = "Direct PDF Document"
        result.extracted_text_size = len(pdf_bytes)
        result.resolved_source_url = url
        return pdf_bytes

    def _retrieve_via_jina(self, url: str, result: URLIngestionResult) -> bytes:
        if not self.jina_service.is_configured:
            raise RetrievalProviderError(
                "JINA_API_KEY is not configured. Web pages are retrieved via Jina Reader only."
            )
        try:
            jina = self.jina_service.retrieve(url)
        except JinaReaderError as e:
            raise RetrievalProviderError(f"Jina Reader failed to retrieve URL: {e}") from e

        result.routing_decision = "jina"
        result.retrieval_provider = "Jina Reader"
        result.provider_status_code = jina.status_code
        result.provider_attempts = jina.attempts
        result.provider_warnings = list(jina.warnings)
        result.resolved_source_url = jina.resolved_url
        result.page_title = jina.title or None
        logger.info(
            "Jina Reader invoked for %s: status=%s attempts=%d chars=%d warnings=%s",
            url, jina.status_code, jina.attempts, len(jina.content), jina.warnings or "-",
        )

        challenge = detect_content_challenge(jina.content, title=jina.title, warnings=jina.warnings)
        if challenge:
            raise ContentChallengeError(challenge)
        if not jina.content.strip():
            raise EmptyContentError(f"Jina Reader returned empty content for '{url}'")

        result.extracted_text_size = len(jina.content)
        result.content_classification = "Web Page Content"
        return normalize_text_to_pdf_bytes(jina.content, url, jina.title, "Jina Reader")

    def _retrieve_local_html(
        self, url: str, client: httpx.Client, retrieval_ts: datetime, result: URLIngestionResult,
    ) -> bytes:
        """Local demo only (ALLOW_PRIVATE_HOSTS): Jina's cloud service cannot reach private hosts."""
        content_bytes, content_type, _ = fetch_url_content(
            url,
            timeout_seconds=self.download_timeout_seconds,
            max_bytes=self.config.max_download_bytes,
            client=client,
            allow_private_hosts=True,
            resolver=self.host_resolver,
        )
        html_text = content_bytes.decode("utf-8", errors="replace")
        pdf_bytes, extractor = normalize_html_to_pdf_bytes(html_text, url, retrieval_ts)
        body_text = " ".join(t for _, t in extractor.sections)
        challenge = detect_content_challenge(body_text, title=extractor.page_title)
        if challenge:
            raise ContentChallengeError(challenge)
        result.routing_decision = "local_direct_html"
        result.retrieval_provider = "Direct HTTP (local demo)"
        result.content_type = content_type
        result.page_title = extractor.page_title
        result.extracted_text_size = extractor.total_text_chars
        result.content_classification = extractor.content_classification
        result.resolved_source_url = url
        return pdf_bytes
