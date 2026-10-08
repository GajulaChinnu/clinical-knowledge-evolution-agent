"""Jina Reader retrieval provider for generic web page URLs.

Jina Reader (https://r.jina.ai/{url}) is a content-retrieval and normalization provider only.
It contains no CKEA business logic and no site-specific handling. Credentials come exclusively
from AppConfig (environment / .env) and are never logged or included in error messages.
"""

from dataclasses import dataclass, field
import logging
import random
import time
from typing import Callable, List, Optional, Tuple

import httpx

from app.services.config_service import AppConfig

logger = logging.getLogger("ckea.services.jina_reader")

DEFAULT_JINA_BASE_URL = "https://r.jina.ai"


class JinaReaderError(Exception):
    """Base error for Jina Reader retrieval failures."""

    transient: bool = False


class JinaConfigurationError(JinaReaderError):
    """Raised when JINA_API_KEY is not configured."""


class JinaAuthError(JinaReaderError):
    """Raised on 401/403 from the Jina API (invalid or unauthorized key)."""


class JinaQuotaError(JinaReaderError):
    """Raised on 402 from the Jina API (quota exhausted)."""


class JinaRateLimitError(JinaReaderError):
    """Raised on 429 from the Jina API."""

    transient = True


class JinaServerError(JinaReaderError):
    """Raised on 5xx from the Jina API."""

    transient = True


class JinaTimeoutError(JinaReaderError):
    """Raised when the Jina API does not respond within the timeout."""

    transient = True


class JinaNetworkError(JinaReaderError):
    """Raised on connection-level failures reaching the Jina API."""

    transient = True


class JinaResponseError(JinaReaderError):
    """Raised when the Jina API returns an unexpected or invalid payload."""


@dataclass
class JinaResult:
    """Normalized Jina Reader retrieval outcome."""

    content: str
    title: str
    resolved_url: str
    status_code: int
    attempts: int
    warnings: List[str] = field(default_factory=list)


class JinaReaderService:
    """Dedicated retrieval service for generic web page URLs using Jina Reader."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: float = 45.0,
        max_attempts: int = 3,
        backoff_base_seconds: float = 1.0,
        backoff_max_seconds: float = 8.0,
        client: Optional[httpx.Client] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_key = api_key.strip() if api_key and api_key.strip() else None
        self.base_url = (base_url or DEFAULT_JINA_BASE_URL).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max(1, max_attempts)
        self.backoff_base_seconds = backoff_base_seconds
        self.backoff_max_seconds = backoff_max_seconds
        self._client = client
        self._sleep = sleep

    @classmethod
    def from_config(cls, config: AppConfig, **kwargs) -> "JinaReaderService":
        """Build a service from application configuration."""
        return cls(
            api_key=config.jina_api_key,
            base_url=config.jina_reader_base_url,
            timeout_seconds=config.jina_timeout_seconds,
            **kwargs,
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def __repr__(self) -> str:
        key_state = "[REDACTED]" if self.api_key else "None"
        return f"JinaReaderService(base_url='{self.base_url}', api_key={key_state})"

    def retrieve(self, url: str) -> JinaResult:
        """Retrieve a web page through Jina Reader, retrying transient failures.

        Raises:
            JinaReaderError subclass describing the failure category.
        """
        if not self.api_key:
            raise JinaConfigurationError("JINA_API_KEY is not configured.")

        last_error: Optional[JinaReaderError] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = self._retrieve_once(url)
                result.attempts = attempt
                return result
            except JinaReaderError as e:
                last_error = e
                if not e.transient or attempt >= self.max_attempts:
                    raise
                delay = min(
                    self.backoff_max_seconds,
                    self.backoff_base_seconds * (2 ** (attempt - 1)),
                ) * (0.5 + random.random() / 2)
                logger.warning(
                    "Jina Reader transient failure (attempt %d/%d): %s; retrying in %.1fs",
                    attempt,
                    self.max_attempts,
                    e,
                    delay,
                )
                self._sleep(delay)

        raise last_error or JinaReaderError("Jina Reader retrieval failed.")

    def retrieve_webpage(self, url: str, timeout_seconds: Optional[float] = None) -> Tuple[str, str, str]:
        """Backwards-compatible tuple API: (markdown_content, page_title, resolved_url)."""
        if timeout_seconds is not None:
            self.timeout_seconds = timeout_seconds
        result = self.retrieve(url)
        return result.content, result.title, result.resolved_url

    def _retrieve_once(self, url: str) -> JinaResult:
        target_url = f"{self.base_url}/{url}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "X-Return-Format": "markdown",
        }

        client = self._client or httpx.Client(timeout=self.timeout_seconds)
        try:
            response = client.get(target_url, headers=headers, timeout=self.timeout_seconds)
        except httpx.TimeoutException:
            raise JinaTimeoutError(f"Request to Jina API timed out after {self.timeout_seconds}s")
        except httpx.HTTPError as e:
            raise JinaNetworkError(f"Network error communicating with Jina API: {type(e).__name__}")
        finally:
            if self._client is None:
                client.close()

        status = response.status_code
        if status in (401, 403):
            raise JinaAuthError(f"Jina API Authentication error ({status}). Check JINA_API_KEY.")
        if status == 402:
            raise JinaQuotaError("Jina API Quota exceeded (402).")
        if status == 429:
            raise JinaRateLimitError("Jina API Rate limited (429).")
        if status >= 500:
            raise JinaServerError(f"Jina API Server error ({status}).")
        if status >= 400:
            raise JinaResponseError(f"Jina API returned HTTP {status}.")

        try:
            data = response.json()
        except ValueError:
            raise JinaResponseError("Jina API returned invalid JSON")

        if not isinstance(data, dict):
            raise JinaResponseError("Jina API returned an unexpected payload shape")

        code = data.get("code")
        if code is not None and code != 200:
            raise JinaResponseError(f"Jina API error {code}: {data.get('message', 'Unknown error')}")

        res_data = data.get("data") or {}
        if not isinstance(res_data, dict):
            res_data = {}

        warnings: List[str] = []
        raw_warning = res_data.get("warning")
        if raw_warning:
            warnings.append(str(raw_warning))

        return JinaResult(
            content=str(res_data.get("content") or ""),
            title=str(res_data.get("title") or ""),
            resolved_url=str(res_data.get("url") or url),
            status_code=status,
            attempts=1,
            warnings=warnings,
        )
