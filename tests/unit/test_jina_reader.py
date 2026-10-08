"""Unit tests for the Jina Reader retrieval provider (mocked transport, no live calls)."""

import httpx
import pytest

from app.services.config_service import AppConfig, load_config
from app.services.jina_reader_service import (
    JinaAuthError,
    JinaConfigurationError,
    JinaQuotaError,
    JinaRateLimitError,
    JinaReaderError,
    JinaReaderService,
    JinaResponseError,
    JinaTimeoutError,
)

TEST_KEY = "test-key-123"


def _ok_payload(content="# Example Domain\n\nThis domain is for use in illustrative examples.", warning=None):
    data = {"title": "Example Domain", "content": content, "url": "https://example.com"}
    if warning:
        data["warning"] = warning
    return {"code": 200, "status": 20000, "data": data}


def _service(handler, **kwargs) -> JinaReaderService:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return JinaReaderService(api_key=TEST_KEY, client=client, sleep=lambda _s: None, **kwargs)


def test_jina_reader_initialization_from_config():
    config = AppConfig(jina_api_key=TEST_KEY, jina_reader_base_url="https://r.jina.ai/", jina_timeout_seconds=50)
    service = JinaReaderService.from_config(config)
    assert service.api_key == TEST_KEY
    assert service.base_url == "https://r.jina.ai"
    assert service.timeout_seconds == 50
    assert TEST_KEY not in repr(service)


def test_jina_key_loaded_from_env_file_without_os_environ(tmp_path, monkeypatch):
    """Defect 2: JINA_API_KEY in .env must reach the service (config uses dotenv_values)."""
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("JINA_API_KEY=file-jina-key\nJINA_READER_BASE_URL=https://r.jina.ai\n", encoding="utf-8")

    config = load_config(env_file=env_file, load_os_environ=False)
    service = JinaReaderService.from_config(config)

    assert config.has_jina_api_key
    assert service.is_configured
    assert service.api_key == "file-jina-key"


def test_jina_key_is_redacted_in_config_output():
    config = AppConfig(jina_api_key="super-secret-jina")
    assert "super-secret-jina" not in repr(config)
    assert "super-secret-jina" not in str(config)
    assert config.to_safe_dict()["jina_api_key"] == "[REDACTED]"


def test_jina_reader_missing_api_key():
    service = JinaReaderService(api_key=None)
    with pytest.raises(JinaConfigurationError, match="JINA_API_KEY is not configured"):
        service.retrieve_webpage("https://example.com")


def test_retrieve_webpage_success_sends_bearer_and_target_url():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=_ok_payload())

    content, title, url = _service(handler).retrieve_webpage("https://example.com/page")

    assert seen["url"] == "https://r.jina.ai/https://example.com/page"
    assert seen["auth"] == f"Bearer {TEST_KEY}"
    assert title == "Example Domain"
    assert url == "https://example.com"
    assert "# Example Domain" in content


def test_retrieve_preserves_non_ascii_clinical_symbols():
    text = "Target HbA1c ≤ 7.0 %; eGFR ≥ 30 mL/min; dose 500 µg ± 10 %"
    result = _service(lambda r: httpx.Response(200, json=_ok_payload(content=text))).retrieve("https://example.com")
    assert result.content == text


def test_retrieve_surfaces_provider_warning():
    handler = lambda r: httpx.Response(200, json=_ok_payload(warning="Target URL returned error 403: Forbidden"))
    result = _service(handler).retrieve("https://example.com")
    assert result.warnings == ["Target URL returned error 403: Forbidden"]


@pytest.mark.parametrize(
    "status, exc, pattern",
    [
        (401, JinaAuthError, r"Authentication error \(401\)"),
        (403, JinaAuthError, r"Authentication error \(403\)"),
        (402, JinaQuotaError, r"Quota exceeded \(402\)"),
        (404, JinaResponseError, r"HTTP 404"),
    ],
)
def test_non_transient_http_errors_are_not_retried(status, exc, pattern):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(status)

    with pytest.raises(exc, match=pattern):
        _service(handler).retrieve("https://example.com")
    assert len(calls) == 1


def test_rate_limit_is_retried_then_raised():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429)

    with pytest.raises(JinaRateLimitError, match=r"Rate limited \(429\)"):
        _service(handler).retrieve("https://example.com")
    assert len(calls) == 3


def test_server_error_recovers_on_retry():
    responses = iter([httpx.Response(503), httpx.Response(200, json=_ok_payload())])
    result = _service(lambda r: next(responses)).retrieve("https://example.com")
    assert result.attempts == 2
    assert result.title == "Example Domain"


def test_timeout_is_retried_and_reported():
    def handler(request):
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(JinaTimeoutError, match="timed out"):
        _service(handler, timeout_seconds=1.0).retrieve("https://example.com")


def test_invalid_json_and_error_codes():
    with pytest.raises(JinaResponseError, match="invalid JSON"):
        _service(lambda r: httpx.Response(200, content=b"not json")).retrieve("https://example.com")
    with pytest.raises(JinaResponseError, match="Jina API error 422"):
        _service(lambda r: httpx.Response(200, json={"code": 422, "message": "bad"})).retrieve("https://example.com")


def test_errors_never_contain_api_key():
    def handler(request):
        raise httpx.ConnectError("boom", request=request)

    with pytest.raises(JinaReaderError) as exc_info:
        _service(handler).retrieve("https://example.com")
    assert TEST_KEY not in str(exc_info.value)


def test_service_module_does_not_read_os_environ():
    import inspect
    import app.services.jina_reader_service as mod

    assert "os.environ" not in inspect.getsource(mod)
