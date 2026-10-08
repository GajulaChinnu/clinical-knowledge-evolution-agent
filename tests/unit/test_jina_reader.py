import json
import pytest
from unittest.mock import patch, MagicMock
from app.services.jina_reader_service import JinaReaderService, JinaReaderError

@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "test-key-123")
    monkeypatch.setenv("JINA_READER_BASE_URL", "https://r.jina.ai")

def test_jina_reader_initialization(mock_env):
    service = JinaReaderService()
    assert service.api_key == "test-key-123"
    assert service.base_url == "https://r.jina.ai"

def test_jina_reader_missing_api_key(monkeypatch):
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    service = JinaReaderService()
    with pytest.raises(JinaReaderError, match="JINA_API_KEY is not configured"):
        service.retrieve_webpage("https://example.com")

@patch("urllib.request.urlopen")
def test_retrieve_webpage_success(mock_urlopen, mock_env):
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read.return_value = json.dumps({
        "code": 200,
        "status": 20000,
        "data": {
            "title": "Example Domain",
            "content": "# Example Domain\n\nThis domain is for use in illustrative examples.",
            "url": "https://example.com"
        }
    }).encode("utf-8")
    
    mock_urlopen.return_value.__enter__.return_value = mock_response

    service = JinaReaderService()
    content, title, url = service.retrieve_webpage("https://example.com")
    
    assert title == "Example Domain"
    assert url == "https://example.com"
    assert "# Example Domain" in content

@patch("urllib.request.urlopen")
def test_retrieve_webpage_auth_error(mock_urlopen, mock_env):
    import urllib.error
    mock_urlopen.side_effect = urllib.error.HTTPError(
        url="https://r.jina.ai/https://example.com",
        code=401,
        msg="Unauthorized",
        hdrs=None,
        fp=None
    )

    service = JinaReaderService()
    with pytest.raises(JinaReaderError, match="Authentication error \\(401\\)"):
        service.retrieve_webpage("https://example.com")

@patch("urllib.request.urlopen")
def test_retrieve_webpage_rate_limit(mock_urlopen, mock_env):
    import urllib.error
    mock_urlopen.side_effect = urllib.error.HTTPError(
        url="https://r.jina.ai/https://example.com",
        code=429,
        msg="Too Many Requests",
        hdrs=None,
        fp=None
    )

    service = JinaReaderService()
    with pytest.raises(JinaReaderError, match="Rate limited \\(429\\)"):
        service.retrieve_webpage("https://example.com")
