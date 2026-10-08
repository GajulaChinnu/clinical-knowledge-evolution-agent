"""Unit tests for SharedLLMClient error classification and retry configuration (no live calls)."""

from unittest.mock import MagicMock

import httpx
import openai
import pytest

from app.schemas.extraction import ExtractionResponse
from app.services.config_service import AppConfig, MissingAPIKeyError
from app.services.llm_client import LLMOutputError, LLMTransportError, SharedLLMClient

REQ = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def _client(create_side_effect=None, content='{"recommendations": []}'):
    mock_openai = MagicMock()
    if create_side_effect is not None:
        mock_openai.chat.completions.create.side_effect = create_side_effect
    else:
        completion = MagicMock()
        completion.choices = [MagicMock()]
        completion.choices[0].message.content = content
        completion.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        mock_openai.chat.completions.create.return_value = completion
    return SharedLLMClient(config=AppConfig(groq_api_key="k"), client=mock_openai)


def _extract(client):
    return client.extract_recommendations("S", "text", 1, "doc")


def test_valid_output_parses():
    assert isinstance(_extract(_client()), ExtractionResponse)


@pytest.mark.parametrize("content", ["", "not json", '{"recommendations": "nope"}'])
def test_invalid_output_raises_llm_output_error(content):
    with pytest.raises(LLMOutputError):
        _extract(_client(content=content))


def test_llm_output_error_is_value_error_for_g2_routing():
    assert issubclass(LLMOutputError, ValueError)


def test_provider_schema_rejection_is_output_error():
    err = openai.BadRequestError("json_validate_failed", response=httpx.Response(400, request=REQ), body=None)
    with pytest.raises(LLMOutputError):
        _extract(_client(create_side_effect=err))


@pytest.mark.parametrize(
    "err",
    [
        openai.APIConnectionError(request=REQ),
        openai.APITimeoutError(request=REQ),
        openai.RateLimitError("rate", response=httpx.Response(429, request=REQ), body=None),
        openai.InternalServerError("boom", response=httpx.Response(503, request=REQ), body=None),
    ],
)
def test_transport_failures_raise_transport_error_not_output_error(err):
    with pytest.raises(LLMTransportError) as exc_info:
        _extract(_client(create_side_effect=err))
    assert not isinstance(exc_info.value, ValueError)


def test_sdk_client_built_with_explicit_bounded_retries_and_timeout(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("app.services.llm_client.OpenAI", FakeOpenAI)
    cfg = AppConfig(groq_api_key="secret-groq", llm_max_retries=2, llm_timeout_seconds=45)
    SharedLLMClient(config=cfg).client
    assert captured["max_retries"] == 2
    assert captured["timeout"] == 45
    assert "secret-groq" not in repr(SharedLLMClient(config=cfg))


def test_missing_key_raises():
    with pytest.raises(MissingAPIKeyError):
        SharedLLMClient(config=AppConfig()).client
