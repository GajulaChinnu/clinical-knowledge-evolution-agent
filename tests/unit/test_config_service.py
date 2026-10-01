"""Unit tests for the CKEA configuration service."""

from pathlib import Path
import os
import pytest

from app.services.config_service import (
    AppConfig,
    MissingAPIKeyError,
    load_config,
)


def test_default_values():
    """Verify that AppConfig default values match architecture specifications."""
    config = AppConfig()

    assert config.groq_api_key is None
    assert config.has_api_key is False
    assert config.groq_model == "openai/gpt-oss-20b"
    assert config.groq_base_url == "https://api.groq.com/openai/v1"
    assert config.database_url == "sqlite:///./data/ckea.db"
    assert config.source_dir == Path("./data/sources")
    assert config.protocol_dir == Path("./data/protocols")
    assert config.chroma_dir == Path("./data/chroma")
    assert config.output_dir == Path("./data/output")
    assert config.extraction_confidence_threshold == 0.70
    assert config.comparison_confidence_threshold == 0.70
    assert config.protocol_similarity_threshold == 0.70


def test_configuration_loading_from_file(tmp_path: Path):
    """Verify loading configuration from a .env file."""
    env_content = (
        "GROQ_API_KEY=test-file-key\n"
        "GROQ_MODEL=groq-custom\n"
        "GROQ_BASE_URL=https://api.groq.com/openai/v1\n"
        "DATABASE_URL=sqlite:///./test.db\n"
        "SOURCE_DIR=./custom/sources\n"
        "EXTRACTION_CONFIDENCE_THRESHOLD=0.85\n"
        "COMPARISON_CONFIDENCE_THRESHOLD=0.75\n"
        "PROTOCOL_SIMILARITY_THRESHOLD=0.80\n"
    )
    env_file = tmp_path / ".env.test"
    env_file.write_text(env_content, encoding="utf-8")

    config = load_config(env_file=env_file, load_os_environ=False)

    assert config.has_api_key is True
    assert config.get_api_key() == "test-file-key"
    assert config.groq_model == "groq-custom"
    assert config.groq_base_url == "https://api.groq.com/openai/v1"
    assert config.database_url == "sqlite:///./test.db"
    assert config.source_dir == Path("./custom/sources")
    assert config.extraction_confidence_threshold == 0.85
    assert config.comparison_confidence_threshold == 0.75
    assert config.protocol_similarity_threshold == 0.80


def test_environment_variable_override(monkeypatch: pytest.MonkeyPatch):
    """Verify that OS environment variables override defaults and file values."""
    monkeypatch.setenv("GROQ_API_KEY", "env-override-key")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b-fast")
    monkeypatch.setenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///./override.db")
    monkeypatch.setenv("SOURCE_DIR", "./override/sources")
    monkeypatch.setenv("EXTRACTION_CONFIDENCE_THRESHOLD", "0.90")
    monkeypatch.setenv("COMPARISON_CONFIDENCE_THRESHOLD", "0.85")
    monkeypatch.setenv("PROTOCOL_SIMILARITY_THRESHOLD", "0.75")

    config = load_config(env_file=None, load_os_environ=True)

    assert config.has_api_key is True
    assert config.get_api_key() == "env-override-key"
    assert config.groq_model == "openai/gpt-oss-20b-fast"
    assert config.groq_base_url == "https://api.groq.com/openai/v1"
    assert config.database_url == "sqlite:///./override.db"
    assert config.source_dir == Path("./override/sources")
    assert config.extraction_confidence_threshold == 0.90
    assert config.comparison_confidence_threshold == 0.85
    assert config.protocol_similarity_threshold == 0.75


def test_explicit_overrides_precedence(monkeypatch: pytest.MonkeyPatch):
    """Verify that explicit kwargs override both environment variables and defaults."""
    monkeypatch.setenv("GROQ_MODEL", "from-env")

    config = load_config(
        load_os_environ=True,
        groq_model="from-override",
        extraction_confidence_threshold=0.95,
    )

    assert config.groq_model == "from-override"
    assert config.extraction_confidence_threshold == 0.95


def test_missing_api_key_behavior():
    """Verify behavior when API key is missing or empty."""
    config = AppConfig(groq_api_key=None)

    assert config.has_api_key is False
    with pytest.raises(MissingAPIKeyError) as exc_info:
        config.get_api_key()

    assert "GROQ_API_KEY is not set or empty" in str(exc_info.value)

    # Empty string should also be treated as missing
    config_empty = AppConfig(groq_api_key="   ")
    assert config_empty.has_api_key is False
    with pytest.raises(MissingAPIKeyError):
        config_empty.get_api_key()


def test_api_key_not_exposed_in_string_representations():
    """Verify that sensitive API key credentials are never exposed in logs or representations."""
    secret_key = "super-secret-groq-api-key-987654321"
    config = AppConfig(groq_api_key=secret_key)

    # Direct access works securely
    assert config.get_api_key() == secret_key

    # Check string and representation outputs
    str_repr = str(config)
    repr_repr = repr(config)
    safe_dict = config.to_safe_dict()

    assert secret_key not in str_repr, "Raw API key found in str(config)"
    assert secret_key not in repr_repr, "Raw API key found in repr(config)"
    assert secret_key not in str(safe_dict), "Raw API key found in to_safe_dict()"

    assert "[REDACTED]" in str_repr
    assert "[REDACTED]" in repr_repr
    assert safe_dict["groq_api_key"] == "[REDACTED]"
