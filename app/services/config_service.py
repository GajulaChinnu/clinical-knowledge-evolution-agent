"""CKEA Application Configuration Service.

Provides typed access to configuration parameters, environment variable loading,
and secure handling of sensitive credentials.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
import os
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConfigError(Exception):
    """Base exception for configuration errors."""
    pass


class MissingAPIKeyError(ConfigError):
    """Raised when GROQ_API_KEY is required but not configured."""
    pass


class AppConfig(BaseModel):
    """Typed CKEA Application Configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    # API Configuration (Groq)
    groq_api_key: Optional[str] = Field(default=None, repr=False)
    groq_model: str = Field(default="openai/gpt-oss-20b")
    groq_base_url: str = Field(default="https://api.groq.com/openai/v1")

    # Web Retrieval (Jina Reader + direct PDF download)
    jina_api_key: Optional[str] = Field(default=None, repr=False)
    jina_reader_base_url: str = Field(default="https://r.jina.ai")
    jina_timeout_seconds: float = Field(default=45.0, gt=0)
    url_probe_timeout_seconds: float = Field(default=10.0, gt=0)
    url_download_timeout_seconds: float = Field(default=30.0, gt=0)
    max_download_bytes: int = Field(default=50 * 1024 * 1024, gt=0)
    allow_private_hosts: bool = Field(default=False)

    # Storage & Database
    database_url: str = Field(default="sqlite:///./data/ckea.db")
    source_dir: Path = Field(default_factory=lambda: Path("./data/sources"))
    protocol_dir: Path = Field(default_factory=lambda: Path("./data/protocols"))
    chroma_dir: Path = Field(default_factory=lambda: Path("./data/chroma"))
    output_dir: Path = Field(default_factory=lambda: Path("./data/output"))
    scoring_rules_path: Path = Field(default_factory=lambda: Path("./config/scoring.yaml"))
    templates_dir: Path = Field(default_factory=lambda: Path("./templates"))

    # Confidence & Routing Thresholds (from AGENTS.md)
    extraction_confidence_threshold: float = Field(default=0.70)
    comparison_confidence_threshold: float = Field(default=0.70)
    protocol_similarity_threshold: float = Field(default=0.70)

    # SLA Scheduler Configuration (Phase 11)
    sla_scheduler_enabled: bool = Field(default=False)
    sla_check_interval_minutes: int = Field(default=60, ge=1)
    sla_timezone: str = Field(default="UTC")

    @field_validator("source_dir", "protocol_dir", "chroma_dir", "output_dir", "scoring_rules_path", "templates_dir", mode="before")
    @classmethod
    def _coerce_path(cls, v: Any) -> Path:
        if isinstance(v, Path):
            return v
        return Path(str(v))

    @property
    def has_api_key(self) -> bool:
        """Check whether a non-empty API key is configured."""
        return bool(self.groq_api_key and self.groq_api_key.strip())

    def get_api_key(self) -> str:
        """Return the API key or raise MissingAPIKeyError without leaking details."""
        if not self.has_api_key:
            raise MissingAPIKeyError("GROQ_API_KEY is not set or empty. Please configure it in .env or environment.")
        return self.groq_api_key  # type: ignore

    @property
    def has_jina_api_key(self) -> bool:
        """Check whether a non-empty Jina Reader API key is configured."""
        return bool(self.jina_api_key and self.jina_api_key.strip())

    def __repr__(self) -> str:
        fields = []
        for k, v in self.__dict__.items():
            if k in _SECRET_FIELDS:
                val = "'[REDACTED]'" if v else "None"
            else:
                val = repr(v)
            fields.append(f"{k}={val}")
        return f"{self.__class__.__name__}({', '.join(fields)})"

    def __str__(self) -> str:
        return self.__repr__()

    def to_safe_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation with sensitive credentials masked."""
        data = self.model_dump()
        for key in _SECRET_FIELDS:
            data[key] = "[REDACTED]" if data.get(key) else None
        return data


_SECRET_FIELDS = ("groq_api_key", "jina_api_key")


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def _optional_secret(value: str) -> Optional[str]:
    return value.strip() if value and value.strip() else None


_ENV_MAPPINGS: Dict[str, Tuple[str, Callable[[str], Any]]] = {
    "GROQ_API_KEY": ("groq_api_key", _optional_secret),
    "GROQ_MODEL": ("groq_model", str),
    "GROQ_BASE_URL": ("groq_base_url", str),
    "JINA_API_KEY": ("jina_api_key", _optional_secret),
    "JINA_READER_BASE_URL": ("jina_reader_base_url", str),
    "JINA_TIMEOUT_SECONDS": ("jina_timeout_seconds", float),
    "URL_PROBE_TIMEOUT_SECONDS": ("url_probe_timeout_seconds", float),
    "URL_DOWNLOAD_TIMEOUT_SECONDS": ("url_download_timeout_seconds", float),
    "MAX_DOWNLOAD_BYTES": ("max_download_bytes", int),
    "ALLOW_PRIVATE_HOSTS": ("allow_private_hosts", _parse_bool),
    "DATABASE_URL": ("database_url", str),
    "SOURCE_DIR": ("source_dir", Path),
    "PROTOCOL_DIR": ("protocol_dir", Path),
    "CHROMA_DIR": ("chroma_dir", Path),
    "OUTPUT_DIR": ("output_dir", Path),
    "SCORING_RULES_PATH": ("scoring_rules_path", Path),
    "TEMPLATES_DIR": ("templates_dir", Path),
    "EXTRACTION_CONFIDENCE_THRESHOLD": ("extraction_confidence_threshold", float),
    "COMPARISON_CONFIDENCE_THRESHOLD": ("comparison_confidence_threshold", float),
    "PROTOCOL_SIMILARITY_THRESHOLD": ("protocol_similarity_threshold", float),
    "SLA_SCHEDULER_ENABLED": ("sla_scheduler_enabled", _parse_bool),
    "SLA_CHECK_INTERVAL_MINUTES": ("sla_check_interval_minutes", int),
    "SLA_TIMEZONE": ("sla_timezone", str),
}


def load_config(
    env_file: Optional[str | Path] = None,
    load_os_environ: bool = True,
    **overrides: Any,
) -> AppConfig:
    """Load configuration from .env file, environment variables, and explicit overrides.

    Precedence order:
    1. Explicit overrides (**overrides)
    2. OS Environment variables (if load_os_environ=True)
    3. Values from env_file (or default .env if present)
    4. Default AppConfig field values

    Args:
        env_file: Optional path to a .env file.
        load_os_environ: Whether to inspect os.environ for overrides.
        **overrides: Explicit values overriding all other sources.

    Returns:
        AppConfig instance with resolved and validated fields.
    """
    raw_values: Dict[str, Any] = {}

    # 1. Parse .env file if available
    target_env = Path(env_file) if env_file else Path(".env")
    if target_env.exists():
        file_values = dotenv_values(target_env)
        for env_key, str_val in file_values.items():
            if str_val is not None and env_key in _ENV_MAPPINGS:
                config_key, converter = _ENV_MAPPINGS[env_key]
                try:
                    raw_values[config_key] = converter(str_val)
                except (ValueError, TypeError):
                    pass

    # 2. Inspect os.environ
    if load_os_environ:
        for env_key, (config_key, converter) in _ENV_MAPPINGS.items():
            if env_key in os.environ:
                val = os.environ[env_key]
                try:
                    raw_values[config_key] = converter(val)
                except (ValueError, TypeError):
                    pass

    # 3. Apply overrides
    raw_values.update(overrides)

    return AppConfig(**raw_values)
