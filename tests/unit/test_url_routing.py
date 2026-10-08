"""Routing, host-guard and challenge-detection tests for URL ingestion (Phase 1 hardening)."""

from pathlib import Path

import httpx
import pytest
from sqlalchemy import create_engine

from app.models.database import Base, get_session_factory
from app.models.entities import IngestedDocument, IngestionFailure
from app.services.config_service import AppConfig
from app.services.evaluation_corpus import make_multipage_pdf_bytes
from app.services.jina_reader_service import JinaRateLimitError, JinaReaderService, JinaResult
from app.services.url_ingestion_service import (
    BlockedURLError,
    ContentChallengeError,
    ContentTooLargeError,
    RetrievalProviderError,
    URLIngestionService,
    detect_content_challenge,
    fetch_url_content,
    is_private_host,
)
from app.ui.streamlit_app import handle_source_url_upload

PUBLIC = lambda host: ["93.184.216.34"]

BOT_WALL_HTML = (
    b"<html><head><title>Just a moment...</title></head>"
    b"<body>Cookies must be enabled. Checking your browser.</body></html>"
)

LONG_ARTICLE = (
    "# Type 2 Diabetes Pharmacotherapy Update\n\n"
    + "Adults with type 2 diabetes should initiate metformin 1000 mg daily when eGFR ≥ 45. " * 40
    + "\n\nUse of sulfonylureas is forbidden in patients with recurrent hypoglycaemia. "
    + "Trial sites used a captcha-protected enrolment portal; access denied events were logged."
)


class FakeJina:
    is_configured = True

    def __init__(self, content=LONG_ARTICLE, title="Diabetes Update", warnings=None, error=None):
        self.calls = []
        self.content, self.title, self.warnings, self.error = content, title, warnings or [], error

    def retrieve(self, url):
        self.calls.append(url)
        if self.error:
            raise self.error
        return JinaResult(self.content, self.title, url, 200, 1, list(self.warnings))


class Recorder:
    """httpx MockTransport handler that records requests and serves a fixed response."""

    def __init__(self, status=200, body=b"", content_type="text/html", routes=None):
        self.requests = []
        self.status, self.body, self.content_type = status, body, content_type
        self.routes = routes or {}

    def __call__(self, request):
        self.requests.append(request)
        if str(request.url) in self.routes:
            return self.routes[str(request.url)]
        body = b"" if request.method == "HEAD" else self.body
        return httpx.Response(self.status, content=body, headers={"content-type": self.content_type})


@pytest.fixture
def env(tmp_path: Path):
    db_path = tmp_path / "routing.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(bind=engine)
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    config = AppConfig(
        database_url=f"sqlite:///{db_path}",
        source_dir=source_dir,
        groq_api_key="mock-key",
        jina_api_key="mock-jina-key",
    )
    yield {"config": config, "session_factory": get_session_factory(engine=engine), "source_dir": source_dir}
    engine.dispose()


def _service(config, handler, jina=None, resolver=PUBLIC, **overrides):
    cfg = config.model_copy(update=overrides) if overrides else config
    return URLIngestionService(
        config=cfg,
        jina_service=jina if jina is not None else FakeJina(),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        host_resolver=resolver,
    )


# ------------------------------------------------------------------------------
# Defect 1 regression: walled direct response must not block Jina
# ------------------------------------------------------------------------------

def test_bot_walled_direct_response_still_routes_to_jina(env):
    handler = Recorder(status=403, body=BOT_WALL_HTML)
    jina = FakeJina()
    url = "https://journals.example.org/article/12345/"

    res = _service(env["config"], handler, jina=jina).ingest_url(url, env["source_dir"])

    assert jina.calls == [url]
    assert res.routing_decision == "jina"
    assert res.retrieval_provider == "Jina Reader"
    assert res.probe_error and "403" in res.probe_error


def test_html_page_is_never_fully_fetched_directly(env):
    handler = Recorder(status=200, body=b"<html><body>page</body></html>")
    _service(env["config"], handler).ingest_url("https://site.example.org/guideline", env["source_dir"])

    for request in handler.requests:
        assert request.method == "HEAD" or "range" in request.headers


def test_direct_html_challenge_text_is_never_inspected(env):
    """Direct-HTTP HTML containing challenge phrases must not cause rejection."""
    handler = Recorder(status=200, body=BOT_WALL_HTML)
    res = _service(env["config"], handler).ingest_url("https://site.example.org/x", env["source_dir"])
    assert res.routing_decision == "jina"


def test_failed_probe_with_pdf_path_uses_pdf_route(env):
    pdf = make_multipage_pdf_bytes([["1. Section", "Patients should take metformin."]])
    url = "https://cdn.example.org/files/guideline.pdf"
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] <= 2:  # HEAD + ranged GET fail
            return httpx.Response(405)
        return httpx.Response(200, content=pdf, headers={"content-type": "application/octet-stream"})

    jina = FakeJina()
    res = _service(env["config"], handler, jina=jina).ingest_url(url, env["source_dir"])
    assert res.routing_decision == "pdf_direct"
    assert jina.calls == []


# ------------------------------------------------------------------------------
# Provider configuration and failures
# ------------------------------------------------------------------------------

def test_missing_jina_key_fails_clearly(env):
    service = _service(env["config"], Recorder(), jina=JinaReaderService(api_key=None))
    with pytest.raises(RetrievalProviderError, match="JINA_API_KEY is not configured"):
        service.ingest_url("https://site.example.org/page", env["source_dir"])


def test_jina_failure_is_surfaced_as_provider_error(env):
    jina = FakeJina(error=JinaRateLimitError("Jina API Rate limited (429)."))
    with pytest.raises(RetrievalProviderError, match="429"):
        _service(env["config"], Recorder(), jina=jina).ingest_url("https://site.example.org/p", env["source_dir"])


def test_service_uses_config_key_not_environment(env, monkeypatch):
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    service = URLIngestionService(config=env["config"])
    assert service.jina_service.is_configured


# ------------------------------------------------------------------------------
# Host guard (SSRF)
# ------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8080/demo.html",
        "http://127.0.0.1/x",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/x",
        "http://10.0.0.5/internal",
    ],
)
def test_private_hosts_blocked_by_default(env, url):
    jina = FakeJina()
    with pytest.raises(BlockedURLError):
        _service(env["config"], Recorder(), jina=jina).ingest_url(url, env["source_dir"])
    assert jina.calls == []


def test_hostname_resolving_to_private_address_blocked(env):
    with pytest.raises(BlockedURLError):
        _service(env["config"], Recorder(), resolver=lambda h: ["192.168.1.20"]).ingest_url(
            "https://intranet.example.org/doc", env["source_dir"]
        )


def test_redirect_to_private_host_blocked(env):
    pdf_url = "https://public.example.org/doc.pdf"
    resolver = lambda host: ["10.1.1.1"] if host == "internal.example.org" else ["93.184.216.34"]
    redirect = httpx.Response(302, headers={"location": "https://internal.example.org/secret.pdf"})
    handler = Recorder(routes={pdf_url: redirect})

    with pytest.raises(BlockedURLError):
        _service(env["config"], handler, resolver=resolver).ingest_url(pdf_url, env["source_dir"])


def test_allow_private_hosts_uses_local_direct_route(env):
    html = b"<html><head><title>Demo Guideline</title></head><body><h1>1. Recommendations</h1><p>Adults should take metformin.</p></body></html>"
    jina = FakeJina()
    res = _service(env["config"], Recorder(body=html), jina=jina, allow_private_hosts=True).ingest_url(
        "http://localhost:8080/demo_clinical_source.html", env["source_dir"]
    )
    assert res.routing_decision == "local_direct_html"
    assert jina.calls == []


def test_is_private_host_unresolvable_raises():
    from app.services.url_ingestion_service import HTTPFetchError

    def failing(host):
        raise OSError("nodename nor servname provided")

    with pytest.raises(HTTPFetchError, match="Could not resolve"):
        is_private_host("no-such-host.invalid", failing)


def test_download_size_cap_enforced():
    big = b"%PDF-1.4\n" + b"0" * 5000
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=big)))
    with pytest.raises(ContentTooLargeError):
        fetch_url_content("https://site.example.org/big.pdf", max_bytes=1000, client=client, resolver=PUBLIC)


# ------------------------------------------------------------------------------
# Challenge detection heuristics
# ------------------------------------------------------------------------------

def test_long_clinical_article_with_trigger_words_is_not_a_challenge():
    assert detect_content_challenge(LONG_ARTICLE, title="Diabetes Update") is None


def test_challenge_title_detected_even_for_long_body():
    assert detect_content_challenge(LONG_ARTICLE, title="Just a moment...") is not None


def test_short_body_with_provider_block_warning_is_challenge():
    assert detect_content_challenge("Sign in", title="", warnings=["Target URL returned error 403: Forbidden"]) is not None


def test_long_body_with_provider_warning_is_accepted_and_warning_recorded(env):
    jina = FakeJina(warnings=["Target URL returned error 403: Forbidden"])
    res = _service(env["config"], Recorder(), jina=jina).ingest_url("https://site.example.org/a", env["source_dir"])
    assert res.provider_warnings == ["Target URL returned error 403: Forbidden"]


def test_unchanged_jina_content_produces_identical_hash(env):
    service = _service(env["config"], Recorder())
    first = service.ingest_url("https://site.example.org/a", env["source_dir"])
    second = service.ingest_url("https://site.example.org/a", env["source_dir"])
    assert first.content_sha256 == second.content_sha256


# ------------------------------------------------------------------------------
# Provenance + failure recording through the UI handler
# ------------------------------------------------------------------------------

def test_retrieval_provenance_persisted_in_doc_metadata(env):
    from unittest.mock import MagicMock
    from app.orchestration.pipeline import ClinicalKnowledgePipeline
    from app.schemas.orchestration import PipelineResult, PipelineStage, PipelineStatus

    pipeline = MagicMock(spec=ClinicalKnowledgePipeline)
    pipeline.process_document.return_value = PipelineResult(
        document_id="d", status=PipelineStatus.COMPLETED, current_stage=PipelineStage.EXTRACTION,
    )
    service = _service(env["config"], Recorder())
    handle_source_url_upload(
        url="https://site.example.org/guideline",
        session_factory=env["session_factory"],
        config=env["config"],
        url_service=service,
        pipeline=pipeline,
    )
    with env["session_factory"]() as session:
        doc = session.query(IngestedDocument).one()
        assert doc.doc_metadata["retrieval_provider"] == "Jina Reader"
        assert doc.doc_metadata["routing_decision"] == "jina"
        assert doc.doc_metadata["provider_attempts"] == 1


def test_challenge_records_ingestion_failure(env):
    jina = FakeJina(content="Please verify you are human.", title="")
    res = handle_source_url_upload(
        url="https://walled.example.org/article",
        session_factory=env["session_factory"],
        config=env["config"],
        url_service=_service(env["config"], Recorder(), jina=jina),
    )
    assert res["pipeline_status"] == "Blocked"
    with env["session_factory"]() as session:
        failure = session.query(IngestionFailure).one()
        assert failure.error_category == "ContentChallengeError"
        assert failure.source_path == "https://walled.example.org/article"


def test_blocked_url_records_ingestion_failure_and_raises(env):
    with pytest.raises(BlockedURLError):
        handle_source_url_upload(
            url="http://127.0.0.1/admin",
            session_factory=env["session_factory"],
            config=env["config"],
            url_service=_service(env["config"], Recorder()),
        )
    with env["session_factory"]() as session:
        assert session.query(IngestionFailure).one().error_category == "BlockedURLError"


def test_no_site_specific_code_in_ingestion_layer():
    import inspect
    import app.services.jina_reader_service as jina_mod
    import app.services.url_ingestion_service as url_mod

    for mod in (jina_mod, url_mod):
        src = inspect.getsource(mod).lower()
        for name in ("pubmed", "ncbi", "nih.gov", "diabetes.org", "w3.org"):
            assert name not in src
