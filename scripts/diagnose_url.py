"""Diagnose how CKEA would retrieve a URL, without ingesting it or running the pipeline.

Usage:
    python scripts/diagnose_url.py <url> [--no-fetch]

Prints the routing decision, whether Jina Reader was invoked, HTTP/provider status,
content length and the challenge verdict. Never prints API keys and writes nothing to
the database or data/sources/.
"""

import argparse
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.config_service import load_config  # noqa: E402
from app.services.jina_reader_service import JinaReaderError, JinaReaderService  # noqa: E402
from app.services.url_ingestion_service import (  # noqa: E402
    URLIngestionError,
    detect_content_challenge,
    guard_url_host,
    probe_url,
)


def _line(label: str, value) -> None:
    print(f"  {label:<24} {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url")
    parser.add_argument("--no-fetch", action="store_true", help="Probe only; do not call Jina Reader.")
    args = parser.parse_args()

    config = load_config()
    jina = JinaReaderService.from_config(config)

    print(f"CKEA URL diagnosis: {args.url}")
    _line("Jina key configured:", "yes" if jina.is_configured else "NO (set JINA_API_KEY in .env)")
    _line("Jina base URL:", jina.base_url)
    _line("Allow private hosts:", config.allow_private_hosts)

    try:
        host_is_private = guard_url_host(args.url, config.allow_private_hosts)
    except URLIngestionError as e:
        _line("Host guard:", f"BLOCKED - {type(e).__name__}: {e}")
        return 2
    _line("Host guard:", "private (demo mode)" if host_is_private else "public - OK")

    with httpx.Client() as client:
        probe = probe_url(args.url, client, config.url_probe_timeout_seconds, config.allow_private_hosts)
    _line("Probe method:", probe.method)
    _line("Probe HTTP status:", probe.status_code)
    _line("Probe content-type:", probe.content_type or "-")
    _line("Probe error:", probe.error or "-")

    if probe.is_pdf:
        routing = "pdf_direct"
    elif host_is_private:
        routing = "local_direct_html"
    else:
        routing = "jina"
    _line("Routing decision:", routing)

    if routing != "jina":
        _line("Jina invoked:", "no (not required for this route)")
        return 0
    if args.no_fetch:
        _line("Jina invoked:", "no (--no-fetch)")
        return 0
    if not jina.is_configured:
        _line("Jina invoked:", "no - JINA_API_KEY missing; ingestion would fail")
        return 3

    try:
        result = jina.retrieve(args.url)
    except JinaReaderError as e:
        _line("Jina invoked:", "yes")
        _line("Jina result:", f"FAILED - {type(e).__name__}: {e}")
        return 4

    _line("Jina invoked:", "yes")
    _line("Jina HTTP status:", result.status_code)
    _line("Jina attempts:", result.attempts)
    _line("Resolved URL:", result.resolved_url)
    _line("Title:", result.title or "-")
    _line("Content length:", f"{len(result.content)} chars")
    _line("Provider warnings:", "; ".join(result.warnings) or "-")
    challenge = detect_content_challenge(result.content, title=result.title, warnings=result.warnings)
    _line("Challenge verdict:", f"BLOCKED - {challenge}" if challenge else "none - content usable")
    return 5 if challenge else 0


if __name__ == "__main__":
    sys.exit(main())
