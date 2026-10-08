# Clinical Knowledge Evolution Agent (CKEA)

The Clinical Knowledge Evolution Agent (CKEA) is an AI-assisted, multi-agent clinical evidence surveillance and governance pipeline designed to monitor external medical evidence, extract actionable clinical recommendations, compare findings against existing clinical protocols, evaluate organizational and patient impact, and assemble structured change briefs for clinical review. By continuously tracking clinical knowledge evolution, CKEA provides governed, auditable, and deterministic decision intelligence while preserving strict safety boundaries and human oversight.

## Core Safety Rule
> **The agent prepares. Clinicians decide.**
> 
> CKEA is a decision-support, evidence-monitoring, and governance-orchestration platform. It never automatically modifies clinical protocols, approves/rejects guidelines, closes reviews, or makes clinical decisions.

## Six Specialized Agents
1. **Monitoring Agent** — Versions every source against a stable identity; a new version is created only when the normalized content changes.
2. **Extraction Agent** — Extracts structured clinical recommendations using Groq (`openai/gpt-oss-20b`) with quotation verification and G1 confidence gate (0.70 threshold).
3. **Comparison Agent** — Compares recommendations against indexed institutional protocols using ChromaDB and Groq to detect specific differences (G2/G3 gates).
4. **Impact Agent** — Calculates deterministic, auditable impact scores from `config/scoring.yaml` across clinical urgency, evidence strength, and pathway breadth.
5. **Briefing Agent** — Compiles structured artifacts into complete 7-section Change Briefs rendered in HTML, Markdown, and JSON companion formats.
6. **Governance Agent** — Manages reviewer assignments, enforces G4 decision gates, executes explicit human decisions, coordinates G5 SLA escalations, and maintains an immutable audit trail.

## Source Inputs
All three entry points produce the same normalized source and share one downstream path:

| Input | Retrieval | Stored as | Identity |
|---|---|---|---|
| PDF upload | local | `.pdf` | new source (content-derived) or "new version of" an existing source — never the filename |
| URL to a PDF | direct download (size-capped) | `.pdf` | hash of the canonical URL |
| URL to a web page | **Jina Reader only** | UTF-8 Markdown (`.md`) | hash of the canonical URL |

- Web pages are never fetched directly; sites that require login, CAPTCHA or bot checks fail visibly as an ingestion failure and are never bypassed. There is no site-specific code.
- Private, loopback and link-local hosts are blocked (SSRF guard). `ALLOW_PRIVATE_HOSTS=true` exists only for the local demo server.
- Versioning uses the SHA-256 of the canonical normalized text; the raw-file SHA-256 is kept for integrity. Scanned (text-less) PDFs are rejected as `unreadable_needs_ocr`.
- Diagnose how any URL would be routed: `.\.venv\Scripts\python.exe scripts/diagnose_url.py <url>` (never prints keys).

## Configuration
Copy `.env.example` to `.env`. Key settings: `GROQ_API_KEY`, `JINA_API_KEY`, `LLM_MAX_RETRIES`, `LLM_TIMEOUT_SECONDS`, `MAX_SOURCE_CHARS`, `SSO_ENABLED`, `REVIEWER_REGISTRY_PATH`.

- **Reviewers**: only identities listed in `config/reviewers.yaml` can assign, review or decide. With `SSO_ENABLED=true` the acting reviewer comes from the Streamlit OIDC sign-in email; without it the governance form runs in a clearly flagged, unauthenticated demo mode.
- **Database migrations**: applied automatically on startup (Alembic, `app/models/migrations`). CLI: `alembic upgrade head`.

## Launching the Streamlit Demonstration UI
```powershell
.\.venv\Scripts\python.exe -m streamlit run app/ui/streamlit_app.py
```
Pages support deep links, e.g. `http://127.0.0.1:8501/?page=governance`. Application views:
- **Dashboard**: Aggregated operational counts across documents, gates, briefs, and SLAs.
- **Source Documents**: Read-only inspection of synthetic evidence documents.
- **Changes & Comparison**: Recommendation extraction, protocol matching, and explicit no-match display.
- **Impact Assessment**: Multidimensional scoring, tier routing, and written bases.
- **Change Briefs**: 7-section clinical briefs in HTML, Markdown, and JSON companion formats.
- **Governance / Human Review**: Explicit clinician actions (`Start Review`, `Approve`, `Reject`, `Defer`, `Close`).
- **Audit History**: Immutable chronological audit log of all system and governance events.
- **Evaluation Reports**: Inspector for Phase 12 evaluation metrics.

## Running Demonstration and Test Scripts
- **Full E2E Demonstration**: `.\.venv\Scripts\python.exe scripts/run_demo.py`
- **Demo Smoke Test**: `.\.venv\Scripts\python.exe scripts/smoke_test_final_demo.py`
- **Reset Demo Data**: `.\.venv\Scripts\python.exe scripts/reset_demo_data.py`
- **Phase 12 Evaluation**: `.\.venv\Scripts\python.exe scripts/run_evaluation.py`
- **Integration Test Suite**: `.\.venv\Scripts\pytest.exe -v tests/integration/test_final_demo.py`
- **Full Test Suite**: `.\.venv\Scripts\pytest.exe -v` (install test extras with `pip install -e ".[dev]"`)

For detailed technical documentation, architecture diagrams, and human gate specifications, see [docs/final_demo.md](docs/final_demo.md).

## Prototype Scope & Disclaimer
All guideline documents and clinical protocols are synthetic. This system is a research prototype demonstration and does not constitute medical advice or production clinical software.
