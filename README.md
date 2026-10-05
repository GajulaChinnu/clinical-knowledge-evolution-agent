# Clinical Knowledge Evolution Agent (CKEA)

The Clinical Knowledge Evolution Agent (CKEA) is an autonomous, multi-agent clinical decision support pipeline designed to monitor external medical evidence, extract actionable clinical recommendations, compare findings against existing clinical protocols, evaluate organizational and patient impact, and assemble structured change briefs for clinical review. By continuously tracking clinical knowledge evolution, CKEA provides governed, auditable, and deterministic decision intelligence while preserving strict safety boundaries and human oversight.

## Current Phase
Phase 13 — Final End-to-End Integration / Demo (Complete)

## Core Safety Rule
> *"System prepares evidence and review material. Authorized clinicians make the final decision."*
> 
> CKEA is a decision-support, evidence-monitoring, and governance-orchestration platform. It never automatically modifies clinical protocols, approves/rejects guidelines, closes reviews, or makes clinical decisions.

## Six Specialized Agents
1. **Monitoring Agent** — Monitors configured source repositories for newly published synthetic clinical evidence and guidelines.
2. **Extraction Agent** — Extracts structured clinical recommendations using Groq (`openai/gpt-oss-20b`) with quotation verification and G1 confidence gate (0.70 threshold).
3. **Comparison Agent** — Compares recommendations against indexed institutional protocols using ChromaDB and Groq to detect specific differences (G2/G3 gates).
4. **Impact Agent** — Calculates deterministic, auditable impact scores from `config/scoring.yaml` across clinical urgency, evidence strength, and pathway breadth.
5. **Briefing Agent** — Compiles structured artifacts into complete 7-section Change Briefs rendered in HTML, Markdown, and JSON companion formats.
6. **Governance Agent** — Manages reviewer assignments, enforces G4 decision gates, executes explicit human decisions, coordinates G5 SLA escalations, and maintains an immutable audit trail.

## Launching the Streamlit Demonstration UI
```powershell
.\.venv\Scripts\python.exe -m streamlit run app/ui/streamlit_app.py
```
Application views:
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
- **Full Test Suite**: `.\.venv\Scripts\pytest.exe -v`

For detailed technical documentation, architecture diagrams, and human gate specifications, see [docs/final_demo.md](docs/final_demo.md).

## Prototype Scope & Disclaimer
All guideline documents and clinical protocols are synthetic. This system is a research prototype demonstration and does not constitute medical advice or production clinical software.
