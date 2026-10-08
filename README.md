# Clinical Knowledge Evolution Agent (CKEA)

CKEA monitors a controlled set of SYNTHETIC publications, safety notices and clinical guidelines,
identifies meaningful changes, compares them with the organisation's current protocols, and produces
review-ready change briefs, so clinicians and the governance team focus only on changes that affect practice.

## Core Safety Rule
> **The agent prepares. Clinicians decide.**
>
> CKEA is decision support. It never modifies a protocol, never approves/rejects/defers/closes a review,
> and never answers without a verified verbatim excerpt from a stored source.

## How clinicians use it: Treatment Check
A clinician enters the **department** and the **treatment** they plan for a patient (plus optional
de-identified context: age band, eGFR band, pregnancy, comorbidities, current medicines). CKEA checks the
**latest** versions of the relevant guidance, safety notices and publications and answers with one of:

| Verdict | Meaning |
|---|---|
| Matches the latest guidance | The plan matches a current recommendation. |
| Guidance updated: follow the new version | The plan follows a superseded version or a withdrawn recommendation; the answer shows previous vs latest text, versions and dates. |
| Conflicts with current guidance | A contraindication or safety notice applies, or the plan differs from current and earlier guidance. |
| No grounded guidance covers this | Nothing in the monitored sources covers it. CKEA never guesses. |

Every answer cites source, version, publication date and the exact excerpt (verified against the stored
file), shows the hospital protocol's position, and, if the protocol is out of date, the live status of the
protocol-update review routed to a specialist. Patient identifiers are rejected before anything runs; patient
context is never stored.

## Six agents, run for every clinician query
| Step | Agent | Action |
|---|---|---|
| 1 | Monitoring | Find watchlist sources for the department/treatment; ingest newer versions; return version history. |
| 2 | Extraction | Index verbatim statements of the latest and previous versions (treatments, doses, thresholds, contraindications, evidence level). |
| 3 | Comparison | Plan vs latest guidance; previous vs latest version; hospital protocol vs latest guidance. |
| 4 | Impact | Rank findings by urgency, relevance, source quality and novelty (`config/ranking.yaml`); impact tiers use the approved `config/scoring.yaml`. |
| 5 | Briefing | Deterministic verdict and answer brief; 7-section change briefs for governance. |
| 6 | Governance | Route protocol-update briefs to a specialist of the department; record human decisions; G4/G5 unchanged. |

**Background surveillance** runs the same agents without a clinician query: a watchlist check detects new
versions, categorises changes (dose, threshold, contraindication added/removed, safety warning, withdrawn,
new recommendation, new evidence), filters non-practice-changing content (kept and restorable), links
duplicates to the higher-quality source, ranks the feed, and routes high-impact briefs to specialists.

## Configuration and data
| File | Purpose |
|---|---|
| `config/taxonomy.yaml` | Departments, pathways, treatment classes, treatments and synonyms (controlled vocabulary). |
| `config/watchlist.yaml` | The defined watchlist: source type, publisher, quality tier, departments, treatments. |
| `config/ranking.yaml` | Relevance, urgency, source-quality, novelty and filter rules (rule ids + written bases). |
| `config/scoring.yaml` | Approved impact tier rules (unchanged). |
| `config/reviewers.yaml` | Reviewer registry: governance role and specialties. Deny-by-default. |
| `data/corpus/` | SYNTHETIC versioned sources (v1 -> v2 per department, safety notices, publications). |
| `data/protocols/` | SYNTHETIC institutional protocols with department metadata (some deliberately out of date). |

Copy `.env.example` to `.env` (`GROQ_API_KEY`, `JINA_API_KEY`, `SSO_ENABLED`, ...). Database migrations run
automatically on startup (Alembic). Ad-hoc sources can still be added as PDF uploads, PDF URLs or web pages
(web pages via Jina Reader only; private hosts blocked; walled sites fail visibly).

## Run
```powershell
.\.venv\Scripts\python.exe -m streamlit run app/ui/streamlit_app.py
```
Pages: **Treatment Check** (default), Dashboard (priority feed), Watchlist & Sources ("Check all sources now"),
Detected Changes (ranked + Filtered tab), Protocol Comparison, Impact, Change Briefs, Governance, Audit Log,
Evaluation. A sidebar department filter applies to feed, changes, briefs and governance. Deep links:
`?page=treatment-check`, `?page=detected-changes`, ...

## Demo and evaluation
- **Clinician + surveillance demo** (own temporary database, never `data/ckea.db`):
  `.\.venv\Scripts\python.exe scripts/run_clinician_demo.py`
- **Labelled evaluation**: 26 clinician cases (`data/evaluation/clinician_queries.json`) covering every
  verdict and identifier rejection; report in `data/evaluation/reports/clinician_query_report.md`.
- Earlier pipeline demo: `.\.venv\Scripts\python.exe scripts/run_demo.py`
- Tests: `.\.venv\Scripts\pytest.exe` (install extras with `pip install -e ".[dev]"`)

See [docs/final_demo.md](docs/final_demo.md) for the walkthrough.

## Prototype Scope & Disclaimer
All guideline documents, safety notices, publications and protocols are SYNTHETIC. This is a research
prototype; it is not medical advice and not production clinical software.
