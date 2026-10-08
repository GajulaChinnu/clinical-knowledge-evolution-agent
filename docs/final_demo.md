# Clinical Knowledge Evolution Agent (CKEA)
## Phase 13 — Final End-to-End Integration & Demonstration

> **CRITICAL CLINICAL SAFETY PRINCIPLE**  
> *"System prepares evidence and review material. Authorized clinicians make the final decision."*  
> The CKEA platform is a decision-support, evidence-monitoring, and governance-orchestration system. It is **NOT** a clinical decision-making tool and never automatically modifies protocols, approves guidelines, or closes clinical reviews.

---

### 1. System Architecture & Overview

The Clinical Knowledge Evolution Agent (CKEA) continuously evaluates synthetic external clinical guidelines against internal institutional clinical protocols. It extracts actionable recommendations, performs semantic and categorical comparison, computes auditable multidimensional impact scores, generates comprehensive 7-section change briefs, routes briefs to clinical governance committees, and enforces Service Level Agreements (SLAs).

```
                      +-----------------------------+
                      |   Synthetic Clinical PDF    |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |      Monitoring Agent       |  SHA-256 Hashing, Idempotent Discovery
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |      Extraction Agent       |  Groq (openai/gpt-oss-20b), Quotation Check
                      +-----------------------------+
                                     |
                          [G1 Confidence Gate] -----> (Hold for Human Review if < 0.70)
                                     |
                                     v
                      +-----------------------------+
                      |      Comparison Agent       |  ChromaDB MiniLM Embeddings + Groq LLM
                      +-----------------------------+
                                     |
                         [G2 / G3 Matching Gates] --> (G2 Hold if Ambiguous; G3 Committee Review if No-Match)
                                     |
                                     v
                      +-----------------------------+
                      |        Impact Agent         |  Deterministic scoring.yaml (Urgency, Evidence, Breadth)
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |       Briefing Agent        |  Jinja2 7-Section HTML/Markdown/JSON Generation
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |      Governance Agent       |  Explicit Human Action & Authorization (G4)
                      +-----------------------------+
                                     |
         +---------------------------+---------------------------+
         |                                                       |
         v                                                       v
+-------------------------------+               +-------------------------------+
|  Periodic G5 SLA Scheduler    |               |  Authorized Clinical Reviewer |
|  - Idempotent Overdue Check   |               |  - Start Review               |
|  - Escalation Notification    |               |  - Explicit Approve           |
|  - Decision remains UNSET     |               |  - Explicit Reject            |
+-------------------------------+               |  - Explicit Defer (Future Date)|
                                                |  - Explicit Close             |
                                                +-------------------------------+
                                                                 |
                                                                 v
                                                +-------------------------------+
                                                |   Append-Only Audit Log DB    |
                                                |  (Protocols Stay Immutable)   |
                                                +-------------------------------+
```

---

### 2. Six Specialized Agents

1. **Monitoring Agent (`MonitoringAgent`)**  
   Discovers new synthetic guideline PDFs, computes canonical SHA-256 hashes, detects version updates, skips unchanged duplicates, and writes immutable provenance records.
2. **Extraction Agent (`ExtractionAgent`)**  
   Performs two-pass extraction on candidate PDF sections using Groq (`openai/gpt-oss-20b`). Verifies verbatim text quotations against page text and enforces the production confidence threshold (0.70).
3. **Comparison Agent (`ComparisonAgent`)**  
   Retrieves candidate institutional protocol sections from ChromaDB (`all-MiniLM-L6-v2`), classifies gaps (`dosage_change`, `contraindication_change`, `population_change`, `new_recommendation`, `minor_wording_change`), validates protocol quotation fidelity, and identifies no-matches.
4. **Impact Agent (`ImpactAgent`)**  
   Applies deterministic, auditable rules from `config/scoring.yaml` across three dimensions: **Clinical Urgency** (1–5), **Evidence Strength** (1–5), and **Pathway Breadth** (1–5). Computes total impact score (3–15) and resolves routing tier (`Critical`, `High`, `Standard`, `Low`).
5. **Briefing Agent (`BriefingAgent`)**  
   Compiles all upstream artifacts into an auditable 7-section Change Brief. Generates validated HTML, Markdown, and JSON companion formats with content SHA-256 verification.
6. **Governance Agent (`GovernanceAgent`)**  
   Enforces human review authorization, tracks assignments, records explicit decisions (Approve, Reject, Defer), enforces mandatory rationales and future deferral dates, blocks unauthorized access, and maintains an append-only audit log.

---

### 3. Human Gates (G1 through G5)

| Gate | Name | Trigger | Behavior | Human Action Required |
|---|---|---|---|---|
| **G1** | **Extraction Quality Gate** | Extraction confidence < 0.70 or quotation mismatch | Halts pipeline; document marked `HELD` | Clinician reviews verbatim text, edits if necessary, and explicitly proceeds or rejects |
| **G2** | **Comparison Quality Gate** | Comparison confidence < 0.70 or contradictory classification | Halts pipeline; gap marked `review_required` | Clinician verifies protocol alignment and confirms or resolves gap |
| **G3** | **No-Match Committee Gate** | No protocol section reaches similarity threshold (0.50) | Flags gap as `no_match`; skips comparison LLM call | Governance committee reviews whether to create a new protocol or dismiss |
| **G4** | **Authorized Governance Gate** | Brief reached; ready for human decision | Requires authorized reviewer and mandatory clinical rationale | Clinician executes `approve`, `reject`, or `defer` (with future date); only decided briefs can be closed |
| **G5** | **SLA Escalation Gate** | Review overdue relative to tier deadline | Dispatches escalation notification and logs audit event | **SLA escalation NEVER makes a decision.** Decision remains unset until a human acts |

---

### 4. How to Launch and Use the Streamlit Application

#### Launch Command
Run the Streamlit application using the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/ui/streamlit_app.py
```
Or:
```powershell
.\.venv\Scripts\streamlit.exe run app/ui/streamlit_app.py
```

The application opens at `http://localhost:8501`.

#### Streamlit Application Views
- **Dashboard**: High-level aggregated operational metrics (total documents, processed count, documents held at G1/G2/G3, briefs awaiting governance, deferred briefs, overdue reviews, and closed briefs). Informational only; does not invent KPIs.
- **Source Documents**: Read-only inspection of synthetic evidence documents (identifier, version, SHA-256, ingestion status, error details). Editing is strictly disabled.
- **Changes & Comparison**: Displays verbatim recommendation text, population, intervention, evidence grade, matched protocol ID, protocol version, section heading, similarity score, difference category, and G2/G3 hold status. For unindexed guidelines, explicitly displays: *"No corresponding institutional protocol section was found."*
- **Impact Assessment**: Displays multidimensional scores (Urgency, Evidence, Breadth, Total), assigned routing tier, SLA deadline, audit rule IDs, scoring YAML version, and written basis statements.
- **Change Briefs**: Full review of generated Change Briefs with all seven required sections:
  1. What changed (Verbatim recommendation & source metadata)
  2. Current protocol (Institutional protocol quote or explicit no-match)
  3. Specific difference (Exact clinical delta and difference category)
  4. Impact assessment (Scores, tier, and written rationales)
  5. Affected workflows (Impacted roles, EHR order sets, care pathways)
  6. Proposed review actions (Confirm applicability, Update protocol, Record rationale)
  7. Source excerpt (Verbatim context from guideline document)
  *Allows switching between rendered HTML, Markdown, and raw JSON companion data.*
- **Governance & Human Review**: The core decision interface. Allows authorized reviewers (`dr_smith`, `dr_jones`, `dr_patel`, `governance_chair`) to:
  - **Start Review** (Transitions brief from `assigned` to `in_review`)
  - **Approve** (Mandates non-empty clinical justification)
  - **Reject** (Mandates non-empty clinical justification)
  - **Defer** (Mandates non-empty justification AND a strictly future follow-up date)
  - **Close After Decision** (Enforces G4: brief must be in `decided` status)
  *No automatic decision buttons exist. No decision is pre-selected.*
- **Audit History**: Chronological, immutable view of all governance and pipeline events (timestamp, actor, action, previous status, new status, rationale, and metadata). Edit and delete operations are strictly prohibited.
- **Evaluation Reports**: Direct viewer for the Phase 12 machine-readable evaluation reports.

---

### 5. Deterministic Demonstration Scripts

Three standalone, reproducible scripts are provided in `scripts/`:

#### A. Full End-to-End Workflow Demonstration (`scripts/run_demo.py`)
Executes the complete three-step demonstration:
1. **Happy-Path Lifecycle**: Ingests synthetic document -> extracts recommendation -> matches protocol -> scores impact -> renders 7-section brief -> assigns reviewer -> starts review -> records explicit approval -> closes brief -> displays audit trail.
2. **Safety Gate G1 Demonstration**: Ingests low-confidence document -> triggers G1 hold -> halts pipeline -> simulates human clinician resolution with clinical rationale -> resumes pipeline downstream.
3. **G5 SLA Escalation Demonstration**: Creates an overdue assignment -> runs single-cycle SLA evaluation -> verifies escalation notification and audit event -> asserts decision remains unset (`None`) -> records explicit human decision post-escalation.
4. **Protocol Immutability Check**: Verifies that the authoritative protocol definition `data/protocols/PROT-DM-001_v1.0.json` (SHA-256: `4203b13ea09d18285ceb34b136841973c7e26b6b59a3943da3a7801d7416289a`) and ChromaDB vector index records remain byte-for-byte identical and unmodified.

Run with:
```powershell
.\.venv\Scripts\python.exe scripts/run_demo.py
```

#### B. Standalone Smoke Test (`scripts/smoke_test_final_demo.py`)
Performs 12 fast, deterministic, non-destructive assertions validating:
1. Protocol snapshot baseline
2. Isolated database initialization
3. Sample record population
4. Dashboard metric calculation
5. Data inspection queries
6. 7-section ChangeBrief verification
7. Start Review transition
8. Decision rationale enforcement
9. Deferral future date enforcement
10. Explicit human decision and closure
11. G4 non-bypassability enforcement
12. Protocol byte immutability

Run with:
```powershell
.\.venv\Scripts\python.exe scripts/smoke_test_final_demo.py
```

#### C. Demo Data Reset (`scripts/reset_demo_data.py`)
Resets the SQLite database (`data/ckea.db`) to a clean state while preserving all protocol definitions in `data/protocols/` and the ChromaDB vector index.

Run with:
```powershell
.\.venv\Scripts\python.exe scripts/reset_demo_data.py
```

---

### 6. Phase 12 Evaluation Framework

The system includes a 24-test evaluation suite validating precision, recall, safety gate triggers, idempotency, and SLA compliance across synthetic evaluation corpora.

To run the full evaluation suite and regenerate evaluation reports:
```powershell
.\.venv\Scripts\python.exe scripts/run_evaluation.py
```

Generated reports are located in `data/evaluation/reports/`:
- `evaluation_summary.json`
- `extraction_report.json`
- `comparison_report.json`
- `impact_report.json`
- `briefing_report.json`
- `governance_safety_report.json`
- `end_to_end_report.json`
- `threshold_calibration_report.json`

---

### 7. LLM & External Service Boundaries

- **LLM Provider**: Groq API (`openai/gpt-oss-20b`).
- **LLM Call Boundary**: Strictly restricted to `ExtractionAgent` and `ComparisonAgent`.
- **Zero LLM Usage**: Monitoring, Impact scoring, Briefing rendering, Governance decisions, SLA scheduling, Streamlit UI, Audit trail, and Evaluation reporting make **zero LLM calls**.
- **External Dependencies**: Zero external network dependencies (no RSS feeds, no live PubMed scraping, no email/SMS dispatch). Operates completely locally using synthetic files and SQLite.
- **Clinician Treatment Check**: The verdict, comparison, ranking and routing are fully deterministic. Query mode makes **zero LLM calls**.

---

### 8. Clinician Treatment Check & Watchlist Surveillance

**Run:** `.\.venv\Scripts\python.exe scripts/run_clinician_demo.py`. It uses its own temporary database and never touches `data/ckea.db`.

1. **Surveillance.** The watchlist (`config/watchlist.yaml`) is checked while only v1 is published. v2 is then "released", and the Monitoring Agent ingests the new versions. Changes are categorised (for example dose change, threshold change, contraindication added, withdrawn) and ranked with `config/ranking.yaml`. Non-practice-changing items are filtered but kept, and can be restored.
2. **Treatment Check.** The demo runs one query per verdict:
   - `metformin 500 mg twice daily` (eGFR >=60) → **Matches the latest guidance**
   - `metformin 500 mg once daily` → **Guidance updated: follow the new version**. It shows v1.0 (2025-01-15) "once daily" against v2.0 (2026-02-01) "twice daily", and PROT-DM-001 is out of date.
   - `levofloxacin` for low-severity CAP → **Conflicts with current guidance** (safety notice)
   - `unicorn extract` → **No grounded guidance covers this**
3. **Governance.** The PROT-DM-001 update brief is routed to the Endocrinology specialist (`dr_wilson`). Their explicit approval is then visible to the next clinician who asks. CKEA never edits the protocol.
4. **Evaluation.** 26 labelled cases (`data/evaluation/clinician_queries.json`) are run, and a report is written to `data/evaluation/reports/clinician_query_report.md`.

Expected result: 26/26 cases pass, 100% of citations are verified, and 0 answers lack a verified citation.

**Every answer contains:** the verdict and its basis; each source checked with its latest version and date; what changed (previous vs latest excerpts); the plan components assessed; the hospital protocol's position; the governance status; and the notice "Decision support only. The treating clinician decides."

---

### 9. Scope & Disclaimers

- This software is a **prototype demonstration** designed for controlled research and evaluation.
- All guideline documents (`data/sources/`) and protocol definitions (`data/protocols/`) are **synthetic**.
- No production clinical validation or medical accreditation is claimed.
- Final clinical decisions must always be made by licensed healthcare professionals.
