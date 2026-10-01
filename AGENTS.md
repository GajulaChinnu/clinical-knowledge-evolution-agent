\# CKEA Engineering Instructions



\## Project



This repository implements the Clinical Knowledge Evolution Agent (CKEA).



The approved CKEA approach document is the source of truth for:

\- architecture

\- agent responsibilities

\- schemas

\- state transitions

\- human gates

\- impact scoring

\- governance

\- evaluation

\- Phase 1 scope



Do not invent requirements that are not supported by the approved design.



\---



\## Six Specialized Agents



CKEA contains exactly six agents:



1\. Monitoring Agent

2\. Extraction Agent

3\. Comparison Agent

4\. Impact Agent

5\. Briefing Agent

6\. Governance Agent



Each agent has one responsibility.



Agents must not silently perform another agent's responsibility.



\---



\## Pipeline



Monitoring

→ Extraction

→ Comparison

→ Impact

→ Briefing

→ Governance



The pipeline uses persisted stage transitions.



Agents exchange structured, versioned records.



Do not pass free-form text directly between agents.



\---



\## Human Gates



Implement:



G1 — Extraction review

G2 — Comparison review

G3 — No-match protocol review

G4 — Mandatory governance decision

G5 — SLA escalation



G4 is unconditional.



Every generated change brief requires an explicit human decision.



The system must never automatically approve, reject, defer, or close a clinical change brief.



\---



\## Clinical Safety



Phase 1 uses synthetic data only.



The system prepares evidence for governance review.



The system must NEVER:



\- automatically modify a protocol

\- automatically approve a protocol change

\- automatically reject a protocol change

\- automatically close a change brief

\- send recommendations directly to clinical staff

\- infer a human governance decision



\---



\## Phase 1 Scope



Use:



\- synthetic PDF documents

\- local configured source folder

\- synthetic protocol versions

\- local SQLite

\- local ChromaDB



Out of scope:



\- RSS ingestion

\- live clinical sources

\- production deployment

\- automatic protocol modification

\- Azure/cloud infrastructure



\---



\## Technology Stack



Use the approved stack:



\- Python

\- Pydantic v2

\- SQLAlchemy

\- SQLite

\- ChromaDB

\- sentence-transformers

\- all-MiniLM-L6-v2

\- pdfplumber

\- spaCy

\- Jinja2

\- Streamlit

\- APScheduler

\- transitions

\- pytest

\- Groq API



\---



\## LLM Usage



LLM provider: Groq API



LLM calls are allowed ONLY for:



1\. recommendation extraction

2\. protocol comparison



Do NOT use an LLM for:



\- PDF parsing

\- section detection

\- file hashing

\- embeddings

\- database operations

\- impact scoring

\- routing

\- SLA calculation

\- audit logging

\- briefing generation

\- governance decisions



\---



\## Token Optimization



Never send an entire PDF to the LLM.



Use:



PDF

→ local parsing

→ section detection

→ relevant section

→ LLM



For protocol comparison:



Recommendation

→ local embedding

→ ChromaDB retrieval

→ small set of candidate protocol sections

→ LLM



Never send the entire protocol repository to the LLM.



Use structured Pydantic outputs.



Use one shared LLM client.



Do not create separate LLM clients inside individual agents.



Avoid unnecessary LLM calls.



Keep stable prompts versioned and reusable.



\---



\## Thresholds



Initial configurable thresholds:



Extraction confidence:

0.70



Comparison confidence:

0.70



Protocol cosine similarity:

0.70



These are provisional routing thresholds, not calibrated probabilities.



\---



\## Persistence



Use SQLite for structured records:



\- ingested\_documents

\- change\_records

\- gap\_records

\- impact\_records

\- change\_briefs

\- review\_assignments

\- audit\_log

\- ingestion\_failures

\- notifications



Use ChromaDB for protocol section embeddings.



Historical protocol versions must remain immutable.



Historical score snapshots must remain immutable.



\---



\## Impact Scoring



Impact scoring is deterministic.



Dimensions:



\- Clinical Urgency

\- Evidence Strength

\- Pathway Breadth



Each dimension is scored 1–5.



Total score is 3–15.



Routing:



12–15 → Critical → 48 hours

8–11 → High → 7 days

4–7 → Standard → 30 days

3 → Low → 90 days



Rules must come from versioned YAML.



Do not interpolate scores.



Do not guess missing or unmapped values.



Incomplete scores remain un-tiered until resolved.



\---



\## Idempotency



Retries must not create duplicate records.



Use a uniqueness strategy based on:



document hash

\+

source version

\+

pipeline version



Unchanged documents must be skipped.



Changed files create new versions.



\---



\## Retry Policy



Transient file/model errors:



\- maximum 3 retries

\- bounded backoff



Invalid or unparseable model output:



\- do not blindly retry indefinitely

\- route to the appropriate human gate



Low-confidence output:



\- route to the appropriate human gate



Conflicting recommendations:



\- retain separately

\- flag for clinical review

\- never silently merge



\---



\## Provenance



Preserve:



\- source identifier

\- source path

\- SHA-256

\- document version

\- page

\- section

\- source excerpt

\- parser version

\- model version

\- prompt version

\- protocol version

\- score rule version



Every comparison must record the exact protocol version used.



\---



\## Git Rules



Development happens on `develop`.



Keep `main` stable.



Use phase-specific commits.



Examples:



phase-00 project instructions

phase-01 dependencies

phase-02 database and schemas

phase-03 monitoring

phase-04 extraction

...



Do not combine unrelated phases into one commit.



Do not modify unrelated files.



Do not rewrite working code without a reason.



\---



\## Phase-Based Development



Only implement the phase explicitly requested.



Do NOT implement future phases.



Do NOT redesign the entire project during a phase.



Before coding:



1\. inspect the repository

2\. read AGENTS.md

3\. inspect the current phase requirements

4\. identify affected files

5\. provide a short implementation plan



Then implement only the requested phase.



After implementation:



1\. run relevant tests

2\. fix failures caused by the current phase

3\. report changed files

4\. report commands executed

5\. report test results

6\. report remaining issues



Then STOP.



Never automatically continue to the next phase.



\---



\## When Uncertain



If a requirement is unclear:



STOP and ask.



Do not invent functionality.



Do not add unnecessary dependencies.



Do not add unnecessary agents.



Do not add unnecessary cloud services.

