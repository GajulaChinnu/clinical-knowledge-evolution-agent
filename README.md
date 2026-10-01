# Clinical Knowledge Evolution Agent (CKEA)

The Clinical Knowledge Evolution Agent (CKEA) is an autonomous, multi-agent clinical decision support pipeline designed to monitor external medical evidence, extract actionable clinical recommendations, compare findings against existing clinical protocols, evaluate organizational and patient impact, and assemble structured change briefs for clinical review. By continuously tracking clinical knowledge evolution, CKEA provides governed, auditable, and deterministic decision intelligence while preserving strict safety boundaries and human oversight.

## Current Phase
Phase 0 — Project Scaffold

## Six Specialized Agents
1. **Monitoring Agent** — Monitors configured source repositories for newly published clinical evidence and guidelines.
2. **Extraction Agent** — Extracts structured clinical recommendations and metadata from detected evidence documents.
3. **Comparison Agent** — Compares extracted recommendations against indexed institutional protocols to identify conflicts, gaps, and confirmations.
4. **Impact Agent** — Calculates deterministic impact scores across clinical urgency, evidence strength, and pathway breadth.
5. **Briefing Agent** — Synthesizes evidence, comparison records, and impact metrics into structured clinical change briefs.
6. **Governance Agent** — Orchestrates human review workflows, tracks review assignments, enforces SLAs, and records final human governance decisions.

## Phase 1 Scope
Phase 1 operates within a strictly contained local environment using synthetic clinical guideline PDFs and synthetic institutional protocol versions. Storage and retrieval rely on local SQLite for relational records and local ChromaDB with `all-MiniLM-L6-v2` embeddings. Live clinical sources, RSS feeds, automatic protocol modifications, and cloud infrastructure are explicitly out of scope for Phase 1.

## Governance & Clinical Safety
Human governance remains strictly responsible for all clinical decisions. The system provides decision support and prepares structured evidence for human review; it must never automatically modify a protocol, approve or reject protocol changes, close change briefs, or infer human governance decisions.
