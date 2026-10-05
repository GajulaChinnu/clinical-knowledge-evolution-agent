# CKEA Phase 12 — Evaluation Framework

## Overview

The Evaluation Framework provides deterministic, reproducible, and offline-capable evaluation of the Clinical Knowledge Evolution Agent (CKEA) pipeline without introducing new clinical reasoning, altering production thresholds, modifying clinical protocols, or bypassing human governance gates (G1..G5).

---

## 1. Evaluation Corpus

The evaluation corpus is strictly synthetic and local to `data/evaluation/`:

- **Extraction Corpus** (`data/evaluation/extraction/`):
  - **20 synthetic documents** (`EXT-01` to `EXT-20`), each provided as valid PDF files.
  - Covers varied extraction conditions: clear recommendations, dosage escalations, monitoring frequency changes, diagnostic criteria shifts, safety warnings, contraindications, population-specific recommendations, missing evidence grades, multiple candidate sections, page provenance tracking, symbol retention, and low-confidence / provenance-failing G1 edge cases.

- **Comparison Scenarios** (`data/evaluation/comparison/`):
  - **16 distinct comparison scenarios** (`CMP-01` to `CMP-16`).
  - Covers exact matches, paraphrased wording variations, dosage changes, monitoring frequency changes, population restrictions, contraindications, workflow requirements, corroborating non-gaps, multi-candidate selection, ambiguous comparisons (G2), contradictory model responses (G2), invalid unparseable model outputs (G2), out-of-domain no-matches (G3), distant low-similarity no-matches (G3), protocol version separation, and near-boundary cases.

- **End-to-End Pipeline Evaluation** (`data/evaluation/end_to_end/`):
  - Uses the **exact same 20-document extraction corpus** to evaluate the complete end-to-end lifecycle: Monitoring -> Extraction -> Comparison -> Impact -> Briefing -> Governance handoff.

---

## 2. Evaluation Suites

The framework implements six specialized evaluation suites:

1. **Extraction Suite (`run_extraction_suite`)**:
   - Evaluates recommendation presence, verbatim text accuracy, population, intervention, recommendation type, evidence grade, provenance integrity, and G1 hold interception.

2. **Comparison Suite (`run_comparison_suite`)**:
   - Evaluates protocol candidate retrieval, comparison result accuracy (`gap`, `no_gap`, `no_match`, `ambiguous`), difference type precision, and G2/G3 gate routing.

3. **Impact Suite (`run_impact_suite`)**:
   - Evaluates deterministic scoring from `config/scoring.yaml`, verifying clinical urgency, evidence strength, pathway breadth, total score, routing tier (`Low`, `Standard`, `High`, `Critical`), SLA hours (`48h`, `168h`, `720h`), and strict rejection of automatic tier downgrades.

4. **Briefing Suite (`run_briefing_suite`)**:
   - Evaluates 7-section structured brief completeness, exact verbatim quote preservation, and deterministic markdown/hash stability.

5. **Governance Safety Suite (`run_governance_safety_suite`)**:
   - Enforces human gates G1..G5. Verifies zero automated approvals, rejections, defers, closures, or protocol mutations.

6. **End-to-End Pipeline Suite (`run_e2e_suite`)**:
   - Validates multi-stage pipeline orchestration across the 20 documents, verifying that human governance boundaries (holding at G4 for review) are strictly preserved.

---

## 3. Evaluation Metrics

- **Extraction**: Total cases, successful extractions, G1 holds, exact-match accuracy, presence accuracy, provenance accuracy, mean confidence, min/max confidence.
- **Comparison**: Total scenarios, matched count, no-match count, G2 holds, G3 holds, difference-type accuracy, precision, recall, F1-score, mean confidence.
- **Impact**: Rule ID preservation, YAML version preservation, SLA accuracy, incomplete score detection, tier immutability enforcement.
- **Briefing**: 7-section completeness rate, verbatim quote preservation rate, deterministic rendering hash stability.
- **Governance**: Safety assertions count, violation detection count (must be 0).
- **End-to-End**: Documents processed, passed to governance, held at human gates, failed count, zero-auto-decisions verified.

---

## 4. Report Locations

Reports are written deterministically to `data/evaluation/reports/`:

- `extraction_report.json` & `extraction_report.md`
- `comparison_report.json` & `comparison_report.md`
- `impact_report.json` & `impact_report.md`
- `briefing_report.json` & `briefing_report.md`
- `governance_safety_report.json` & `governance_safety_report.md`
- `end_to_end_report.json` & `end_to_end_pipeline_report.md`
- `threshold_calibration_report.json` & `threshold_calibration_report.md`
- `evaluation_summary.json` & `evaluation_summary.md`

All JSON reports are deterministically sorted by `case_id` for hash and diff stability.

---

## 5. How to Run Evaluation

### Running Full Evaluation

Execute the evaluation runner across all suites (default is offline, deterministic mode):

```powershell
.\.venv\Scripts\python.exe scripts/run_evaluation.py --suite all
```

Run a specific suite:

```powershell
.\.venv\Scripts\python.exe scripts/run_evaluation.py --suite extraction
.\.venv\Scripts\python.exe scripts/run_evaluation.py --suite comparison
.\.venv\Scripts\python.exe scripts/run_evaluation.py --suite impact
.\.venv\Scripts\python.exe scripts/run_evaluation.py --suite briefing
.\.venv\Scripts\python.exe scripts/run_evaluation.py --suite governance
.\.venv\Scripts\python.exe scripts/run_evaluation.py --suite e2e
.\.venv\Scripts\python.exe scripts/run_evaluation.py --suite calibration
```

### Running the Smoke Test

Execute the standalone smoke test:

```powershell
.\.venv\Scripts\python.exe scripts/smoke_test_evaluation.py
```

### Running Evaluation Unit/Integration Tests

```powershell
.\.venv\Scripts\pytest.exe -v tests/evaluation/test_evaluation_framework.py
```

---

## 6. Offline vs. Live LLM Evaluation

- **Default (Offline Mode)**:
  - Uses deterministic mock responses matching defined evaluation scenarios.
  - Zero network, external API, or live Groq LLM calls.
  - Briefing, impact, governance safety, idempotency, provenance, and reports NEVER invoke an LLM in either mode.

- **Opt-in Live LLM Mode (`--live-llm`)**:
  - Activated only with explicit `--live-llm` CLI flag and valid `GROQ_API_KEY`.
  - Evaluates live Groq model extraction and comparison performance against ground truth.

---

## 7. Threshold Calibration Is Observational Only

The `threshold_calibration_report.json` analyzes cases lying near the provisional `0.70` boundary (e.g. `EXT-16` at 0.65, `EXT-17` at 0.58, `CMP-16` at 0.71).

**Core Safety Principle**:
Threshold calibration findings are **strictly observational and informational**. The evaluation framework **NEVER** automatically alters production thresholds (`EXTRACTION_CONFIDENCE_THRESHOLD=0.70`, `COMPARISON_CONFIDENCE_THRESHOLD=0.70`, `PROTOCOL_SIMILARITY_THRESHOLD=0.70`). All threshold changes require authorized institutional review.
