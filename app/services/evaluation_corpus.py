"""Synthetic evaluation corpus definitions and fixture generator for CKEA (Phase 12).

Provides:
- 20 synthetic extraction documents covering varied conditions.
- 16 comparison scenarios covering matched, gaps, ambiguity, no-match, and boundary cases.
- Generation of local synthetic PDFs into data/evaluation/extraction/ and data/evaluation/end_to_end/.
- Serialization of expected ground-truth metadata into data/evaluation/expected/.
"""

import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from app.schemas.changes import ChangeStatus
from app.schemas.comparison import ComparisonResponse
from app.schemas.extraction import ExtractedRecommendation, ExtractionResponse
from app.schemas.gaps import ComparisonResult, DifferenceType, GapStatus
from app.schemas.orchestration import HumanGate, PipelineStage

logger = logging.getLogger("ckea.services.evaluation_corpus")


def make_multipage_pdf_bytes(pages: List[List[str]]) -> bytes:
    """Generate valid multi-page PDF-1.4 binary bytes from lists of string lines per page."""
    count = len(pages)
    if count == 0:
        raise ValueError("Cannot create PDF with 0 pages.")

    page_ids = [3 + 2 * i for i in range(count)]
    stream_ids = [4 + 2 * i for i in range(count)]
    font_id = 3 + 2 * count
    kids_str = " ".join(f"{pid} 0 R" for pid in page_ids)

    obj_defs: Dict[int, Any] = {
        1: "<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{kids_str}] /Count {count} >>",
    }

    for i in range(count):
        pid = page_ids[i]
        sid = stream_ids[i]
        lines = pages[i]
        stream_lines = ["BT", "/F1 12 Tf", "14 TL", "50 750 Td"]
        for line in lines:
            clean = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream_lines.append(f"({clean}) '")
        stream_lines.append("ET")
        content = "\n".join(stream_lines).encode("latin1", "replace")
        obj_defs[pid] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {sid} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>"
        )
        obj_defs[sid] = (f"<< /Length {len(content)} >> stream\n", content, "\nendstream")

    obj_defs[font_id] = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    total_objs = font_id
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets: Dict[int, int] = {0: 0}
    for oid in range(1, total_objs + 1):
        offsets[oid] = out.tell()
        out.write(f"{oid} 0 obj\n".encode("latin1"))
        val = obj_defs[oid]
        if isinstance(val, tuple):
            out.write(val[0].encode("latin1"))
            out.write(val[1])
            out.write(val[2].encode("latin1"))
        else:
            out.write(val.encode("latin1"))
        out.write(b"\nendobj\n")

    xref_pos = out.tell()
    out.write(f"xref\n0 {total_objs + 1}\n".encode("latin1"))
    out.write(b"0000000000 65535 f \n")
    for oid in range(1, total_objs + 1):
        out.write(f"{offsets[oid]:010d} 00000 n \n".encode("latin1"))
    out.write(
        f"trailer\n<< /Size {total_objs + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode("latin1")
    )
    return out.getvalue()


# =========================================================================
# 20 EXTRACTION / END-TO-END EVALUATION CASES
# =========================================================================

RAW_EXTRACTION_CASES: List[Dict[str, Any]] = [
    {
        "case_id": "EXT-01",
        "title": "Clear first-line recommendation",
        "filename": "ext_01_clear_rec.pdf",
        "source_identifier": "SYN-ADA-2026-EXT-01",
        "condition_type": "clear_recommendation",
        "pages": [
            [
                "1. Clinical Recommendations",
                "Adult patients with type 2 diabetes should receive metformin 500mg daily as first-line therapy.",
            ]
        ],
        "expected_behavior": "extracted_successfully",
        "expected_status": ChangeStatus.EXTRACTED.value,
        "expected_gate": None,
        "expected_pipeline_stage": PipelineStage.GOVERNANCE.value,
        "expected_difference_type": DifferenceType.INTERVENTION_CHANGE.value,
        "expected_impact_tier": "standard",
        "expected_sla": "30 days",
        "confidence": 0.95,
        "verbatim_text": "Adult patients with type 2 diabetes should receive metformin 500mg daily as first-line therapy.",
        "recommendation_type": "treatment",
        "target_population": "Adult patients with type 2 diabetes",
        "intervention": "Metformin 500mg daily as first-line therapy",
        "evidence_grade": "Grade A",
        "page": 1,
        "section": "1. Clinical Recommendations",
        "rationale": "Clear explicit treatment recommendation with high evidence and confidence.",
    },
    {
        "case_id": "EXT-02",
        "title": "Dosage change recommendation",
        "filename": "ext_02_dosage_change.pdf",
        "source_identifier": "SYN-ADA-2026-EXT-02",
        "condition_type": "dosage_change",
        "pages": [
            [
                "1. Pharmacotherapy Dosing",
                "Adults with type 2 diabetes should initiate metformin at 1000mg twice daily with meals.",
            ]
        ],
        "expected_behavior": "extracted_successfully",
        "expected_status": ChangeStatus.EXTRACTED.value,
        "expected_gate": None,
        "expected_pipeline_stage": PipelineStage.GOVERNANCE.value,
        "expected_difference_type": DifferenceType.DOSAGE_CHANGE.value,
        "expected_impact_tier": "standard",
        "expected_sla": "30 days",
        "confidence": 0.94,
        "verbatim_text": "Adults with type 2 diabetes should initiate metformin at 1000mg twice daily with meals.",
        "recommendation_type": "dosing",
        "target_population": "Adults with type 2 diabetes",
        "intervention": "Metformin 1000mg twice daily with meals",
        "evidence_grade": "Grade B",
        "page": 1,
        "section": "1. Pharmacotherapy Dosing",
        "rationale": "Explicit dosage increase compared to baseline institutional protocol (1000mg BID vs 500mg daily).",
    },
    {
        "case_id": "EXT-03",
        "title": "Monitoring frequency change",
        "filename": "ext_03_monitoring_freq.pdf",
        "source_identifier": "SYN-ADA-2026-EXT-03",
        "condition_type": "monitoring_frequency_change",
        "pages": [
            [
                "1. Renal Monitoring Recommendations",
                "Clinicians should monitor eGFR every 6 months in all adult patients receiving metformin.",
            ]
        ],
        "expected_behavior": "extracted_successfully",
        "expected_status": ChangeStatus.EXTRACTED.value,
        "expected_gate": None,
        "expected_pipeline_stage": PipelineStage.GOVERNANCE.value,
        "expected_difference_type": DifferenceType.FREQUENCY_CHANGE.value,
        "expected_impact_tier": "high",
        "expected_sla": "7 days",
        "confidence": 0.92,
        "verbatim_text": "Clinicians should monitor eGFR every 6 months in all adult patients receiving metformin.",
        "recommendation_type": "monitoring",
        "target_population": "Adult patients receiving metformin",
        "intervention": "eGFR monitoring every 6 months",
        "evidence_grade": "Grade B",
        "page": 1,
        "section": "1. Renal Monitoring Recommendations",
        "rationale": "Tightens renal surveillance frequency from annual to semi-annual.",
    },
    {
        "case_id": "EXT-04",
        "title": "Diagnostic criterion threshold change",
        "filename": "ext_04_diagnostic_criterion.pdf",
        "source_identifier": "SYN-ADA-2026-EXT-04",
        "condition_type": "diagnostic_criterion_change",
        "pages": [
            [
                "1. Diagnostic Thresholds and Recommendations",
                "The recommended diagnostic threshold for type 2 diabetes is fasting plasma glucose >= 125 mg/dL or HbA1c >= 6.3%.",
            ]
        ],
        "expected_behavior": "extracted_successfully",
        "expected_status": ChangeStatus.EXTRACTED.value,
        "expected_gate": None,
        "expected_pipeline_stage": PipelineStage.GOVERNANCE.value,
        "expected_difference_type": DifferenceType.THRESHOLD_CHANGE.value,
        "expected_impact_tier": "high",
        "expected_sla": "7 days",
        "confidence": 0.93,
        "verbatim_text": "The recommended diagnostic threshold for type 2 diabetes is fasting plasma glucose >= 125 mg/dL or HbA1c >= 6.3%.",
        "recommendation_type": "diagnostic",
        "target_population": "Adult patients undergoing screening",
        "intervention": "Fasting plasma glucose >= 125 mg/dL or HbA1c >= 6.3%",
        "evidence_grade": "Grade A",
        "page": 1,
        "section": "1. Diagnostic Thresholds and Recommendations",
        "rationale": "Changes diagnostic thresholds downward from standard 126 mg/dL and 6.5%.",
    },
    {
        "case_id": "EXT-05",
        "title": "Urgent safety warning",
        "filename": "ext_05_safety_warning.pdf",
        "source_identifier": "SYN-ADA-2026-EXT-05",
        "condition_type": "safety_warning",
        "pages": [
            [
                "1. Safety Notice and Warnings",
                "Metformin therapy should be immediately suspended in patients developing acute hypoxemia due to lactic acidosis risk.",
            ]
        ],
        "expected_behavior": "extracted_successfully",
        "expected_status": ChangeStatus.EXTRACTED.value,
        "expected_gate": None,
        "expected_pipeline_stage": PipelineStage.GOVERNANCE.value,
        "expected_difference_type": DifferenceType.CONTRAINDICATION.value,
        "expected_impact_tier": "critical",
        "expected_sla": "48 hours",
        "confidence": 0.91,
        "verbatim_text": "Metformin therapy should be immediately suspended in patients developing acute hypoxemia due to lactic acidosis risk.",
        "recommendation_type": "safety_warning",
        "target_population": "Patients on metformin developing acute hypoxemia",
        "intervention": "Immediate metformin suspension",
        "evidence_grade": "Grade A",
        "page": 1,
        "section": "1. Safety Notice and Warnings",
        "rationale": "Acute life-safety notice with critical impact tier requiring rapid 48h turnaround.",
    },
    {
        "case_id": "EXT-06",
        "title": "Strict contraindication rule",
        "filename": "ext_06_contraindication.pdf",
        "source_identifier": "SYN-ADA-2026-EXT-06",
        "condition_type": "contraindication",
        "pages": [
            [
                "1. Contraindications",
                "Metformin is contraindicated in patients with eGFR below 30 mL/min/1.73m2 or undergoing iodinated radiocontrast.",
            ]
        ],
        "expected_behavior": "extracted_successfully",
        "expected_status": ChangeStatus.EXTRACTED.value,
        "expected_gate": None,
        "expected_pipeline_stage": PipelineStage.GOVERNANCE.value,
        "expected_difference_type": DifferenceType.CONTRAINDICATION.value,
        "expected_impact_tier": "critical",
        "expected_sla": "48 hours",
        "confidence": 0.96,
        "verbatim_text": "Metformin is contraindicated in patients with eGFR below 30 mL/min/1.73m2 or undergoing iodinated radiocontrast.",
        "recommendation_type": "contraindication",
        "target_population": "Patients with eGFR < 30 or undergoing iodinated radiocontrast",
        "intervention": "Withhold or avoid metformin",
        "evidence_grade": "Grade A",
        "page": 1,
        "section": "1. Contraindications",
        "rationale": "Formal contraindication expanding safety restrictions for radiocontrast.",
    },
    {
        "case_id": "EXT-07",
        "title": "Population-specific recommendation",
        "filename": "ext_07_population_specific.pdf",
        "source_identifier": "SYN-ADA-2026-EXT-07",
        "condition_type": "population_specific",
        "pages": [
            [
                "1. Geriatric Considerations",
                "In elderly adults aged 75 years and older, target HbA1c should be relaxed to less than 8.0%.",
            ]
        ],
        "expected_behavior": "extracted_successfully",
        "expected_status": ChangeStatus.EXTRACTED.value,
        "expected_gate": None,
        "expected_pipeline_stage": PipelineStage.GOVERNANCE.value,
        "expected_difference_type": DifferenceType.POPULATION_RESTRICTION.value,
        "expected_impact_tier": "standard",
        "expected_sla": "30 days",
        "confidence": 0.90,
        "verbatim_text": "In elderly adults aged 75 years and older, target HbA1c should be relaxed to less than 8.0%.",
        "recommendation_type": "treatment",
        "target_population": "Elderly adults aged 75 years and older",
        "intervention": "Relaxed glycemic target HbA1c < 8.0%",
        "evidence_grade": "Grade B",
        "page": 1,
        "section": "1. Geriatric Considerations",
        "rationale": "Population subgroup stratification tailoring glycemic target.",
    },
    {
        "case_id": "EXT-08",
        "title": "Explicit evidence grade present",
        "filename": "ext_08_evidence_grade_present.pdf",
        "source_identifier": "SYN-ADA-2026-EXT-08",
        "condition_type": "evidence_grade_present",
        "pages": [
            [
                "1. Evidence-Based Interventions",
                "Clinicians should prescribe SGLT2 inhibitors for diabetic patients with established cardiovascular disease (Grade A).",
            ]
        ],
        "expected_behavior": "extracted_successfully",
        "expected_status": ChangeStatus.EXTRACTED.value,
        "expected_gate": None,
        "expected_pipeline_stage": PipelineStage.GOVERNANCE.value,
        "expected_difference_type": DifferenceType.INTERVENTION_CHANGE.value,
        "expected_impact_tier": "high",
        "expected_sla": "7 days",
        "confidence": 0.94,
        "verbatim_text": "Clinicians should prescribe SGLT2 inhibitors for diabetic patients with established cardiovascular disease (Grade A).",
        "recommendation_type": "treatment",
        "target_population": "Diabetic patients with established cardiovascular disease",
        "intervention": "Prescribe SGLT2 inhibitors",
        "evidence_grade": "Grade A",
        "page": 1,
        "section": "1. Evidence-Based Interventions",
        "rationale": "Clear Grade A citation directly in sentence body.",
    },
    {
        "case_id": "EXT-09",
        "title": "Evidence grade missing",
        "filename": "ext_09_evidence_grade_missing.pdf",
        "source_identifier": "SYN-ADA-2026-EXT-09",
        "condition_type": "evidence_grade_missing",
        "pages": [
            [
                "1. Practice Recommendations",
                "Lifestyle modification including 150 minutes of weekly aerobic exercise should be recommended to all diabetic adults.",
            ]
        ],
        "expected_behavior": "extracted_successfully",
        "expected_status": ChangeStatus.EXTRACTED.value,
        "expected_gate": None,
        "expected_pipeline_stage": PipelineStage.IMPACT.value,
        "expected_difference_type": DifferenceType.INTERVENTION_CHANGE.value,
        "expected_impact_tier": None,
        "expected_sla": None,
        "confidence": 0.88,
        "verbatim_text": "Lifestyle modification including 150 minutes of weekly aerobic exercise should be recommended to all diabetic adults.",
        "recommendation_type": "lifestyle",
        "target_population": "All diabetic adults",
        "intervention": "150 minutes of weekly aerobic exercise",
        "evidence_grade": None,
        "page": 1,
        "section": "1. Practice Recommendations",
        "rationale": "Valid clinical recommendation where evidence grade is unstated; halts at impact stage due to incomplete evidence dimension without guessing tiers.",
    },
    {
        "case_id": "EXT-10",
        "title": "Multiple discrete recommendations in single section",
        "filename": "ext_10_multiple_recs.pdf",
        "source_identifier": "SYN-ADA-2026-EXT-10",
        "condition_type": "multiple_recommendations",
        "pages": [
            [
                "1. Combined Clinical Guidance",
                "Adult patients with type 2 diabetes should receive metformin 500mg daily.",
                "Additionally, clinicians should prescribe GLP-1 receptor agonists if HbA1c remains above target.",
            ]
        ],
        "expected_behavior": "extracted_successfully",
        "expected_status": ChangeStatus.EXTRACTED.value,
        "expected_gate": None,
        "expected_pipeline_stage": PipelineStage.GOVERNANCE.value,
        "expected_difference_type": DifferenceType.INTERVENTION_CHANGE.value,
        "expected_impact_tier": "high",
        "expected_sla": "7 days",
        "confidence": 0.93,
        "verbatim_text": "Adult patients with type 2 diabetes should receive metformin 500mg daily.",
        "recommendation_type": "treatment",
        "target_population": "Adult patients with type 2 diabetes",
        "intervention": "Metformin 500mg daily",
        "evidence_grade": "Grade A",
        "page": 1,
        "section": "1. Combined Clinical Guidance",
        "rationale": "Verifies extractor isolates multiple distinct recommendations.",
    },
    {
        "case_id": "EXT-11",
        "title": "Different section layout (Appendix)",
        "filename": "ext_11_appendix_layout.pdf",
        "source_identifier": "SYN-ADA-2026-EXT-11",
        "condition_type": "different_section_layouts",
        "pages": [
            [
                "Section 9: Appendix B Clinical Guidance",
                "Clinicians should initiate basal insulin therapy when symptomatic hyperglycemia persists despite dual oral therapy.",
            ]
        ],
        "expected_behavior": "extracted_successfully",
        "expected_status": ChangeStatus.EXTRACTED.value,
        "expected_gate": None,
        "expected_pipeline_stage": PipelineStage.GOVERNANCE.value,
        "expected_difference_type": DifferenceType.INTERVENTION_CHANGE.value,
        "expected_impact_tier": "standard",
        "expected_sla": "30 days",
        "confidence": 0.92,
        "verbatim_text": "Clinicians should initiate basal insulin therapy when symptomatic hyperglycemia persists despite dual oral therapy.",
        "recommendation_type": "treatment",
        "target_population": "Patients with persistent symptomatic hyperglycemia",
        "intervention": "Basal insulin therapy",
        "evidence_grade": "Grade B",
        "page": 1,
        "section": "Section 9: Appendix B Clinical Guidance",
        "rationale": "Recommendation nested in non-standard Appendix heading pattern.",
    },
    {
        "case_id": "EXT-12",
        "title": "Recommendation on Page 2 of multi-page document",
        "filename": "ext_12_multipage_page2.pdf",
        "source_identifier": "SYN-ADA-2026-EXT-12",
        "condition_type": "recommendation_different_page",
        "pages": [
            [
                "CLINICAL GUIDELINE OVERVIEW",
                "This document summarizes institutional glycemic guidance for ambulatory clinics.",
            ],
            [
                "1. Treatment Recommendations",
                "Adult patients with type 2 diabetes should be started on metformin 850mg once daily.",
            ],
        ],
        "expected_behavior": "extracted_successfully",
        "expected_status": ChangeStatus.EXTRACTED.value,
        "expected_gate": None,
        "expected_pipeline_stage": PipelineStage.GOVERNANCE.value,
        "expected_difference_type": DifferenceType.DOSAGE_CHANGE.value,
        "expected_impact_tier": "standard",
        "expected_sla": "30 days",
        "confidence": 0.93,
        "verbatim_text": "Adult patients with type 2 diabetes should be started on metformin 850mg once daily.",
        "recommendation_type": "dosing",
        "target_population": "Adult patients with type 2 diabetes",
        "intervention": "Metformin 850mg once daily",
        "evidence_grade": "Grade A",
        "page": 2,
        "section": "1. Treatment Recommendations",
        "rationale": "Verifies multi-page tracking and page provenance attribution (page 2).",
    },
    {
        "case_id": "EXT-13",
        "title": "Irrelevant preface section prior to recommendations",
        "filename": "ext_13_irrelevant_preface.pdf",
        "source_identifier": "SYN-ADA-2026-EXT-13",
        "condition_type": "irrelevant_sections_before",
        "pages": [
            [
                "Historical Background and Epidemiology",
                "Type 2 diabetes prevalence has escalated globally over recent decades.",
                "1. Management Protocol",
                "Adult patients should receive dietary counseling and metformin 500mg daily.",
            ]
        ],
        "expected_behavior": "extracted_successfully",
        "expected_status": ChangeStatus.EXTRACTED.value,
        "expected_gate": None,
        "expected_pipeline_stage": PipelineStage.GOVERNANCE.value,
        "expected_difference_type": DifferenceType.INTERVENTION_CHANGE.value,
        "expected_impact_tier": "standard",
        "expected_sla": "30 days",
        "confidence": 0.94,
        "verbatim_text": "Adult patients should receive dietary counseling and metformin 500mg daily.",
        "recommendation_type": "treatment",
        "target_population": "Adult patients",
        "intervention": "Dietary counseling and metformin 500mg daily",
        "evidence_grade": "Grade A",
        "page": 1,
        "section": "1. Management Protocol",
        "rationale": "Proves candidate section filter bypasses background section without recommendations.",
    },
    {
        "case_id": "EXT-14",
        "title": "Recommendation surrounded by distracting narrative",
        "filename": "ext_14_distracting_text.pdf",
        "source_identifier": "SYN-ADA-2026-EXT-14",
        "condition_type": "distracting_clinical_text",
        "pages": [
            [
                "1. Pharmacotherapy and Observations",
                "Observational cohort data from 2021 noted varied adherence rates in clinic settings.",
                "Adult patients with type 2 diabetes should receive metformin 500mg daily.",
                "Further retrospective analyses are planned next year by the registry team.",
            ]
        ],
        "expected_behavior": "extracted_successfully",
        "expected_status": ChangeStatus.EXTRACTED.value,
        "expected_gate": None,
        "expected_pipeline_stage": PipelineStage.GOVERNANCE.value,
        "expected_difference_type": DifferenceType.DOSAGE_CHANGE.value,
        "expected_impact_tier": "standard",
        "expected_sla": "30 days",
        "confidence": 0.92,
        "verbatim_text": "Adult patients with type 2 diabetes should receive metformin 500mg daily.",
        "recommendation_type": "treatment",
        "target_population": "Adult patients with type 2 diabetes",
        "intervention": "Metformin 500mg daily",
        "evidence_grade": "Grade A",
        "page": 1,
        "section": "1. Pharmacotherapy and Observations",
        "rationale": "Verbatim recommendation isolated accurately amidst non-actionable commentary.",
    },
    {
        "case_id": "EXT-15",
        "title": "Complex punctuation and symbols provenance edge case",
        "filename": "ext_15_provenance_symbols.pdf",
        "source_identifier": "SYN-ADA-2026-EXT-15",
        "condition_type": "provenance_edge_case",
        "pages": [
            [
                "1. Glycemic Targets and Thresholds",
                "Target HbA1c for non-pregnant adult patients is < 7.0% (53 mmol/mol), reducing microvascular complications.",
            ]
        ],
        "expected_behavior": "extracted_successfully",
        "expected_status": ChangeStatus.EXTRACTED.value,
        "expected_gate": None,
        "expected_pipeline_stage": PipelineStage.GOVERNANCE.value,
        "expected_difference_type": DifferenceType.THRESHOLD_CHANGE.value,
        "expected_impact_tier": "high",
        "expected_sla": "7 days",
        "confidence": 0.95,
        "verbatim_text": "Target HbA1c for non-pregnant adult patients is < 7.0% (53 mmol/mol), reducing microvascular complications.",
        "recommendation_type": "target",
        "target_population": "Non-pregnant adult patients",
        "intervention": "Target HbA1c < 7.0% (53 mmol/mol)",
        "evidence_grade": "Grade A",
        "page": 1,
        "section": "1. Glycemic Targets and Thresholds",
        "rationale": "Exact match verification retains parentheses and mathematical symbols.",
    },
    {
        "case_id": "EXT-16",
        "title": "Ambiguous recommendation phrasing requiring human review",
        "filename": "ext_16_ambiguous_rec.pdf",
        "source_identifier": "SYN-ADA-2026-EXT-16",
        "condition_type": "ambiguous_recommendation",
        "pages": [
            [
                "1. Treatment Considerations",
                "Clinicians may consider alternative non-metformin therapies if individual clinical circumstances suggest potential benefits.",
            ]
        ],
        "expected_behavior": "held_for_g1",
        "expected_status": ChangeStatus.HELD_FOR_G1.value,
        "expected_gate": HumanGate.G1.value,
        "expected_pipeline_stage": PipelineStage.EXTRACTION.value,
        "expected_difference_type": DifferenceType.NONE.value,
        "expected_impact_tier": None,
        "expected_sla": None,
        "confidence": 0.65,
        "verbatim_text": "Clinicians may consider alternative non-metformin therapies if individual clinical circumstances suggest potential benefits.",
        "recommendation_type": "treatment",
        "target_population": "Patients with potential benefits from alternatives",
        "intervention": "Alternative non-metformin therapies",
        "evidence_grade": None,
        "page": 1,
        "section": "1. Treatment Considerations",
        "rationale": "Weak recommendation confidence (0.65 < 0.70 threshold) routes to G1 human review.",
    },
    {
        "case_id": "EXT-17",
        "title": "Low-confidence extraction case",
        "filename": "ext_07_low_confidence_g1.pdf",
        "source_identifier": "SYN-ADA-2026-EXT-17",
        "condition_type": "low_confidence_g1",
        "pages": [
            [
                "1. Preliminary Therapy Suggestions",
                "Some clinical practitioners suggest adjusting bedtime dosing for borderline glucose readings.",
            ]
        ],
        "expected_behavior": "held_for_g1",
        "expected_status": ChangeStatus.HELD_FOR_G1.value,
        "expected_gate": HumanGate.G1.value,
        "expected_pipeline_stage": PipelineStage.EXTRACTION.value,
        "expected_difference_type": DifferenceType.NONE.value,
        "expected_impact_tier": None,
        "expected_sla": None,
        "confidence": 0.58,
        "verbatim_text": "Some clinical practitioners suggest adjusting bedtime dosing for borderline glucose readings.",
        "recommendation_type": "dosing",
        "target_population": "Patients with borderline glucose",
        "intervention": "Adjust bedtime dosing",
        "evidence_grade": None,
        "page": 1,
        "section": "1. Preliminary Therapy Suggestions",
        "rationale": "Confidence 0.58 is strictly below production 0.70 threshold, halting at G1.",
    },
    {
        "case_id": "EXT-18",
        "title": "Low-confidence recommendation text held for G1 review",
        "filename": "ext_18_malformed_invalid.pdf",
        "source_identifier": "SYN-ADA-2026-EXT-18",
        "condition_type": "malformed_invalid",
        "pages": [
            [
                "1. Protocol Considerations and Recommendations",
                "Clinicians should consider uncertain therapy adjustments as deemed appropriate.",
            ]
        ],
        "expected_behavior": "held_for_g1",
        "expected_status": ChangeStatus.HELD_FOR_G1.value,
        "expected_gate": HumanGate.G1.value,
        "expected_pipeline_stage": PipelineStage.EXTRACTION.value,
        "expected_difference_type": DifferenceType.NONE.value,
        "expected_impact_tier": None,
        "expected_sla": None,
        "confidence": 0.40,
        "verbatim_text": "Clinicians should consider uncertain therapy adjustments as deemed appropriate.",
        "recommendation_type": "treatment",
        "target_population": "Patients with uncertain therapy needs",
        "intervention": "Uncertain therapy adjustments",
        "evidence_grade": None,
        "page": 1,
        "section": "1. Protocol Considerations and Recommendations",
        "rationale": "Produces low-confidence / unparseable extraction held strictly for G1 human review.",
    },
    {
        "case_id": "EXT-19",
        "title": "Source quote hallucination mismatch edge case",
        "filename": "ext_19_quote_mismatch.pdf",
        "source_identifier": "SYN-ADA-2026-EXT-19",
        "condition_type": "source_quote_mismatch",
        "pages": [
            [
                "1. Pharmacologic Therapy",
                "Adult patients with type 2 diabetes should take metformin 500mg once daily.",
            ]
        ],
        "expected_behavior": "held_for_g1",
        "expected_status": ChangeStatus.HELD_FOR_G1.value,
        "expected_gate": HumanGate.G1.value,
        "expected_pipeline_stage": PipelineStage.EXTRACTION.value,
        "expected_difference_type": DifferenceType.NONE.value,
        "expected_impact_tier": None,
        "expected_sla": None,
        "confidence": 0.95,
        "verbatim_text": "Patients with diabetes must take sulfonylureas 10mg daily.",
        "recommendation_type": "treatment",
        "target_population": "Patients with diabetes",
        "intervention": "Sulfonylureas 10mg daily",
        "evidence_grade": "Grade A",
        "page": 1,
        "section": "1. Pharmacologic Therapy",
        "rationale": "Verbatim quote does not exist in source section text; fails provenance verification and halts at G1.",
    },
    {
        "case_id": "EXT-20",
        "title": "Document with multiple plausible candidate sections",
        "filename": "ext_20_multiple_candidates.pdf",
        "source_identifier": "SYN-ADA-2026-EXT-20",
        "condition_type": "multiple_candidate_sections",
        "pages": [
            [
                "1. Primary First-Line Treatment",
                "Adult patients should receive metformin 500mg daily.",
                "2. Secondary Treatment Options",
                "If metformin intolerance occurs, clinicians should substitute with DPP-4 inhibitors.",
            ]
        ],
        "expected_behavior": "extracted_successfully",
        "expected_status": ChangeStatus.EXTRACTED.value,
        "expected_gate": None,
        "expected_pipeline_stage": PipelineStage.GOVERNANCE.value,
        "expected_difference_type": DifferenceType.INTERVENTION_CHANGE.value,
        "expected_impact_tier": "standard",
        "expected_sla": "30 days",
        "confidence": 0.94,
        "verbatim_text": "Adult patients should receive metformin 500mg daily.",
        "recommendation_type": "treatment",
        "target_population": "Adult patients",
        "intervention": "Metformin 500mg daily",
        "evidence_grade": "Grade A",
        "page": 1,
        "section": "1. Primary First-Line Treatment",
        "rationale": "Verifies candidate section detection and extraction across multiple eligible sections.",
    },
]


# =========================================================================
# 16 COMPARISON EVALUATION SCENARIOS
# =========================================================================

RAW_COMPARISON_SCENARIOS: List[Dict[str, Any]] = [
    {
        "scenario_id": "CMP-01",
        "name": "Exact protocol match",
        "condition_type": "exact_match",
        "input_recommendation": "Metformin 500mg once daily with evening meals is the preferred initial pharmacologic agent for type 2 diabetes.",
        "target_population": "Adult patients with type 2 diabetes",
        "intervention": "Metformin 500mg once daily with evening meals",
        "evidence_grade": "Grade A",
        "protocol_id": "PROT-DM-001",
        "protocol_version": "v1.0",
        "candidate_section_id": "SEC-2",
        "candidate_section_heading": "First-Line Pharmacotherapy",
        "candidate_section_text": "Metformin 500mg once daily with evening meals is the preferred initial pharmacologic agent for type 2 diabetes.",
        "expected_comparison_result": ComparisonResult.NO_GAP.value,
        "expected_difference_type": DifferenceType.NONE.value,
        "expected_gap_status": GapStatus.MATCHED.value,
        "expected_gate": None,
        "expected_confidence": 0.98,
        "is_match": True,
        "is_boundary_case": False,
        "rationale": "Identical wording matches institutional protocol exactly with no gap.",
    },
    {
        "scenario_id": "CMP-02",
        "name": "Wording variation with semantic equivalence",
        "condition_type": "wording_variation",
        "input_recommendation": "For patients with type 2 diabetes mellitus, initial therapy should begin with metformin 500 mg once per day taken with the evening meal.",
        "target_population": "Patients with type 2 diabetes mellitus",
        "intervention": "Metformin 500 mg once per day with evening meal",
        "evidence_grade": "Grade A",
        "protocol_id": "PROT-DM-001",
        "protocol_version": "v1.0",
        "candidate_section_id": "SEC-2",
        "candidate_section_heading": "First-Line Pharmacotherapy",
        "candidate_section_text": "Metformin 500mg once daily with evening meals is the preferred initial pharmacologic agent for type 2 diabetes.",
        "expected_comparison_result": ComparisonResult.NO_GAP.value,
        "expected_difference_type": DifferenceType.NO_MATERIAL_DIFFERENCE.value,
        "expected_gap_status": GapStatus.MATCHED.value,
        "expected_gate": None,
        "expected_confidence": 0.94,
        "is_match": True,
        "is_boundary_case": False,
        "rationale": "Paraphrased syntax representing clinically identical instruction.",
    },
    {
        "scenario_id": "CMP-03",
        "name": "Dosage change finding",
        "condition_type": "dosage_change",
        "input_recommendation": "Initiate metformin at 1000mg twice daily with meals for adults with type 2 diabetes.",
        "target_population": "Adults with type 2 diabetes",
        "intervention": "Metformin 1000mg twice daily with meals",
        "evidence_grade": "Grade B",
        "protocol_id": "PROT-DM-001",
        "protocol_version": "v1.0",
        "candidate_section_id": "SEC-2",
        "candidate_section_heading": "First-Line Pharmacotherapy",
        "candidate_section_text": "Metformin 500mg once daily with evening meals is the preferred initial pharmacologic agent for type 2 diabetes.",
        "expected_comparison_result": ComparisonResult.GAP.value,
        "expected_difference_type": DifferenceType.DOSAGE_CHANGE.value,
        "expected_gap_status": GapStatus.MATCHED.value,
        "expected_gate": None,
        "expected_confidence": 0.95,
        "is_match": True,
        "is_boundary_case": False,
        "rationale": "Material dosing escalation from 500mg once daily to 1000mg twice daily.",
    },
    {
        "scenario_id": "CMP-04",
        "name": "Monitoring frequency change",
        "condition_type": "frequency_change",
        "input_recommendation": "Monitor eGFR every 6 months in all adult patients receiving metformin.",
        "target_population": "Adult patients receiving metformin",
        "intervention": "eGFR monitoring every 6 months",
        "evidence_grade": "Grade B",
        "protocol_id": "PROT-DM-001",
        "protocol_version": "v1.0",
        "candidate_section_id": "SEC-4",
        "candidate_section_heading": "Renal Monitoring and Dose Adjustments",
        "candidate_section_text": "Monitor eGFR annually. Discontinue metformin if eGFR falls below 30 mL/min/1.73m2 due to lactic acidosis risk.",
        "expected_comparison_result": ComparisonResult.GAP.value,
        "expected_difference_type": DifferenceType.FREQUENCY_CHANGE.value,
        "expected_gap_status": GapStatus.MATCHED.value,
        "expected_gate": None,
        "expected_confidence": 0.92,
        "is_match": True,
        "is_boundary_case": False,
        "rationale": "Annual eGFR requirement accelerated to semi-annual monitoring.",
    },
    {
        "scenario_id": "CMP-05",
        "name": "Target population restriction",
        "condition_type": "population_change",
        "input_recommendation": "Metformin is recommended as first-line therapy exclusively for adults under 65 years of age with type 2 diabetes.",
        "target_population": "Adults under 65 years of age",
        "intervention": "Metformin as first-line therapy",
        "evidence_grade": "Grade B",
        "protocol_id": "PROT-DM-001",
        "protocol_version": "v1.0",
        "candidate_section_id": "SEC-2",
        "candidate_section_heading": "First-Line Pharmacotherapy",
        "candidate_section_text": "Metformin 500mg once daily with evening meals is the preferred initial pharmacologic agent for type 2 diabetes.",
        "expected_comparison_result": ComparisonResult.GAP.value,
        "expected_difference_type": DifferenceType.POPULATION_RESTRICTION.value,
        "expected_gap_status": GapStatus.MATCHED.value,
        "expected_gate": None,
        "expected_confidence": 0.91,
        "is_match": True,
        "is_boundary_case": False,
        "rationale": "Restricts broadly indicated population to patients under 65.",
    },
    {
        "scenario_id": "CMP-06",
        "name": "Contraindication and safety difference",
        "condition_type": "contraindication",
        "input_recommendation": "Metformin must be temporarily withheld in patients undergoing iodinated radiocontrast procedures.",
        "target_population": "Patients undergoing iodinated contrast",
        "intervention": "Temporary metformin withholding",
        "evidence_grade": "Grade A",
        "protocol_id": "PROT-DM-001",
        "protocol_version": "v1.0",
        "candidate_section_id": "SEC-4",
        "candidate_section_heading": "Renal Monitoring and Dose Adjustments",
        "candidate_section_text": "Monitor eGFR annually. Discontinue metformin if eGFR falls below 30 mL/min/1.73m2 due to lactic acidosis risk.",
        "expected_comparison_result": ComparisonResult.GAP.value,
        "expected_difference_type": DifferenceType.CONTRAINDICATION.value,
        "expected_gap_status": GapStatus.MATCHED.value,
        "expected_gate": None,
        "expected_confidence": 0.96,
        "is_match": True,
        "is_boundary_case": False,
        "rationale": "Introduces new radiocontrast safety restriction not in existing protocol.",
    },
    {
        "scenario_id": "CMP-07",
        "name": "Clinical workflow change",
        "condition_type": "workflow_difference",
        "input_recommendation": "Prior to initiating metformin, obtain mandatory clinical pharmacist medication reconciliation.",
        "target_population": "Patients initiating metformin",
        "intervention": "Mandatory pharmacist medication reconciliation",
        "evidence_grade": "Grade B",
        "protocol_id": "PROT-DM-001",
        "protocol_version": "v1.0",
        "candidate_section_id": "SEC-2",
        "candidate_section_heading": "First-Line Pharmacotherapy",
        "candidate_section_text": "Metformin 500mg once daily with evening meals is the preferred initial pharmacologic agent for type 2 diabetes.",
        "expected_comparison_result": ComparisonResult.GAP.value,
        "expected_difference_type": DifferenceType.OTHER_SUPPORTED_CHANGE.value,
        "expected_gap_status": GapStatus.MATCHED.value,
        "expected_gate": None,
        "expected_confidence": 0.89,
        "is_match": True,
        "is_boundary_case": False,
        "rationale": "Mandates prerequisite pharmacist workflow prior to dispensing.",
    },
    {
        "scenario_id": "CMP-08",
        "name": "Protocol section with no meaningful change",
        "condition_type": "no_gap",
        "input_recommendation": "Target HbA1c for non-pregnant adult patients is < 7.0% to minimize microvascular complications.",
        "target_population": "Non-pregnant adult patients",
        "intervention": "Target HbA1c < 7.0%",
        "evidence_grade": "Grade A",
        "protocol_id": "PROT-DM-001",
        "protocol_version": "v1.0",
        "candidate_section_id": "SEC-3",
        "candidate_section_heading": "Glycemic Targets",
        "candidate_section_text": "Target HbA1c for non-pregnant adult patients is < 7.0% to minimize microvascular and macrovascular complications.",
        "expected_comparison_result": ComparisonResult.NO_GAP.value,
        "expected_difference_type": DifferenceType.NONE.value,
        "expected_gap_status": GapStatus.MATCHED.value,
        "expected_gate": None,
        "expected_confidence": 0.97,
        "is_match": True,
        "is_boundary_case": False,
        "rationale": "Directly corroborates existing institutional glycemic target.",
    },
    {
        "scenario_id": "CMP-09",
        "name": "Multiple candidate sections comparison",
        "condition_type": "multiple_candidates",
        "input_recommendation": "Initiate metformin 500mg daily while monitoring HbA1c to achieve target < 7.0%.",
        "target_population": "Adult patients with type 2 diabetes",
        "intervention": "Metformin 500mg daily with HbA1c tracking",
        "evidence_grade": "Grade A",
        "protocol_id": "PROT-DM-001",
        "protocol_version": "v1.0",
        "candidate_section_id": "SEC-2",
        "candidate_section_heading": "First-Line Pharmacotherapy",
        "candidate_section_text": "Metformin 500mg once daily with evening meals is the preferred initial pharmacologic agent for type 2 diabetes.",
        "expected_comparison_result": ComparisonResult.GAP.value,
        "expected_difference_type": DifferenceType.DOSAGE_CHANGE.value,
        "expected_gap_status": GapStatus.MATCHED.value,
        "expected_gate": None,
        "expected_confidence": 0.90,
        "is_match": True,
        "is_boundary_case": False,
        "rationale": "Matches multiple candidate sections (SEC-2 and SEC-3); best candidate SEC-2 correctly chosen.",
    },
    {
        "scenario_id": "CMP-10",
        "name": "Ambiguous comparison requiring G2 hold",
        "condition_type": "ambiguous",
        "input_recommendation": "Consider metformin or alternative depending on ambiguous clinical judgment.",
        "target_population": "Patients with uncertain indications",
        "intervention": "Metformin or alternative based on clinician discretion",
        "evidence_grade": None,
        "protocol_id": "PROT-DM-001",
        "protocol_version": "v1.0",
        "candidate_section_id": "SEC-2",
        "candidate_section_heading": "First-Line Pharmacotherapy",
        "candidate_section_text": "Metformin 500mg once daily with evening meals is the preferred initial pharmacologic agent for type 2 diabetes.",
        "expected_comparison_result": ComparisonResult.AMBIGUOUS.value,
        "expected_difference_type": DifferenceType.NONE.value,
        "expected_gap_status": GapStatus.REVIEW_REQUIRED.value,
        "expected_gate": HumanGate.G2.value,
        "expected_confidence": 0.50,
        "is_match": True,
        "is_boundary_case": False,
        "rationale": "Equivocal clinical instruction creates ambiguous comparison routing to G2.",
    },
    {
        "scenario_id": "CMP-11",
        "name": "Internally contradictory comparison output",
        "condition_type": "contradiction",
        "input_recommendation": "Prescribe metformin 500mg daily.",
        "target_population": "Adults with diabetes",
        "intervention": "Metformin 500mg daily",
        "evidence_grade": "Grade A",
        "protocol_id": "PROT-DM-001",
        "protocol_version": "v1.0",
        "candidate_section_id": "SEC-2",
        "candidate_section_heading": "First-Line Pharmacotherapy",
        "candidate_section_text": "Metformin 500mg once daily with evening meals is the preferred initial pharmacologic agent for type 2 diabetes.",
        "expected_comparison_result": ComparisonResult.NO_GAP.value,
        "expected_difference_type": DifferenceType.DOSAGE_CHANGE.value,
        "expected_gap_status": GapStatus.REVIEW_REQUIRED.value,
        "expected_gate": HumanGate.G2.value,
        "expected_confidence": 0.60,
        "is_match": True,
        "is_boundary_case": False,
        "rationale": "Contradictory output (no_gap with dosage_change) is intercepted and routed to G2.",
    },
    {
        "scenario_id": "CMP-12",
        "name": "Invalid or unparseable comparison response",
        "condition_type": "invalid",
        "input_recommendation": "Metformin recommendation text with malformed LLM response.",
        "target_population": "Adults with diabetes",
        "intervention": "Metformin therapy",
        "evidence_grade": "Grade A",
        "protocol_id": "PROT-DM-001",
        "protocol_version": "v1.0",
        "candidate_section_id": "SEC-2",
        "candidate_section_heading": "First-Line Pharmacotherapy",
        "candidate_section_text": "Metformin 500mg once daily with evening meals is the preferred initial pharmacologic agent for type 2 diabetes.",
        "expected_comparison_result": ComparisonResult.AMBIGUOUS.value,
        "expected_difference_type": DifferenceType.NONE.value,
        "expected_gap_status": GapStatus.REVIEW_REQUIRED.value,
        "expected_gate": HumanGate.G2.value,
        "expected_confidence": 0.0,
        "is_match": True,
        "is_boundary_case": False,
        "rationale": "Model failure or unparseable JSON safely caught and routed to G2.",
    },
    {
        "scenario_id": "CMP-13",
        "name": "Confirmed no-match finding requiring G3",
        "condition_type": "no_match_g3",
        "input_recommendation": "Administer Pembrolizumab 200mg IV every 3 weeks for metastatic non-small cell lung cancer.",
        "target_population": "Patients with metastatic NSCLC",
        "intervention": "Pembrolizumab 200mg IV every 3 weeks",
        "evidence_grade": "Grade A",
        "protocol_id": "PROT-DM-001",
        "protocol_version": "v1.0",
        "candidate_section_id": None,
        "candidate_section_heading": None,
        "candidate_section_text": None,
        "expected_comparison_result": ComparisonResult.NO_MATCH.value,
        "expected_difference_type": DifferenceType.NO_MATCH.value,
        "expected_gap_status": GapStatus.NO_MATCH.value,
        "expected_gate": HumanGate.G3.value,
        "expected_confidence": 1.0,
        "is_match": False,
        "is_boundary_case": False,
        "rationale": "Oncology recommendation has no semantic match in diabetes protocol (< 0.70 similarity) -> G3.",
    },
    {
        "scenario_id": "CMP-14",
        "name": "Low-similarity distinct specialty no-match",
        "condition_type": "low_similarity_no_match",
        "input_recommendation": "Prescribe Albuterol HFA 90mcg 2 puffs every 4 to 6 hours as needed for acute bronchospasm.",
        "target_population": "Patients with acute bronchospasm",
        "intervention": "Albuterol HFA 90mcg inhalation",
        "evidence_grade": "Grade A",
        "protocol_id": "PROT-DM-001",
        "protocol_version": "v1.0",
        "candidate_section_id": None,
        "candidate_section_heading": None,
        "candidate_section_text": None,
        "expected_comparison_result": ComparisonResult.NO_MATCH.value,
        "expected_difference_type": DifferenceType.NO_MATCH.value,
        "expected_gap_status": GapStatus.NO_MATCH.value,
        "expected_gate": HumanGate.G3.value,
        "expected_confidence": 1.0,
        "is_match": False,
        "is_boundary_case": False,
        "rationale": "Respiratory therapy has distant semantic similarity (< 0.40) -> G3.",
    },
    {
        "scenario_id": "CMP-15",
        "name": "Protocol version difference verification",
        "condition_type": "protocol_version_difference",
        "input_recommendation": "Metformin 500mg daily is first-line pharmacotherapy.",
        "target_population": "Adults with type 2 diabetes",
        "intervention": "Metformin 500mg daily",
        "evidence_grade": "Grade A",
        "protocol_id": "PROT-DM-001",
        "protocol_version": "v2.0",
        "candidate_section_id": "SEC-2",
        "candidate_section_heading": "First-Line Pharmacotherapy",
        "candidate_section_text": "Metformin 500mg once daily with evening meals is the preferred initial pharmacologic agent for type 2 diabetes.",
        "expected_comparison_result": ComparisonResult.GAP.value,
        "expected_difference_type": DifferenceType.OTHER_SUPPORTED_CHANGE.value,
        "expected_gap_status": GapStatus.MATCHED.value,
        "expected_gate": None,
        "expected_confidence": 0.93,
        "is_match": True,
        "is_boundary_case": False,
        "rationale": "Verifies historical protocol version v2.0 attribution and provenance tracking.",
    },
    {
        "scenario_id": "CMP-16",
        "name": "Provisional threshold boundary case",
        "condition_type": "threshold_boundary",
        "input_recommendation": "Adult patients with fasting glucose >= 125 mg/dL should be evaluated for glycemic control.",
        "target_population": "Adult patients with fasting glucose >= 125 mg/dL",
        "intervention": "Glycemic evaluation",
        "evidence_grade": "Grade B",
        "protocol_id": "PROT-DM-001",
        "protocol_version": "v1.0",
        "candidate_section_id": "SEC-1",
        "candidate_section_heading": "Screening and Diagnostic Criteria",
        "candidate_section_text": "Fasting plasma glucose >= 126 mg/dL or HbA1c >= 6.5% confirms diagnosis of type 2 diabetes mellitus.",
        "expected_comparison_result": ComparisonResult.GAP.value,
        "expected_difference_type": DifferenceType.THRESHOLD_CHANGE.value,
        "expected_gap_status": GapStatus.MATCHED.value,
        "expected_gate": None,
        "expected_confidence": 0.71,
        "is_match": True,
        "is_boundary_case": True,
        "rationale": "Score sits near 0.70 boundary (0.71) for empirical calibration observation.",
    },
]


class EvaluationCorpusService:
    """Manages the generation, discovery, and retrieval of evaluation datasets and expected results."""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = base_dir or Path("data/evaluation")
        self.extraction_dir = self.base_dir / "extraction"
        self.e2e_dir = self.base_dir / "end_to_end"
        self.comparison_dir = self.base_dir / "comparison"
        self.expected_dir = self.base_dir / "expected"
        self.reports_dir = self.base_dir / "reports"

    def ensure_directories(self) -> None:
        """Create standard evaluation directories if they do not exist."""
        for d in [
            self.extraction_dir,
            self.e2e_dir,
            self.comparison_dir,
            self.expected_dir,
            self.reports_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def generate_corpus_files(self) -> Tuple[int, int]:
        """Generate synthetic PDFs and ground-truth JSON files.

        Returns:
            Tuple of (extraction_pdfs_generated, expected_json_files_written).
        """
        self.ensure_directories()

        # 1. Generate 20 extraction / E2E PDFs
        pdf_count = 0
        for case in RAW_EXTRACTION_CASES:
            pdf_bytes = make_multipage_pdf_bytes(case["pages"])
            ext_path = self.extraction_dir / case["filename"]
            ext_path.write_bytes(pdf_bytes)

            e2e_path = self.e2e_dir / case["filename"]
            e2e_path.write_bytes(pdf_bytes)
            pdf_count += 1

        # 2. Write expected metadata JSON files
        ext_expected_path = self.expected_dir / "extraction_cases.json"
        with open(ext_expected_path, "w", encoding="utf-8") as f:
            json.dump(RAW_EXTRACTION_CASES, f, indent=2)

        cmp_expected_path = self.expected_dir / "comparison_cases.json"
        with open(cmp_expected_path, "w", encoding="utf-8") as f:
            json.dump(RAW_COMPARISON_SCENARIOS, f, indent=2)

        logger.info(
            "Evaluation corpus ready: %d PDFs in %s, metadata in %s",
            pdf_count,
            self.extraction_dir,
            self.expected_dir,
        )
        return pdf_count, 2

    def get_extraction_cases(self) -> List[Dict[str, Any]]:
        """Return the 20 structured extraction cases."""
        return list(RAW_EXTRACTION_CASES)

    def get_comparison_scenarios(self) -> List[Dict[str, Any]]:
        """Return the 16 structured comparison scenarios."""
        return list(RAW_COMPARISON_SCENARIOS)
