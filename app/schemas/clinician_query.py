"""Clinician Treatment Check input schema, patient-identifier guard and treatment normalisation.

The guard runs before anything else. It reports *what kind* of identifier was found, never the
value itself, and the raw free-text context is never persisted.
"""

from dataclasses import dataclass, field
import re
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.services.clinical_statements import ParsedStatement, parse_statement
from app.services.taxonomy import Taxonomy, TaxonomyError

AgeBand = Literal["18-39", "40-64", "65-79", "80+"]
EgfrBand = Literal[">=60", "45-59", "30-44", "15-29", "<15"]
EGFR_BAND_RANGES = {">=60": (60.0, 200.0), "45-59": (45.0, 59.99), "30-44": (30.0, 44.99),
                    "15-29": (15.0, 29.99), "<15": (0.0, 14.99)}
AGE_BAND_RANGES = {"18-39": (18, 39), "40-64": (40, 64), "65-79": (65, 79), "80+": (80, 130)}

MAX_FREE_TEXT = 200


class PatientIdentifierError(ValueError):
    """Raised when the query contains something that looks like a patient identifier."""

    def __init__(self, findings: List[str]) -> None:
        self.findings = findings
        super().__init__(
            "Patient identifiers are not allowed in a treatment check ("
            + "; ".join(findings)
            + "). Remove them and describe the patient only with the de-identified fields."
        )


_IDENTIFIER_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "email address"),
    (re.compile(r"\b(?:mrn|nhs|hospital|patient|record|chart)\s*(?:no\.?|number|num|id|#)?\s*[:#]?\s*[A-Z0-9-]{4,}", re.I), "medical record / patient number"),
    (re.compile(r"\b\d{3}[\s-]?\d{3}[\s-]?\d{4}\b|\b0\d{2,4}[\s-]?\d{3,4}[\s-]?\d{3,4}\b"), "phone or national health number"),
    (re.compile(r"\+\d[\d\s-]{7,}\d"), "phone number"),
    (re.compile(r"\b\d{6,}\b"), "long identifier number"),
    (re.compile(r"\b(?:dob|d\.o\.b\.?|date of birth|born on|born)\b", re.I), "date of birth"),
    (re.compile(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b"), "date"),
    (re.compile(r"\b\d+\s+[A-Za-z]+\s+(?:street|st|road|rd|avenue|ave|lane|ln|drive|dr|close|way)\b", re.I), "street address"),
    (re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b"), "postcode"),
    # Titles and capitalised names are matched case-sensitively so clinical terms such as
    # "MS" (multiple sclerosis) or "patient is pregnant" are not mistaken for names.
    (re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Mx|MR|MRS)\.?\s+[A-Z][a-z]+"), "personal name"),
    (re.compile(r"\b(?i:patient|name)\s*(?i:is|:|called|named)\s*[A-Z][a-z]+"), "personal name"),
]


def find_patient_identifiers(*texts: Optional[str]) -> List[str]:
    """Kinds of patient identifiers found in the given texts (values are never returned)."""
    kinds: List[str] = []
    for text in texts:
        if not text:
            continue
        for pattern, kind in _IDENTIFIER_PATTERNS:
            if pattern.search(text) and kind not in kinds:
                kinds.append(kind)
    return kinds


class PatientContext(BaseModel):
    """De-identified patient context. Only bands and short clinical terms are accepted."""

    model_config = ConfigDict(extra="forbid")

    age_band: Optional[AgeBand] = None
    pregnancy: Optional[bool] = None
    egfr_band: Optional[EgfrBand] = None
    comorbidities: List[str] = Field(default_factory=list, max_length=10)
    current_medications: List[str] = Field(default_factory=list, max_length=15)

    @field_validator("comorbidities", "current_medications")
    @classmethod
    def _short_terms(cls, values: List[str]) -> List[str]:
        cleaned = [" ".join(str(v).split()) for v in values if str(v).strip()]
        for v in cleaned:
            if len(v) > 80:
                raise ValueError("Context terms must be short clinical terms (max 80 characters).")
        return cleaned

    def summary(self) -> dict:
        """Non-identifying summary suitable for the audit log."""
        return {
            "age_band": self.age_band,
            "pregnancy": self.pregnancy,
            "egfr_band": self.egfr_band,
            "comorbidity_count": len(self.comorbidities),
            "medication_count": len(self.current_medications),
        }


class ClinicianQueryInput(BaseModel):
    """Raw clinician input as submitted from the UI or API."""

    model_config = ConfigDict(extra="forbid")

    department: str = Field(..., min_length=1, max_length=80)
    treatment: str = Field(..., min_length=2, max_length=MAX_FREE_TEXT)
    condition: Optional[str] = Field(default=None, max_length=MAX_FREE_TEXT)
    context: PatientContext = Field(default_factory=PatientContext)


@dataclass
class PlannedTreatment:
    """The clinician's plan, parsed with the same rules used for guidance statements."""
    text: str
    treatments: List[str]
    treatment_classes: List[str]
    parsed: ParsedStatement


@dataclass
class ClinicianQuery:
    """Validated, normalised query that the six agents operate on."""
    department: str
    department_name: str
    plan: PlannedTreatment
    condition: Optional[str]
    context: PatientContext
    medication_treatments: List[str] = field(default_factory=list)
    comorbidity_terms: List[str] = field(default_factory=list)

    @property
    def treatment_ids(self) -> List[str]:
        return self.plan.treatments


def build_clinician_query(raw: ClinicianQueryInput, taxonomy: Taxonomy) -> ClinicianQuery:
    """Guard, validate and normalise a clinician query.

    Raises:
        PatientIdentifierError: if any free-text field looks like it contains an identifier.
        TaxonomyError: if the department is not in the controlled list.
    """
    findings = find_patient_identifiers(
        raw.treatment, raw.condition, *raw.context.comorbidities, *raw.context.current_medications
    )
    if findings:
        raise PatientIdentifierError(findings)

    department = taxonomy.resolve_department(raw.department)
    treatment_text = " ".join(raw.treatment.split())
    parsed = parse_statement(treatment_text, taxonomy)
    medications = sorted({t for m in raw.context.current_medications for t in taxonomy.find_treatments(m).treatments})
    return ClinicianQuery(
        department=department,
        department_name=taxonomy.department_name(department),
        plan=PlannedTreatment(treatment_text, parsed.treatments, parsed.treatment_classes, parsed),
        condition=" ".join(raw.condition.split()) if raw.condition else None,
        context=raw.context,
        medication_treatments=medications,
        comorbidity_terms=[c.lower() for c in raw.context.comorbidities],
    )


__all__ = [
    "AGE_BAND_RANGES", "ClinicianQuery", "ClinicianQueryInput", "EGFR_BAND_RANGES", "PatientContext",
    "PatientIdentifierError", "PlannedTreatment", "TaxonomyError", "build_clinician_query",
    "find_patient_identifiers",
]
