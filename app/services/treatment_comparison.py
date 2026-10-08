"""Deterministic three-way comparison for a clinician Treatment Check.

(a) planned treatment vs the latest recommendations, (b) previous vs latest version of those
recommendations, (c) institutional protocol vs the latest guidance. All inputs are verbatim
GuidanceStatements, so every conclusion points at an exact, verifiable excerpt.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from app.models.entities import GuidanceChange, GuidanceStatement, IngestedDocument
from app.schemas.clinician_query import EGFR_BAND_RANGES, ClinicianQuery
from app.schemas.protocol import ProtocolDocument
from app.schemas.treatment_check import AttributeDiff, Citation, ProtocolItem, ProtocolPosition
from app.services.clinical_statements import (
    ParsedStatement,
    attribute_differences,
    parse_statement,
    similarity,
    skeleton,
    split_statements,
)
from app.services.taxonomy import Taxonomy

GUIDANCE_SOURCE_TYPES = ("guideline", "safety_notice")
_INITIATION = re.compile(r"^(?:start|give|offer|initiate|prescribe|use|consider)\b", re.I)
_STOPWORDS = {
    "a", "an", "the", "of", "in", "with", "and", "or", "to", "for", "on", "at", "by", "is", "are",
    "patient", "patients", "adult", "adults", "who", "has", "have", "when", "if",
}
ALIGN_THRESHOLD = 0.45


# ------------------------------------------------------------------------------ citations
@lru_cache(maxsize=256)
def _artifact_text(path: str, mtime: float) -> str:
    return Path(path).read_text(encoding="utf-8")


def verify_statement(stmt: GuidanceStatement, doc: IngestedDocument) -> bool:
    """True if the statement is still present verbatim at its offsets in the stored artifact."""
    path = Path(doc.source_path)
    if not path.exists() or path.suffix.lower() not in (".md", ".txt"):
        # PDF artifacts: fall back to whitespace-normalised containment in extracted text.
        try:
            from app.services.source_documents import load_source_text

            text = " ".join(load_source_text(path).split())
            return " ".join(stmt.verbatim_text.split()) in text
        except (OSError, ValueError):
            return False
    text = _artifact_text(str(path), path.stat().st_mtime)
    if stmt.char_start is None or stmt.char_end is None:
        return stmt.verbatim_text in text
    return text[stmt.char_start:stmt.char_end] == stmt.verbatim_text


def build_citation(stmt: GuidanceStatement, doc: IngestedDocument) -> Citation:
    meta = doc.doc_metadata or {}
    return Citation(
        statement_id=stmt.id,
        document_id=doc.id,
        source_identity=stmt.source_identity,
        watchlist_id=stmt.watchlist_id,
        source_title=meta.get("title"),
        source_type=stmt.source_type,
        publisher=meta.get("publisher"),
        version=stmt.publisher_version,
        published_date=stmt.published_date,
        section_heading=stmt.section_heading,
        char_start=stmt.char_start,
        char_end=stmt.char_end,
        excerpt=stmt.verbatim_text,
        verified=verify_statement(stmt, doc),
    )


def to_diffs(diffs) -> List[AttributeDiff]:
    return [AttributeDiff(kind=d.kind, before=d.before, after=d.after) for d in diffs]


# ------------------------------------------------------------------------------ plan vs statement
def plan_differences(plan: ParsedStatement, stmt: ParsedStatement) -> Tuple[List[AttributeDiff], bool]:
    """Differences on attributes present on BOTH sides; second value says whether anything was comparable."""
    diffs: List[AttributeDiff] = []
    compared = False
    if plan.doses and stmt.doses:
        compared = True
        plan_vals = {(d.value, d.unit) for d in plan.doses}
        stmt_vals = {(d.value, d.unit) for d in stmt.doses}
        if not plan_vals <= stmt_vals:
            fmt = lambda vals: ", ".join(f"{v:g} {u}" for v, u in sorted(vals))
            diffs.append(AttributeDiff(kind="dose", before=fmt(plan_vals), after=fmt(stmt_vals)))
    for attr in ("frequency", "interval", "duration", "route"):
        a, b = getattr(plan, attr), getattr(stmt, attr)
        if a and b:
            compared = True
            if a != b:
                fmt = lambda v: v if isinstance(v, str) else f"{v[0]:g} {v[1]}(s)"
                diffs.append(AttributeDiff(kind=attr, before=fmt(a), after=fmt(b)))
    return diffs, compared


def _band(query: ClinicianQuery) -> Optional[Tuple[float, float]]:
    return EGFR_BAND_RANGES.get(query.context.egfr_band) if query.context.egfr_band else None


def egfr_condition(parsed: ParsedStatement, query: ClinicianQuery) -> Tuple[Optional[str], str]:
    """Does the statement's eGFR condition hold for the patient's eGFR band?

    Returns (status, basis) where status is applies / not_applicable / unknown, or None when the
    statement has no eGFR condition.
    """
    egfr = [t for t in parsed.thresholds if t.measure == "egfr"]
    if not egfr:
        return None, ""
    band = _band(query)
    if band is None:
        return "unknown", "eGFR band not provided"
    lo, hi = band
    low_bound = next((t.value for t in egfr if t.comparator in (">=", ">", "range_low")), None)
    high_bound = next((t.value for t in egfr if t.comparator in ("<", "<=", "range_high")), None)
    strict_high = any(t.comparator == "<" for t in egfr)
    cond_lo = low_bound if low_bound is not None else float("-inf")
    cond_hi = high_bound if high_bound is not None else float("inf")
    if strict_high and high_bound is not None:
        cond_hi = high_bound - 0.01
    if cond_lo <= lo and hi <= cond_hi:
        return "applies", f"patient eGFR band {query.context.egfr_band} is within the stated eGFR range"
    if hi < cond_lo or lo > cond_hi:
        return "not_applicable", f"patient eGFR band {query.context.egfr_band} is outside the stated eGFR range"
    return "unknown", f"patient eGFR band {query.context.egfr_band} straddles the stated eGFR threshold"


def _context_phrases(query: ClinicianQuery) -> List[str]:
    phrases: List[str] = []
    for term in [*query.context.comorbidities, query.condition or ""]:
        words = [w for w in re.findall(r"[a-z0-9-]+", term.lower())]
        for n in (4, 3, 2):
            for i in range(len(words) - n + 1):
                gram = words[i:i + n]
                if any(len(w) >= 4 and w not in _STOPWORDS for w in gram):
                    phrases.append(" ".join(gram))
        phrases += [w for w in words if len(w) >= 6 and w not in _STOPWORDS]
    return phrases


def contraindication_applicability(
    parsed: ParsedStatement, query: ClinicianQuery, taxonomy: Taxonomy
) -> Tuple[str, str]:
    """Whether a contraindication / restriction applies to this patient, from de-identified context.

    Returns (relation, basis) with relation applies / not_applicable / check_applicability.
    Never guesses: anything the context cannot settle is check_applicability.
    """
    text = parsed.text.lower()
    plan_classes = {taxonomy.class_of(t) for t in query.treatment_ids} - {None}
    plan_family = set(query.treatment_ids) | {m for c in plan_classes for m in taxonomy.members_of_class(c)}

    # 1. Interaction with another medicine the patient takes
    other_drugs = set(parsed.treatments) - plan_family
    other_classes = set(parsed.treatment_classes) - plan_classes
    if other_drugs or other_classes:
        meds = set(query.medication_treatments)
        med_classes = {taxonomy.class_of(m) for m in meds} - {None}
        names = ", ".join(sorted(taxonomy.treatment_name(t) for t in other_drugs) or
                          sorted(taxonomy.classes[c].name for c in other_classes if c in taxonomy.classes))
        if meds & other_drugs or med_classes & other_classes:
            return "applies", f"patient's current medicines include {names}"
        if query.context.current_medications:
            return "not_applicable", f"{names} is not in the current medication list provided"
        return "check_applicability", f"check whether the patient takes {names}"

    # 2. eGFR condition (plus any other condition the statement adds, e.g. recent contrast)
    egfr_status, egfr_basis = egfr_condition(parsed, query)
    extra_conditions = [t for t in parsed.thresholds if t.measure not in ("egfr",)]
    if egfr_status == "not_applicable":
        return "not_applicable", egfr_basis
    if egfr_status == "applies" and not extra_conditions:
        return "applies", egfr_basis

    # 3. Route-specific restriction
    if parsed.route:
        plan_route = query.plan.parsed.route
        if plan_route and plan_route != parsed.route:
            return "not_applicable", f"restriction concerns {parsed.route} use; the plan is {plan_route}"
        if plan_route == parsed.route:
            return "applies", f"the plan uses the {parsed.route} route covered by this restriction"
        return "check_applicability", f"restriction concerns {parsed.route} use; route not stated in the plan"

    # 4. Pregnancy
    if "pregnan" in text:
        if query.context.pregnancy is True:
            return "applies", "patient is pregnant"
        if query.context.pregnancy is False:
            return "not_applicable", "patient is not pregnant"
        return "check_applicability", "pregnancy status not provided"

    # 5. Conditions named in the de-identified context (comorbidities, indication)
    for phrase in _context_phrases(query):
        if phrase in text:
            basis = f"context mentions '{phrase}'"
            if egfr_status == "unknown":
                return "check_applicability", f"{basis}; {egfr_basis}"
            return "applies", basis
    if egfr_status == "unknown":
        return "check_applicability", egfr_basis
    return "check_applicability", "confirm whether the stated condition applies to this patient"


def initiation_score(parsed: ParsedStatement, plan: ParsedStatement) -> float:
    score = 2.0 if _INITIATION.search(parsed.text.strip()) else 0.0
    if parsed.route and parsed.route == plan.route:
        score += 1.0
    return score + similarity(skeleton(plan.text), skeleton(parsed.text))


# ------------------------------------------------------------------------------ protocol comparison
@dataclass
class StatementRef:
    stmt: GuidanceStatement
    doc: IngestedDocument
    parsed: ParsedStatement


def _compatible(a: ParsedStatement, b: ParsedStatement) -> bool:
    if a.statement_type == "contraindication" or b.statement_type == "contraindication":
        return a.statement_type == b.statement_type
    return True


def compare_protocol(
    protocol: ProtocolDocument,
    query: ClinicianQuery,
    latest: Sequence[StatementRef],
    changes: Sequence[Tuple[GuidanceChange, Optional[StatementRef], Optional[StatementRef]]],
    taxonomy: Taxonomy,
) -> ProtocolPosition:
    """Compare protocol sections about the planned treatment with the latest guidance."""
    plan_t = set(query.treatment_ids)
    proto_statements: List[Tuple[str, str, str, ParsedStatement]] = []
    for sec in protocol.sections:
        for span in split_statements(sec.section_text):
            parsed = parse_statement(span.text, taxonomy)
            if plan_t & set(parsed.treatments):
                proto_statements.append((sec.section_id, sec.section_heading, span.text, parsed))

    position = ProtocolPosition(
        protocol_id=protocol.protocol_id, protocol_version=protocol.protocol_version,
        title=protocol.title, effective_date=protocol.effective_date, status="not_covered",
    )
    if not proto_statements:
        return position

    def same_treatment(p: ParsedStatement, ref: Optional[StatementRef]) -> bool:
        return ref is not None and bool(plan_t & set(ref.parsed.treatments) & set(p.treatments))

    items: List[ProtocolItem] = []
    for sec_id, heading, text, p in proto_statements:
        item: Optional[ProtocolItem] = None
        # 1. Does the protocol still follow a superseded or withdrawn version?
        for change, old, new in changes:
            if not same_treatment(p, old) or not _compatible(p, old.parsed):
                continue
            if similarity(skeleton(text), skeleton(old.stmt.verbatim_text)) < ALIGN_THRESHOLD:
                continue
            if attribute_differences(p, old.parsed):
                continue
            new_differs = new is None or new.stmt.statement_type == "withdrawal" or bool(attribute_differences(p, new.parsed))
            if new_differs:
                latest_cit = build_citation(new.stmt, new.doc) if new else None
                diffs = to_diffs(attribute_differences(p, new.parsed)) if (new and new.stmt.statement_type != "withdrawal") else []
                item = ProtocolItem(
                    section_id=sec_id, section_heading=heading, section_text=text, status="out_of_date",
                    reason=(f"Protocol follows version {old.stmt.publisher_version} of "
                            f"{(old.doc.doc_metadata or {}).get('title', old.stmt.source_identity)}; "
                            f"version {new.stmt.publisher_version if new else 'latest'} changed this "
                            f"({change.change_category.replace('_', ' ')})."),
                    latest_citation=latest_cit,
                    previous_citation=build_citation(old.stmt, old.doc),
                    differences=diffs,
                )
                break
        # 2. Otherwise compare with the closest latest statement.
        if item is None:
            candidates = [r for r in latest if same_treatment(p, r) and _compatible(p, r.parsed)
                          and r.stmt.statement_type != "withdrawal"]
            best = max(candidates, key=lambda r: similarity(skeleton(text), skeleton(r.stmt.verbatim_text)), default=None)
            if best is not None and similarity(skeleton(text), skeleton(best.stmt.verbatim_text)) >= ALIGN_THRESHOLD:
                diffs = to_diffs(attribute_differences(p, best.parsed))
                item = ProtocolItem(
                    section_id=sec_id, section_heading=heading, section_text=text,
                    status="out_of_date" if diffs else "aligned",
                    reason=("Protocol differs from the latest guidance." if diffs
                            else "Protocol matches the latest guidance."),
                    latest_citation=build_citation(best.stmt, best.doc),
                    differences=diffs,
                )
        if item is not None:
            items.append(item)

    # 3. Contraindications added in the latest version that the protocol does not state.
    proto_ci = [text for _, _, text, p in proto_statements if p.statement_type == "contraindication"]
    for change, old, new in changes:
        if change.change_category != "contraindication_added" or new is None:
            continue
        if not plan_t & set(new.parsed.treatments):
            continue
        if any(similarity(skeleton(t), skeleton(new.stmt.verbatim_text)) >= 0.6 for t in proto_ci):
            continue
        items.append(ProtocolItem(
            section_id="-", section_heading="(not in protocol)", section_text="",
            status="missing_from_protocol",
            reason=f"Contraindication added in version {new.stmt.publisher_version} is not stated in the protocol.",
            latest_citation=build_citation(new.stmt, new.doc),
        ))

    position.items = items
    if any(i.status in ("out_of_date", "missing_from_protocol") for i in items):
        position.status = "out_of_date"
    elif items:
        position.status = "aligned"
    return position


def mentions_plan(parsed: ParsedStatement, query: ClinicianQuery, taxonomy: Taxonomy) -> Tuple[bool, bool]:
    """(direct mention of the planned treatment, mention via its treatment class)."""
    direct = bool(set(parsed.treatments) & set(query.treatment_ids))
    plan_classes = {taxonomy.class_of(t) for t in query.treatment_ids} - {None}
    via_class = bool(set(parsed.treatment_classes) & plan_classes)
    return direct, via_class


def unique(items: Iterable[str]) -> List[str]:
    return sorted(set(items))
