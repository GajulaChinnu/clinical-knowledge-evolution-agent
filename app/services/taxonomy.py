"""Controlled clinical taxonomy: departments, pathways, treatment classes and treatments.

Every department, pathway and treatment value CKEA stores must resolve through this module.
Matching is deterministic (word-boundary, case-insensitive, synonym-aware).
"""

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import yaml

DEFAULT_TAXONOMY_PATH = Path("./config/taxonomy.yaml")


class TaxonomyError(ValueError):
    """Raised when the taxonomy file is malformed or a value is not in the controlled lists."""


@dataclass(frozen=True)
class Pathway:
    id: str
    name: str
    department: str
    treatments: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Department:
    id: str
    name: str
    synonyms: Tuple[str, ...]
    pathways: Tuple[Pathway, ...]


@dataclass(frozen=True)
class TreatmentClass:
    id: str
    name: str
    synonyms: Tuple[str, ...]


@dataclass(frozen=True)
class Treatment:
    id: str
    name: str
    class_id: Optional[str]
    synonyms: Tuple[str, ...]
    departments: Tuple[str, ...]


@dataclass(frozen=True)
class ProtocolMetadata:
    department: str
    pathways: Tuple[str, ...] = ()
    treatments: Tuple[str, ...] = ()
    owner: Optional[str] = None
    effective_date: Optional[str] = None


@dataclass
class TreatmentMention:
    """Treatments and treatment classes found in a piece of text."""
    treatments: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.treatments and not self.classes


def _norm(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _term_pattern(term: str) -> "re.Pattern[str]":
    # Word boundaries that also work for terms with hyphens/digits (e.g. "sglt2 inhibitor").
    return re.compile(r"(?<![a-z0-9])" + re.escape(_norm(term)) + r"(?![a-z0-9])")


class Taxonomy:
    """Loaded, validated taxonomy with deterministic lookup and matching helpers."""

    def __init__(
        self,
        version: str,
        departments: List[Department],
        classes: List[TreatmentClass],
        treatments: List[Treatment],
        legacy_protocol_metadata: Dict[str, ProtocolMetadata],
    ) -> None:
        self.version = version
        self.departments: Dict[str, Department] = {d.id: d for d in departments}
        self.classes: Dict[str, TreatmentClass] = {c.id: c for c in classes}
        self.treatments: Dict[str, Treatment] = {t.id: t for t in treatments}
        self.legacy_protocol_metadata = legacy_protocol_metadata
        self.pathways: Dict[str, Pathway] = {p.id: p for d in departments for p in d.pathways}

        self._department_terms: List[Tuple[str, str]] = []
        for d in departments:
            for term in (d.id, d.name, *d.synonyms):
                self._department_terms.append((_norm(term), d.id))

        # Longest terms first so "sodium bicarbonate" wins over shorter overlaps.
        treatment_terms = [(term, t.id) for t in treatments for term in (t.name, t.id.replace("-", " "), *t.synonyms)]
        class_terms = [(term, c.id) for c in classes for term in (c.name, *c.synonyms)]
        self._treatment_patterns = [
            (_term_pattern(term), tid) for term, tid in sorted(treatment_terms, key=lambda x: -len(x[0]))
        ]
        self._class_patterns = [
            (_term_pattern(term), cid) for term, cid in sorted(class_terms, key=lambda x: -len(x[0]))
        ]

    # ---- departments -------------------------------------------------------------
    def resolve_department(self, value: Optional[str]) -> str:
        """Map an id, name or synonym to a department id. Raises TaxonomyError if unknown."""
        key = _norm(value or "")
        for term, dept_id in self._department_terms:
            if key == term:
                return dept_id
        raise TaxonomyError(f"Unknown department '{value}'. Allowed: {', '.join(sorted(self.departments))}.")

    def department_name(self, dept_id: str) -> str:
        dept = self.departments.get(dept_id)
        return dept.name if dept else dept_id

    def validate_departments(self, values: Iterable[str]) -> List[str]:
        return sorted({self.resolve_department(v) for v in values})

    def validate_pathways(self, values: Iterable[str]) -> List[str]:
        unknown = [v for v in values if v not in self.pathways]
        if unknown:
            raise TaxonomyError(f"Unknown pathway(s): {', '.join(unknown)}")
        return sorted(set(values))

    def pathways_for_departments(self, dept_ids: Iterable[str]) -> List[str]:
        wanted = set(dept_ids)
        return sorted(p.id for p in self.pathways.values() if p.department in wanted)

    def pathways_for(self, treatments: Iterable[str], dept_ids: Iterable[str] = ()) -> List[str]:
        """Pathways that use any of the treatments, optionally limited to departments."""
        wanted_t, wanted_d = set(treatments), set(dept_ids)
        return sorted(
            p.id for p in self.pathways.values()
            if wanted_t & set(p.treatments) and (not wanted_d or p.department in wanted_d)
        )

    def departments_for_treatments(self, treatments: Iterable[str]) -> List[str]:
        return sorted({d for t in treatments if t in self.treatments for d in self.treatments[t].departments})

    # ---- treatments ----------------------------------------------------------------
    def validate_treatments(self, values: Iterable[str]) -> List[str]:
        unknown = [v for v in values if v not in self.treatments]
        if unknown:
            raise TaxonomyError(f"Unknown treatment(s): {', '.join(unknown)}")
        return sorted(set(values))

    def find_treatments(self, text: str) -> TreatmentMention:
        """Return treatment ids and class ids explicitly mentioned in text."""
        haystack = _norm(text)
        found_t: List[str] = []
        found_c: List[str] = []
        for pattern, tid in self._treatment_patterns:
            if tid not in found_t and pattern.search(haystack):
                found_t.append(tid)
        for pattern, cid in self._class_patterns:
            if cid not in found_c and pattern.search(haystack):
                found_c.append(cid)
        return TreatmentMention(treatments=sorted(found_t), classes=sorted(found_c))

    def treatment_terms(self, treatment_id: str) -> List[str]:
        t = self.treatments[treatment_id]
        return [t.name, *t.synonyms]

    def class_of(self, treatment_id: str) -> Optional[str]:
        t = self.treatments.get(treatment_id)
        return t.class_id if t else None

    def members_of_class(self, class_id: str) -> List[str]:
        return sorted(t.id for t in self.treatments.values() if t.class_id == class_id)

    def treatment_name(self, treatment_id: str) -> str:
        t = self.treatments.get(treatment_id)
        return t.name if t else treatment_id

    # ---- protocols -----------------------------------------------------------------
    def legacy_metadata_for(self, protocol_id: str) -> Optional[ProtocolMetadata]:
        return self.legacy_protocol_metadata.get(protocol_id)


def _require(entry: dict, key: str, where: str):
    value = entry.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise TaxonomyError(f"{where}: missing '{key}'")
    return value


def load_taxonomy(path: Union[str, Path] = DEFAULT_TAXONOMY_PATH) -> Taxonomy:
    """Load and validate the taxonomy YAML."""
    path = Path(path)
    if not path.exists():
        raise TaxonomyError(f"Taxonomy file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise TaxonomyError(f"Taxonomy '{path}' is not valid YAML: {e}") from e

    departments: List[Department] = []
    seen_pathways: Set[str] = set()
    for i, d in enumerate(data.get("departments") or []):
        dept_id = _require(d, "id", f"departments[{i}]")
        pathways = []
        for j, p in enumerate(d.get("pathways") or []):
            pid = _require(p, "id", f"departments[{i}].pathways[{j}]")
            if pid in seen_pathways:
                raise TaxonomyError(f"Duplicate pathway id '{pid}'")
            seen_pathways.add(pid)
            pathways.append(Pathway(pid, _require(p, "name", f"pathway {pid}"), dept_id,
                                    tuple(p.get("treatments") or ())))
        departments.append(Department(dept_id, _require(d, "name", f"department {dept_id}"),
                                      tuple(d.get("synonyms") or ()), tuple(pathways)))
    if not departments:
        raise TaxonomyError("Taxonomy defines no departments.")
    dept_ids = {d.id for d in departments}
    if len(dept_ids) != len(departments):
        raise TaxonomyError("Duplicate department ids in taxonomy.")

    classes = [
        TreatmentClass(_require(c, "id", f"treatment_classes[{i}]"), _require(c, "name", f"class {i}"),
                       tuple(c.get("synonyms") or ()))
        for i, c in enumerate(data.get("treatment_classes") or [])
    ]
    class_ids = {c.id for c in classes}

    treatments: List[Treatment] = []
    for i, t in enumerate(data.get("treatments") or []):
        tid = _require(t, "id", f"treatments[{i}]")
        cls = t.get("class")
        if cls and cls not in class_ids:
            raise TaxonomyError(f"Treatment '{tid}' references unknown class '{cls}'")
        depts = tuple(t.get("departments") or ())
        bad = [d for d in depts if d not in dept_ids]
        if bad:
            raise TaxonomyError(f"Treatment '{tid}' references unknown department(s) {bad}")
        treatments.append(Treatment(tid, _require(t, "name", f"treatment {tid}"), cls,
                                    tuple(t.get("synonyms") or ()), depts))
    if len({t.id for t in treatments}) != len(treatments):
        raise TaxonomyError("Duplicate treatment ids in taxonomy.")
    treatment_ids = {t.id for t in treatments}
    for d in departments:
        for p in d.pathways:
            unknown = [t for t in p.treatments if t not in treatment_ids]
            if unknown:
                raise TaxonomyError(f"Pathway '{p.id}' references unknown treatment(s) {unknown}")

    legacy: Dict[str, ProtocolMetadata] = {}
    for protocol_id, meta in (data.get("legacy_protocol_metadata") or {}).items():
        dept = _require(meta, "department", f"legacy_protocol_metadata.{protocol_id}")
        if dept not in dept_ids:
            raise TaxonomyError(f"Legacy protocol '{protocol_id}' has unknown department '{dept}'")
        legacy[protocol_id] = ProtocolMetadata(
            department=dept,
            pathways=tuple(meta.get("pathways") or ()),
            treatments=tuple(meta.get("treatments") or ()),
            owner=meta.get("owner"),
            effective_date=str(meta["effective_date"]) if meta.get("effective_date") else None,
        )

    return Taxonomy(str(data.get("version") or "1.0"), departments, classes, treatments, legacy)


@lru_cache(maxsize=4)
def get_taxonomy(path: str = str(DEFAULT_TAXONOMY_PATH)) -> Taxonomy:
    """Process-wide cached taxonomy."""
    return load_taxonomy(path)
