"""Deterministic clinical statement parsing.

Splits source text into verbatim statements (with character offsets) and parses each into
structured attributes: treatments, doses, frequency, interval, duration, route, thresholds,
statement type and evidence level. Used for source statements, protocol sections and the
clinician's planned treatment, so all three are compared on the same footing.

Nothing here paraphrases text: every statement is an exact substring of its source.
"""

from dataclasses import asdict, dataclass, field
import difflib
import re
from typing import Iterable, List, Optional, Sequence, Tuple

from app.services.taxonomy import Taxonomy

# ------------------------------------------------------------------------------ patterns
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_EVIDENCE = re.compile(r"\(\s*evidence level\s+([A-D])\s*\)", re.IGNORECASE)

# Dose: number + unit, excluding rate/concentration units such as "mL/min" or "mg/mmol".
_DOSE = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)\s*(mg|g|mcg|micrograms?|µg|ml|units?)(?![\w/])(?!\s*/)",
    re.IGNORECASE,
)
_FREQUENCIES = [
    (re.compile(r"\bfour times (?:a |per )?day\b|\bfour times daily\b|\bqds\b|\bqid\b", re.I), "four_times_daily"),
    (re.compile(r"\bthree times (?:a |per )?day\b|\bthree times daily\b|\btds\b|\btid\b", re.I), "three_times_daily"),
    (re.compile(r"\btwice (?:a |per )?day\b|\btwice daily\b|\bbd\b|\bbid\b", re.I), "twice_daily"),
    (re.compile(r"\bonce (?:a |per )?day\b|\bonce daily\b|\bod\b|\bqd\b", re.I), "once_daily"),
    (re.compile(r"\bat night\b|\bnocte\b", re.I), "at_night"),
]
_EVERY = re.compile(r"\bevery\s+(\d+)\s*(minutes?|hours?|days?|weeks?)\b", re.I)
_INTERVAL = re.compile(r"\bafter\s+(\d+)\s*(minutes?|hours?)\b", re.I)
_DURATION = re.compile(r"\bfor\s+(\d+)\s*(days?|weeks?|months?)\b", re.I)
# Route words match in any case; abbreviations (IM, IV, SC, PO) only in capitals.
_ROUTES = [
    (re.compile(r"\b(?i:intramuscular(?:ly)?)\b|\bIM\b"), "intramuscular"),
    (re.compile(r"\b(?i:intravenous(?:ly)?)\b|\bIV\b"), "intravenous"),
    (re.compile(r"\b(?i:subcutaneous(?:ly)?)\b|\bSC\b|\bs/c\b"), "subcutaneous"),
    (re.compile(r"\b(?i:oral(?:ly)?|by mouth)\b|\bPO\b"), "oral"),
]

_WITHDRAWAL = re.compile(r"\bwithdrawn\b|\bno longer recommended\b", re.I)
_CONTRAINDICATION = re.compile(
    r"\bcontraindicated\b|\bmust not\b|\bshould not be (?:used|given|combined|prescribed)\b|"
    r"\bdo not (?:give|use|prescribe|combine)\b|\bavoid\b|\bis not used\b|\bnot recommended\b",
    re.I,
)
_SAFETY = re.compile(r"\brisk of\b|\bwarning\b|\bcaution\b|\baccumulates\b|\bbleeding risk\b", re.I)
_RECOMMENDATION = re.compile(
    r"^(?:start|give|offer|consider|use|increase|reduce|repeat|check|continue|add|prescribe|initiate|treat|stop|discontinue|monitor)\b|"
    r"\b(?:should|may be|is recommended|are recommended|is preferred|is the preferred)\b",
    re.I,
)
_EVIDENCE_FINDING = re.compile(r"\b(?:trial|randomi[sz]ed|case series|cohort|reduced|was reported|enrolled|tolerated)\b", re.I)

# Threshold measures recognised in text (longest first); value tokens near them are thresholds.
_MEASURES = [
    ("albumin-to-creatinine ratio", "acr"), ("acr", "acr"), ("cha2ds2-vasc score", "cha2ds2_vasc"),
    ("cha2ds2-vasc", "cha2ds2_vasc"), ("egfr", "egfr"), ("serum creatinine", "creatinine"),
    ("creatinine", "creatinine"), ("serum bicarbonate", "bicarbonate"), ("bicarbonate", "bicarbonate"),
    ("hba1c", "hba1c"), ("potassium", "potassium"), ("curb-65 score", "curb65"), ("age", "age"),
    ("weight", "weight"), ("within", "time_window"),
]
_MEASURE_PATTERNS = [
    (re.compile(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])"), key) for term, key in _MEASURES
]
_NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.]*\d)")


@dataclass
class Dose:
    value: float
    unit: str


@dataclass
class Threshold:
    measure: str
    value: float
    comparator: str = "="  # "<", "<=", ">=", ">", "=", "range_low", "range_high"


@dataclass
class ParsedStatement:
    text: str
    treatments: List[str] = field(default_factory=list)
    treatment_classes: List[str] = field(default_factory=list)
    doses: List[Dose] = field(default_factory=list)
    frequency: Optional[str] = None
    interval: Optional[Tuple[float, str]] = None
    duration: Optional[Tuple[float, str]] = None
    route: Optional[str] = None
    thresholds: List[Threshold] = field(default_factory=list)
    statement_type: str = "informational"
    evidence_level: Optional[str] = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("text")
        return data

    @property
    def has_dosing(self) -> bool:
        return bool(self.doses or self.frequency or self.interval or self.duration)


@dataclass
class StatementSpan:
    text: str
    char_start: int
    char_end: int


# ------------------------------------------------------------------------------ splitting
def split_statements(text: str, base_offset: int = 0) -> List[StatementSpan]:
    """Split text into sentence-level statements with exact character offsets.

    Lines are split first (one statement per line in structured documents), then sentences.
    Markdown headings, block quotes and empty lines are skipped.
    """
    spans: List[StatementSpan] = []
    pos = 0
    for line in text.split("\n"):
        line_start = pos
        pos += len(line) + 1
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ">", "---")):
            continue
        offset_in_line = line.index(stripped)
        chunk_start = line_start + offset_in_line
        last = 0
        pieces = []
        for m in _SENTENCE_END.finditer(stripped):
            pieces.append((last, m.start()))
            last = m.end()
        pieces.append((last, len(stripped)))
        for a, b in pieces:
            sentence = stripped[a:b].strip().lstrip("-*• ").strip()
            if len(sentence) < 3:
                continue
            start = chunk_start + stripped.index(sentence, a)
            spans.append(StatementSpan(sentence, base_offset + start, base_offset + start + len(sentence)))
    return spans


# ------------------------------------------------------------------------------ parsing
def _consumed(spans: Sequence[Tuple[int, int]], start: int, end: int) -> bool:
    return any(a <= start and end <= b for a, b in spans)


_BELOW = re.compile(r"(?:below|less than|under|fewer than|<)\s*$")
_ABOVE_BEFORE = re.compile(r"(?:above|more than|greater than|over|at least|>=|>)\s*$")
_AT_LEAST_AFTER = re.compile(r"^\s*(?:[a-z/%.\d-]+\s*){0,2}?(?:or above|or more|or older|or greater|or over|and above)")
_AT_MOST_AFTER = re.compile(r"^\s*(?:[a-z/%.\d-]+\s*){0,2}?(?:or below|or less|or under|or younger)")


def _comparator(lowered: str, start: int, end: int) -> str:
    """Direction of a threshold from the words around its number ("below 30", "45 or above")."""
    before = lowered[max(0, start - 20):start]
    after = lowered[end:end + 30]
    if re.search(r"(?:^|\s)to\s*$", before):
        return "range_high"
    if re.match(r"^\s*(?:[a-z/%.\d-]+\s*){0,2}?to\s+\d", after):
        return "range_low"
    if _BELOW.search(before):
        return "<"
    if _ABOVE_BEFORE.search(before):
        return ">"
    if _AT_LEAST_AFTER.match(after):
        return ">="
    if _AT_MOST_AFTER.match(after):
        return "<="
    return "="


def parse_statement(text: str, taxonomy: Taxonomy, source_type: Optional[str] = None) -> ParsedStatement:
    """Parse one statement (or a planned treatment) into structured attributes."""
    parsed = ParsedStatement(text=text)
    mention = taxonomy.find_treatments(text)
    parsed.treatments, parsed.treatment_classes = mention.treatments, mention.classes

    evidence = _EVIDENCE.search(text)
    parsed.evidence_level = evidence.group(1).upper() if evidence else None
    body = _EVIDENCE.sub("", text)
    consumed: List[Tuple[int, int]] = []

    for m in _DOSE.finditer(body):
        unit = m.group(2).lower()
        unit = {"micrograms": "mcg", "microgram": "mcg", "µg": "mcg", "units": "unit"}.get(unit, unit)
        parsed.doses.append(Dose(float(m.group(1)), unit))
        consumed.append(m.span())
    for pattern, label in _FREQUENCIES:
        m = pattern.search(body)
        if m:
            parsed.frequency = label
            break
    m = _EVERY.search(body)
    if m:
        parsed.frequency = parsed.frequency or f"every_{m.group(1)}_{m.group(2).rstrip('s').lower()}"
        consumed.append(m.span())
    m = _INTERVAL.search(body)
    if m:
        parsed.interval = (float(m.group(1)), m.group(2).rstrip("s").lower())
        consumed.append(m.span())
    m = _DURATION.search(body)
    if m:
        parsed.duration = (float(m.group(1)), m.group(2).rstrip("s").lower())
        consumed.append(m.span())
    for pattern, label in _ROUTES:
        if pattern.search(body):
            parsed.route = label
            break

    lowered = body.lower()
    for m in _NUMBER.finditer(body):
        if _consumed(consumed, *m.span()):
            continue
        before = lowered[max(0, m.start() - 60):m.start()]
        measure = None
        best = (-1, 0)  # (end position of the term, term length): nearest term wins, longest on ties
        for pattern, key in _MEASURE_PATTERNS:
            for found in pattern.finditer(before):
                if (found.end(), found.end() - found.start()) > best:
                    best, measure = (found.end(), found.end() - found.start()), key
        if measure is None:
            continue  # bare numbers (e.g. "1.73m2", ages in trial sizes) are not thresholds
        parsed.thresholds.append(Threshold(measure, float(m.group(1)), _comparator(lowered, m.start(), m.end())))

    if _WITHDRAWAL.search(body):
        parsed.statement_type = "withdrawal"
    elif _CONTRAINDICATION.search(body):
        parsed.statement_type = "contraindication"
    elif source_type == "safety_notice" or _SAFETY.search(body):
        parsed.statement_type = "safety_warning" if (parsed.treatments or parsed.treatment_classes) else "informational"
    elif source_type == "publication" and _EVIDENCE_FINDING.search(body) and not _RECOMMENDATION.search(body.strip()):
        parsed.statement_type = "evidence"
    elif _RECOMMENDATION.search(body.strip()):
        parsed.statement_type = "recommendation"
    return parsed


# ------------------------------------------------------------------------------ comparison
def skeleton(text: str) -> str:
    """Text with numbers and frequency words neutralised, for aligning versions of a statement."""
    s = _EVIDENCE.sub("", text.lower())
    s = re.sub(r"\d+(?:\.\d+)?", "#", s)
    for pattern, _ in _FREQUENCIES:
        s = pattern.sub("<freq>", s)
    return " ".join(s.split())


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


@dataclass
class AttributeDifference:
    kind: str  # dose | frequency | interval | duration | route | threshold
    before: str
    after: str


def _fmt_dose(doses: Sequence[Dose]) -> str:
    return ", ".join(f"{d.value:g} {d.unit}" for d in doses) or "none"


def _fmt_pair(pair) -> str:
    return f"{pair[0]:g} {pair[1]}(s)" if pair else "none"


def attribute_differences(old: ParsedStatement, new: ParsedStatement) -> List[AttributeDifference]:
    """Structured differences between two parsed statements about the same thing."""
    diffs: List[AttributeDifference] = []
    if sorted((d.value, d.unit) for d in old.doses) != sorted((d.value, d.unit) for d in new.doses):
        diffs.append(AttributeDifference("dose", _fmt_dose(old.doses), _fmt_dose(new.doses)))
    if old.frequency != new.frequency:
        diffs.append(AttributeDifference("frequency", old.frequency or "none", new.frequency or "none"))
    if old.interval != new.interval:
        diffs.append(AttributeDifference("interval", _fmt_pair(old.interval), _fmt_pair(new.interval)))
    if old.duration != new.duration:
        diffs.append(AttributeDifference("duration", _fmt_pair(old.duration), _fmt_pair(new.duration)))
    if old.route and new.route and old.route != new.route:
        diffs.append(AttributeDifference("route", old.route, new.route))
    old_t = sorted((t.measure, t.value) for t in old.thresholds)
    new_t = sorted((t.measure, t.value) for t in new.thresholds)
    if old_t != new_t:
        fmt = lambda ts: ", ".join(f"{m} {v:g}" for m, v in ts) or "none"
        diffs.append(AttributeDifference("threshold", fmt(old_t), fmt(new_t)))
    return diffs


def change_category_for(old: Optional[ParsedStatement], new: Optional[ParsedStatement], source_type: Optional[str] = None) -> str:
    """Deterministic change category between two versions of a statement."""
    if old is None and new is not None:
        if new.statement_type == "withdrawal":
            return "withdrawn"
        if new.statement_type == "contraindication":
            return "contraindication_added"
        if new.statement_type == "safety_warning" or source_type == "safety_notice":
            return "safety_warning"
        if new.statement_type in ("recommendation",):
            return "new_recommendation"
        return "no_practice_change"
    if new is None and old is not None:
        if old.statement_type == "contraindication":
            return "contraindication_removed"
        return "withdrawn" if old.statement_type in ("recommendation", "safety_warning") else "no_practice_change"
    assert old is not None and new is not None
    if old.statement_type != "contraindication" and new.statement_type == "contraindication":
        return "contraindication_added"
    if old.statement_type == "contraindication" and new.statement_type != "contraindication":
        return "contraindication_removed"
    if new.statement_type == "withdrawal":
        return "withdrawn"
    kinds = {d.kind for d in attribute_differences(old, new)}
    if kinds & {"dose", "frequency", "interval", "duration", "route"}:
        return "dose_change"
    if "threshold" in kinds:
        return "threshold_change"
    if skeleton(old.text) != skeleton(new.text):
        return "revised_recommendation"
    return "unchanged"


@dataclass
class Alignment:
    old_index: Optional[int]
    new_index: Optional[int]
    score: float


def align_statements(old: Sequence[str], new: Sequence[str], threshold: float = 0.72) -> List[Alignment]:
    """Greedy one-to-one alignment of statement versions (exact text first, then skeleton similarity)."""
    pairs: List[Alignment] = []
    used_old, used_new = set(), set()
    for j, n in enumerate(new):
        for i, o in enumerate(old):
            if i not in used_old and " ".join(o.split()) == " ".join(n.split()):
                pairs.append(Alignment(i, j, 1.0))
                used_old.add(i)
                used_new.add(j)
                break
    candidates = sorted(
        ((similarity(skeleton(o), skeleton(n)), i, j)
         for i, o in enumerate(old) if i not in used_old
         for j, n in enumerate(new) if j not in used_new),
        reverse=True,
    )
    for score, i, j in candidates:
        if score < threshold:
            break
        if i in used_old or j in used_new:
            continue
        pairs.append(Alignment(i, j, score))
        used_old.add(i)
        used_new.add(j)
    pairs += [Alignment(i, None, 0.0) for i in range(len(old)) if i not in used_old]
    pairs += [Alignment(None, j, 0.0) for j in range(len(new)) if j not in used_new]
    return pairs


def mentions_any(parsed: ParsedStatement, treatments: Iterable[str], classes: Iterable[str] = ()) -> bool:
    return bool(set(parsed.treatments) & set(treatments) or set(parsed.treatment_classes) & set(classes))
