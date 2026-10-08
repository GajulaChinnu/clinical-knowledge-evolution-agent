"""Service for deterministically identifying changes between source versions.

This is SOURCE EVOLUTION (external v(n-1) -> v(n)). It is entirely separate from the
institutional protocol comparison performed by the Comparison Agent.
"""

import difflib
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

from app.services.pdf_parser import DocumentSection

# Headings at least this similar are treated as the same section (renamed / renumbered).
HEADING_MATCH_RATIO = 0.80
# Unmatched removed/added sections whose bodies are at least this similar are a rename.
BODY_MATCH_RATIO = 0.60
# Text changes below this dissimilarity are trivial (whitespace/punctuation).
TRIVIAL_CHANGE_RATIO = 0.99


@dataclass
class ChangedSection:
    """Represents a discrete added, modified or removed section between versions."""
    heading: str
    old_text: str
    new_text: str
    change_type: str  # 'added', 'modified', 'removed'
    old_heading: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def _merge_by_heading(sections: Sequence[DocumentSection]) -> Dict[str, str]:
    """Merge same-heading fragments (e.g. a section continued on the next page)."""
    merged: Dict[str, List[str]] = {}
    for sec in sections:
        merged.setdefault(sec.section_heading, []).append(sec.text)
    return {heading: "\n".join(parts) for heading, parts in merged.items()}


class SourceVersionDiffService:
    """Deterministic differ for clinical source versions."""

    def diff_sections(
        self,
        old_sections: Sequence[DocumentSection],
        new_sections: Sequence[DocumentSection],
    ) -> List[ChangedSection]:
        """Compare old and new document sections.

        Matching: exact heading first, then fuzzy heading (renamed/renumbered sections),
        then body similarity for remaining pairs. Only genuinely unmatched sections are
        reported as added or removed.

        Returns:
            ChangedSection list ordered as: modified/added (new-document order), then removed.
        """
        old_dict = _merge_by_heading(old_sections)
        new_dict = _merge_by_heading(new_sections)

        pairs: Dict[str, str] = {h: h for h in new_dict if h in old_dict}  # new_heading -> old_heading
        unmatched_old = [h for h in old_dict if h not in pairs.values()]
        unmatched_new = [h for h in new_dict if h not in pairs]

        for threshold, key in ((HEADING_MATCH_RATIO, "heading"), (BODY_MATCH_RATIO, "body")):
            for new_h in list(unmatched_new):
                best: Optional[str] = None
                best_score = threshold
                for old_h in unmatched_old:
                    score = _ratio(new_h.lower(), old_h.lower()) if key == "heading" else _ratio(new_dict[new_h], old_dict[old_h])
                    if score >= best_score:
                        best, best_score = old_h, score
                if best is not None:
                    pairs[new_h] = best
                    unmatched_old.remove(best)
                    unmatched_new.remove(new_h)

        changes: List[ChangedSection] = []
        for new_h, new_text in new_dict.items():
            if new_h in unmatched_new:
                changes.append(ChangedSection(heading=new_h, old_text="", new_text=new_text, change_type="added"))
                continue
            old_h = pairs[new_h]
            old_text = old_dict[old_h]
            if old_text != new_text and _ratio(old_text, new_text) < TRIVIAL_CHANGE_RATIO:
                changes.append(ChangedSection(
                    heading=new_h,
                    old_text=old_text,
                    new_text=new_text,
                    change_type="modified",
                    old_heading=old_h if old_h != new_h else None,
                ))

        for old_h in unmatched_old:
            changes.append(ChangedSection(heading=old_h, old_text=old_dict[old_h], new_text="", change_type="removed"))

        return changes
