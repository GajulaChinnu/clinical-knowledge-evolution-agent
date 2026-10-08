"""Service for deterministically identifying changes between source versions."""

import difflib
from dataclasses import dataclass
from typing import List, Sequence

from app.services.pdf_parser import DocumentSection


@dataclass
class ChangedSection:
    """Represents a discrete changed or newly added section between versions."""
    heading: str
    old_text: str
    new_text: str
    change_type: str  # 'added', 'modified', 'removed'


class SourceVersionDiffService:
    """Deterministic differ for clinical source versions."""

    def diff_sections(
        self,
        old_sections: Sequence[DocumentSection],
        new_sections: Sequence[DocumentSection],
    ) -> List[ChangedSection]:
        """Compare old and new document sections and extract significant textual changes.
        
        Args:
            old_sections: Parsed sections from previous source version.
            new_sections: Parsed sections from new source version.
            
        Returns:
            List of ChangedSection containing only added or modified clinical text.
        """
        changes: List[ChangedSection] = []

        # Simple section matching based on headings
        old_dict = {sec.section_heading: sec.text for sec in old_sections}
        new_dict = {sec.section_heading: sec.text for sec in new_sections}

        for heading, new_text in new_dict.items():
            if heading not in old_dict:
                changes.append(
                    ChangedSection(
                        heading=heading,
                        old_text="",
                        new_text=new_text,
                        change_type="added",
                    )
                )
            else:
                old_text = old_dict[heading]
                if old_text != new_text:
                    # Token optimization: verify structural difference using SequenceMatcher
                    ratio = difflib.SequenceMatcher(None, old_text, new_text).ratio()
                    if ratio < 0.99:  # Allow trivial whitespace/punctuation changes
                        changes.append(
                            ChangedSection(
                                heading=heading,
                                old_text=old_text,
                                new_text=new_text,
                                change_type="modified",
                            )
                        )

        # Track removed sections
        for heading, old_text in old_dict.items():
            if heading not in new_dict:
                changes.append(
                    ChangedSection(
                        heading=heading,
                        old_text=old_text,
                        new_text="",
                        change_type="removed",
                    )
                )

        return changes
