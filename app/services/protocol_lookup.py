"""Exact institutional protocol section lookup.

The only way CKEA resolves protocol text outside the comparison step: same protocol id,
same immutable version, same section id. No fallback to another version, another
section, or placeholder text — callers must surface an unresolved lookup instead.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

from app.schemas.protocol import ProtocolDocument, ProtocolSection
from app.services.protocol_index import ProtocolError, parse_protocol_file

logger = logging.getLogger("ckea.services.protocol_lookup")

PROTOCOL_SUFFIXES = (".json", ".md", ".txt")


def section_id_from_candidates(candidate_section_ids: Optional[List[str]]) -> Optional[str]:
    """Section id of the top candidate ("<protocol>__<version>__<section>")."""
    if not candidate_section_ids:
        return None
    return str(candidate_section_ids[0]).split("__")[-1] or None


def find_protocol_section(
    protocol_dir: Union[str, Path],
    protocol_id: Optional[str],
    protocol_version: Optional[str],
    section_id: Optional[str],
) -> Optional[Tuple[ProtocolDocument, ProtocolSection]]:
    """Return (protocol, section) for an exact id/version/section match, else None."""
    if not (protocol_id and protocol_version and section_id):
        return None
    directory = Path(protocol_dir)
    if not directory.exists():
        return None
    for p_file in sorted(directory.iterdir()):
        if not (p_file.is_file() and p_file.suffix.lower() in PROTOCOL_SUFFIXES):
            continue
        try:
            p_doc = parse_protocol_file(p_file)
        except ProtocolError as e:
            logger.warning("Protocol file %s could not be parsed: %s", p_file.name, e)
            continue
        if p_doc.protocol_id != protocol_id or p_doc.protocol_version != protocol_version:
            continue
        for sec in p_doc.sections:
            if sec.section_id == section_id:
                return p_doc, sec
    return None
