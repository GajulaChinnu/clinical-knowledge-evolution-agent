"""Institutional (SYNTHETIC) protocol repository with clinical ownership metadata.

Protocol files are immutable. Department/pathway/treatment metadata comes from the file itself
or, for legacy files without metadata, from `legacy_protocol_metadata` in config/taxonomy.yaml.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

from app.schemas.protocol import ProtocolDocument
from app.services.protocol_index import ProtocolError, parse_protocol_file
from app.services.protocol_lookup import PROTOCOL_SUFFIXES
from app.services.taxonomy import Taxonomy, TaxonomyError, get_taxonomy

logger = logging.getLogger("ckea.services.protocol_repository")


def resolve_protocol_metadata(doc: ProtocolDocument, taxonomy: Taxonomy) -> ProtocolDocument:
    """Fill department/pathways/treatments from taxonomy legacy metadata and validate them."""
    if not doc.department:
        legacy = taxonomy.legacy_metadata_for(doc.protocol_id)
        if legacy is not None:
            doc = doc.model_copy(update={
                "department": legacy.department,
                "pathways": list(legacy.pathways),
                "treatments": list(legacy.treatments),
                "owner": doc.owner or legacy.owner,
                "effective_date": doc.effective_date or legacy.effective_date,
            })
    if doc.department:
        doc = doc.model_copy(update={
            "department": taxonomy.resolve_department(doc.department),
            "pathways": taxonomy.validate_pathways(doc.pathways),
            "treatments": taxonomy.validate_treatments(doc.treatments),
        })
    return doc


def load_protocols(
    protocol_dir: Union[str, Path],
    taxonomy: Optional[Taxonomy] = None,
) -> List[ProtocolDocument]:
    """All parseable protocols in a directory with resolved metadata (unparseable files are logged)."""
    taxonomy = taxonomy or get_taxonomy()
    directory = Path(protocol_dir)
    if not directory.exists():
        return []
    protocols: List[ProtocolDocument] = []
    for path in sorted(directory.iterdir()):
        if not (path.is_file() and path.suffix.lower() in PROTOCOL_SUFFIXES and not path.name.startswith(".")):
            continue
        try:
            protocols.append(resolve_protocol_metadata(parse_protocol_file(path), taxonomy))
        except (ProtocolError, TaxonomyError, FileNotFoundError) as e:
            logger.warning("Protocol file %s skipped: %s", path.name, e)
    return protocols


def protocols_for_department(protocols: List[ProtocolDocument], department: str) -> List[ProtocolDocument]:
    return [p for p in protocols if p.department == department]
