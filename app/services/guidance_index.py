"""Statement-level guidance index and deterministic source-evolution change detection.

For every ingested source version:
1. Split the stored artifact into verbatim statements (char offsets verified against the file).
2. Parse each statement (treatments, doses, thresholds, type) and store it immutably.
3. Align statements with the previous version and record categorised GuidanceChange rows
   (dose/threshold changes, new/withdrawn recommendations, contraindications, safety warnings).

No LLM is involved, so the index is reproducible and every citation is a real substring.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from sqlalchemy.orm import Session, sessionmaker

from app.models.entities import GuidanceChange, GuidanceStatement, IngestedDocument
from app.services.clinical_statements import (
    ParsedStatement,
    align_statements,
    attribute_differences,
    change_category_for,
    parse_statement,
    split_statements,
)
from app.services.source_documents import load_source_sections, load_source_text
from app.services.taxonomy import Taxonomy, get_taxonomy

logger = logging.getLogger("ckea.services.guidance_index")


@dataclass
class IndexResult:
    document_id: str
    statements_created: int = 0
    changes_created: int = 0
    skipped: bool = False
    warnings: List[str] = field(default_factory=list)


def _metadata(doc: IngestedDocument) -> dict:
    return dict(doc.doc_metadata or {})


class GuidanceIndexService:
    """Builds the statement index and change log for ingested source versions."""

    def __init__(self, session_factory: sessionmaker[Session], taxonomy: Optional[Taxonomy] = None) -> None:
        self.session_factory = session_factory
        self.taxonomy = taxonomy or get_taxonomy()

    # ---------------------------------------------------------------- public API
    def index_document(self, document_id: str) -> IndexResult:
        """Index statements for a document (and its previous version), then record changes. Idempotent."""
        with self.session_factory() as session:
            doc = session.get(IngestedDocument, document_id)
            if doc is None:
                raise ValueError(f"IngestedDocument '{document_id}' not found.")
            result = IndexResult(document_id=document_id)

            previous = session.get(IngestedDocument, doc.previous_source_version_id) if doc.previous_source_version_id else None
            if previous is not None and not self._has_statements(session, previous.id):
                self._create_statements(session, previous, result)

            if self._has_statements(session, doc.id):
                result.skipped = True
            else:
                self._create_statements(session, doc, result)

            if not session.query(GuidanceChange.id).filter_by(to_document_id=doc.id).first():
                result.changes_created = self._create_changes(session, doc, previous)
            session.commit()
            return result

    def statements_for(self, session: Session, document_id: str) -> List[GuidanceStatement]:
        return (
            session.query(GuidanceStatement)
            .filter_by(ingested_document_id=document_id)
            .order_by(GuidanceStatement.sequence)
            .all()
        )

    # ---------------------------------------------------------------- statements
    @staticmethod
    def _has_statements(session: Session, document_id: str) -> bool:
        return session.query(GuidanceStatement.id).filter_by(ingested_document_id=document_id).first() is not None

    def _statement_scope(self, doc: IngestedDocument, parsed: ParsedStatement) -> Tuple[List[str], List[str]]:
        meta = _metadata(doc)
        source_departments = list(meta.get("departments") or [])
        departments = source_departments or self.taxonomy.departments_for_treatments(parsed.treatments)
        pathways = self.taxonomy.pathways_for(parsed.treatments, departments)
        return sorted(set(departments)), pathways

    def _create_statements(self, session: Session, doc: IngestedDocument, result: IndexResult) -> int:
        meta = _metadata(doc)
        path = Path(doc.source_path)
        try:
            artifact_text = load_source_text(path)
            sections = load_source_sections(path)
        except (OSError, ValueError) as e:
            result.warnings.append(f"Document {doc.id} could not be indexed: {e}")
            logger.warning("Guidance index skipped document %s: %s", doc.id, e)
            return 0

        created = 0
        seq = 0
        is_text_artifact = path.suffix.lower() in (".md", ".txt")
        for section in sections:
            base = section.char_start if (is_text_artifact and section.char_start is not None) else None
            for span in split_statements(section.text, base_offset=base or 0):
                start = span.char_start if base is not None else None
                end = span.char_end if base is not None else None
                if start is not None and artifact_text[start:end] != span.text:
                    # Offsets must point at the exact text; otherwise keep the statement unanchored.
                    result.warnings.append(f"Offset mismatch in {path.name} at {start}; stored without offsets.")
                    start = end = None
                parsed = parse_statement(span.text, self.taxonomy, meta.get("source_type"))
                departments, pathways = self._statement_scope(doc, parsed)
                session.add(GuidanceStatement(
                    ingested_document_id=doc.id,
                    source_identity=doc.source_identifier,
                    watchlist_id=meta.get("watchlist_id"),
                    source_type=meta.get("source_type"),
                    publisher_version=str(meta.get("publisher_version") or doc.source_version),
                    published_date=meta.get("published_date"),
                    section_heading=section.section_heading,
                    char_start=start,
                    char_end=end,
                    sequence=seq,
                    verbatim_text=span.text,
                    statement_type=parsed.statement_type,
                    treatments=parsed.treatments,
                    treatment_classes=parsed.treatment_classes,
                    departments=departments,
                    pathways=pathways,
                    parsed=parsed.to_dict(),
                    evidence_level=parsed.evidence_level,
                ))
                seq += 1
                created += 1
        session.flush()
        result.statements_created += created
        return created

    # ---------------------------------------------------------------- changes
    def _parsed(self, stmt: GuidanceStatement) -> ParsedStatement:
        return parse_statement(stmt.verbatim_text, self.taxonomy, stmt.source_type)

    def _create_changes(self, session: Session, doc: IngestedDocument, previous: Optional[IngestedDocument]) -> int:
        meta = _metadata(doc)
        source_type = meta.get("source_type")
        new_stmts = self.statements_for(session, doc.id)
        old_stmts = self.statements_for(session, previous.id) if previous is not None else []

        alignments = align_statements([s.verbatim_text for s in old_stmts], [s.verbatim_text for s in new_stmts])
        pairs: List[Tuple[Optional[GuidanceStatement], Optional[GuidanceStatement]]] = [
            (old_stmts[a.old_index] if a.old_index is not None else None,
             new_stmts[a.new_index] if a.new_index is not None else None)
            for a in alignments
        ]

        # Pair an explicit withdrawal notice with the statement it withdraws (shared treatment),
        # so a withdrawal is one change that shows both the old text and the withdrawal text.
        removed = [o for o, n in pairs if n is None and o is not None]
        merged: List[Tuple[Optional[GuidanceStatement], Optional[GuidanceStatement]]] = []
        consumed_old = set()
        for old, new in pairs:
            if old is None and new is not None and new.statement_type == "withdrawal":
                match = next(
                    (o for o in removed if o.id not in consumed_old and set(o.treatments or []) & set(new.treatments or [])),
                    None,
                )
                if match is not None:
                    consumed_old.add(match.id)
                    merged.append((match, new))
                    continue
            merged.append((old, new))
        merged = [(o, n) for o, n in merged if not (n is None and o is not None and o.id in consumed_old)]

        created = 0
        for old, new in merged:
            old_p = self._parsed(old) if old is not None else None
            new_p = self._parsed(new) if new is not None else None
            if old is not None and new is not None and new.statement_type == "withdrawal":
                category = "withdrawn"
            else:
                category = change_category_for(old_p, new_p, source_type)
            if category == "unchanged":
                continue
            treatments = sorted(set((old.treatments if old else []) or []) | set((new.treatments if new else []) or []))
            departments = sorted(set((old.departments if old else []) or []) | set((new.departments if new else []) or []))
            diffs = attribute_differences(old_p, new_p) if (old_p and new_p) else []
            session.add(GuidanceChange(
                source_identity=doc.source_identifier,
                watchlist_id=meta.get("watchlist_id"),
                source_type=source_type,
                from_document_id=previous.id if (previous is not None and old is not None) else None,
                to_document_id=doc.id,
                from_statement_id=old.id if old is not None else None,
                to_statement_id=new.id if new is not None else None,
                change_category=category,
                attribute_changes=[d.__dict__ for d in diffs],
                treatments=treatments,
                departments=departments,
                pathways=self.taxonomy.pathways_for(treatments, departments),
            ))
            created += 1
        session.flush()
        return created
