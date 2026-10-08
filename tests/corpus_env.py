"""Shared test helper: a temporary CKEA database with the whole SYNTHETIC corpus ingested and indexed."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock

from app.agents.comparison_agent import ComparisonAgent, SourceScope
from app.agents.extraction_agent import ExtractionAgent
from app.agents.monitoring_agent import MonitoringAgent, SourceCheckResult
from app.models.database import get_engine, get_session_factory, init_db
from app.schemas.clinician_query import ClinicianQueryInput, PatientContext, build_clinician_query
from app.services.config_service import AppConfig
from app.services.protocol_repository import load_protocols, protocols_for_department
from app.services.taxonomy import load_taxonomy
from app.services.watchlist import load_watchlist

TAXONOMY = load_taxonomy()
WATCHLIST = load_watchlist(taxonomy=TAXONOMY)
PROTOCOLS = load_protocols("data/protocols", TAXONOMY)


@dataclass
class CorpusEnv:
    engine: object
    session_factory: object
    config: AppConfig
    monitoring: MonitoringAgent
    extraction: ExtractionAgent
    comparison: ComparisonAgent
    checks: Dict[str, SourceCheckResult]

    def scopes(self, department: str, treatments: List[str]) -> List[SourceScope]:
        entries = WATCHLIST.select(department, treatments, TAXONOMY)
        out = []
        for e in entries:
            check = self.checks[e.id]
            out.append(SourceScope(e.id, check.latest.document_id if check.latest else None,
                                   check.previous.document_id if check.previous else None))
        return out

    def compare(self, department: str, treatment: str, condition: Optional[str] = None, **context):
        query = build_clinician_query(ClinicianQueryInput(
            department=department, treatment=treatment, condition=condition,
            context=PatientContext(**context)), TAXONOMY)
        outcome = self.comparison.compare_treatment_plan(
            query, self.scopes(query.department, query.treatment_ids),
            protocols_for_department(PROTOCOLS, query.department), TAXONOMY)
        return query, outcome


def build_corpus_env(tmp_path: Path) -> CorpusEnv:
    db = tmp_path / "corpus.db"
    engine = get_engine(db_url=f"sqlite:///{db}")
    init_db(engine=engine)
    sf = get_session_factory(engine=engine)
    (tmp_path / "sources").mkdir(exist_ok=True)
    config = AppConfig(database_url=f"sqlite:///{db}", source_dir=tmp_path / "sources",
                       output_dir=tmp_path / "output", groq_api_key="mock-key")
    monitoring = MonitoringAgent(source_dir=config.source_dir, session_factory=sf, config=config)
    extraction = ExtractionAgent(session_factory=sf, config=config, llm_client=MagicMock())
    comparison = ComparisonAgent(session_factory=sf, config=config, llm_client=MagicMock(),
                                 protocol_index_service=MagicMock())
    checks = {}
    for entry in WATCHLIST:
        result = monitoring.check_watchlist_entry(entry, WATCHLIST)
        assert result.error is None, result.error
        extraction.index_guidance_statements([v.document_id for v in result.versions])
        checks[entry.id] = result
    return CorpusEnv(engine, sf, config, monitoring, extraction, comparison, checks)
