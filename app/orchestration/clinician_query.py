"""Clinician Treatment Check workflow: the six CKEA agents run for one clinician question.

Monitoring -> Extraction -> Comparison -> Impact -> Briefing -> Governance.
This is orchestration, not a seventh agent: each step calls the owning agent's query-mode method.
Every query is audited (department, treatment, verdict, cited ids). Patient context is never stored.
"""

import json
import logging
from typing import Callable, List, Optional, Sequence

from sqlalchemy.orm import Session, sessionmaker

from app.agents.briefing_agent import BriefingAgent
from app.agents.comparison_agent import ComparisonAgent, SourceScope
from app.agents.extraction_agent import ExtractionAgent
from app.agents.governance_agent import GovernanceAgent
from app.agents.impact_agent import ImpactAgent
from app.agents.monitoring_agent import MonitoringAgent, SourceCheckResult
from app.models.entities import AuditLog, ClinicianQueryRecord
from app.schemas.clinician_query import ClinicianQuery, ClinicianQueryInput, build_clinician_query
from app.schemas.protocol import ProtocolDocument
from app.schemas.treatment_check import ClinicianAnswer, ComparisonOutcome, Finding, SourceChecked, StepRecord
from app.services.config_service import AppConfig
from app.services.protocol_repository import load_protocols, protocols_for_department
from app.services.taxonomy import Taxonomy, get_taxonomy
from app.services.watchlist import Watchlist, load_watchlist

logger = logging.getLogger("ckea.orchestration.clinician_query")

StepCallback = Callable[[StepRecord], None]


class ClinicianQueryWorkflow:
    """Runs the six agents for a clinician's department + planned treatment."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        config: AppConfig,
        monitoring_agent: MonitoringAgent,
        extraction_agent: ExtractionAgent,
        comparison_agent: ComparisonAgent,
        impact_agent: ImpactAgent,
        briefing_agent: BriefingAgent,
        governance_agent: GovernanceAgent,
        watchlist: Optional[Watchlist] = None,
        taxonomy: Optional[Taxonomy] = None,
        protocols: Optional[Sequence[ProtocolDocument]] = None,
        url_service=None,
    ) -> None:
        self.session_factory = session_factory
        self.config = config
        self.monitoring = monitoring_agent
        self.extraction = extraction_agent
        self.comparison = comparison_agent
        self.impact = impact_agent
        self.briefing = briefing_agent
        self.governance = governance_agent
        self.taxonomy = taxonomy or get_taxonomy()
        self.watchlist = watchlist or load_watchlist(taxonomy=self.taxonomy)
        self._protocols = list(protocols) if protocols is not None else None
        self.url_service = url_service

    @property
    def protocols(self) -> List[ProtocolDocument]:
        if self._protocols is None:
            self._protocols = load_protocols(self.config.protocol_dir, self.taxonomy)
        return self._protocols

    def run(self, raw: ClinicianQueryInput, actor: str = "clinician", on_step: Optional[StepCallback] = None) -> ClinicianAnswer:
        """Run the Treatment Check. Raises PatientIdentifierError / TaxonomyError before any agent runs."""
        query = build_clinician_query(raw, self.taxonomy)
        steps: List[StepRecord] = []

        def record(step: StepRecord) -> None:
            steps.append(step)
            if on_step is not None:
                on_step(step)

        # 1. Monitoring --------------------------------------------------------------
        entries = self.watchlist.select(query.department, query.treatment_ids, self.taxonomy)
        checks: List[SourceCheckResult] = self.monitoring.check_watchlist(self.watchlist, entries, self.url_service)
        sources = [self._source_checked(c) for c in checks]
        new_versions = sum(len(c.new_document_ids) for c in checks)
        failures = [c for c in checks if c.error]
        record(StepRecord(
            agent="Monitoring", status="failed" if failures and len(failures) == len(checks) and checks else "completed",
            summary=(f"Checked {len(checks)} watchlist source(s) for {query.department_name}; "
                     f"{new_versions} new version(s) ingested" + (f"; {len(failures)} failed" if failures else "") + "."),
            details=[f"{s.title}: latest v{s.latest_version} ({s.latest_published})" + (f" - ERROR {s.error}" if s.error else "")
                     for s in sources],
        ))

        # 2. Extraction ---------------------------------------------------------------
        doc_ids = [v.document_id for c in checks for v in (c.latest, c.previous) if v is not None]
        indexed = self.extraction.index_guidance_statements(doc_ids)
        record(StepRecord(
            agent="Extraction", status="completed",
            summary=(f"Indexed verbatim statements for {len(doc_ids)} source version(s) "
                     f"({sum(r.statements_created for r in indexed)} new statements)."),
            details=[w for r in indexed for w in r.warnings],
        ))

        # 3. Comparison ---------------------------------------------------------------
        scopes = [SourceScope(c.entry_id, c.latest.document_id if c.latest else None,
                              c.previous.document_id if c.previous else None) for c in checks]
        protocols = protocols_for_department(self.protocols, query.department)
        outcome: Optional[ComparisonOutcome] = None
        try:
            outcome = self.comparison.compare_treatment_plan(query, scopes, protocols, self.taxonomy)
            record(StepRecord(
                agent="Comparison", status="completed",
                summary=(f"{len(outcome.findings)} grounded finding(s); {len(outcome.version_changes)} version change(s); "
                         f"{len(outcome.protocol_positions)} protocol(s) compared."),
                details=outcome.notes + [f"{p.protocol_id} {p.protocol_version}: {p.status}" for p in outcome.protocol_positions],
            ))
        except Exception as e:  # comparison failure degrades to passages-only; it is shown, never hidden
            logger.exception("Comparison failed for clinician query")
            record(StepRecord(agent="Comparison", status="failed", summary=f"Comparison unavailable: {type(e).__name__}: {e}"))

        # 4. Impact -------------------------------------------------------------------
        findings: List[Finding] = outcome.findings if outcome else []
        ranked = self.impact.score_findings(query.department, query.treatment_ids, findings) if findings else []
        record(StepRecord(
            agent="Impact", status="completed" if outcome else "skipped",
            summary=(f"Ranked {len(ranked)} finding(s) by urgency, relevance, source quality and novelty."
                     if ranked else "No findings to rank."),
            details=[f"{f.priority_score}: {f.kind} ({f.relation}) - {f.citation.source_title} v{f.citation.version}" for f in ranked[:5]],
        ))

        # 5. Briefing -----------------------------------------------------------------
        answer = self.briefing.build_clinician_answer(query, outcome, ranked, sources, steps, self.taxonomy)
        record(StepRecord(
            agent="Briefing", status="completed",
            summary=f"Verdict: {answer.verdict_label} ({len(answer.citations)} verified citation(s)).",
        ))

        # 6. Governance ---------------------------------------------------------------
        governance = self.governance.governance_for_query(query, answer, actor=actor)
        answer.governance = governance
        record(StepRecord(
            agent="Governance", status="completed",
            summary=("; ".join(g.message for g in governance) if governance
                     else "No protocol update review needed for this query."),
        ))
        answer.steps = steps

        answer.query_id = self._persist(query, answer, actor)
        return answer

    # ------------------------------------------------------------------------------------
    def _source_checked(self, check: SourceCheckResult) -> SourceChecked:
        entry = self.watchlist.get(check.entry_id)
        return SourceChecked(
            watchlist_id=check.entry_id, title=entry.title, source_type=entry.source_type,
            checked_at=check.checked_at.isoformat(),
            latest_version=(check.latest.publisher_version or check.latest.internal_version) if check.latest else None,
            latest_published=check.latest.published_date if check.latest else None,
            previous_version=(check.previous.publisher_version or check.previous.internal_version) if check.previous else None,
            previous_published=check.previous.published_date if check.previous else None,
            new_versions_ingested=len(check.new_document_ids), error=check.error,
        )

    @staticmethod
    def _without_patient_context(query: ClinicianQuery, answer: ClinicianAnswer) -> dict:
        """Stored copy of the answer with patient-context values (bands, comorbidities, medicines) masked."""
        ctx = query.context
        sensitive = [v for v in (ctx.egfr_band, ctx.age_band) if v]
        sensitive += [t for t in (*ctx.comorbidities, *ctx.current_medications) if t]
        sensitive += [t.lower() for t in (*ctx.comorbidities, *ctx.current_medications) if t]
        for med in query.medication_treatments:
            sensitive.append(med)
            sensitive.append(get_taxonomy().treatment_name(med))
        if ctx.pregnancy:
            sensitive.append("patient is pregnant")
        text = json.dumps(answer.model_dump(mode="json"))
        for value in sorted(set(sensitive), key=len, reverse=True):
            text = text.replace(json.dumps(value)[1:-1], "[context]")
        return json.loads(text)

    def _persist(self, query: ClinicianQuery, answer: ClinicianAnswer, actor: str) -> str:
        cited = [c.statement_id for c in answer.citations]
        with self.session_factory() as session:
            record = ClinicianQueryRecord(
                actor=actor, department=query.department, treatment=query.plan.text[:255],
                condition=(query.condition or None) and query.condition[:255], verdict=answer.verdict,
                cited_statement_ids=cited, answer=self._without_patient_context(query, answer),
            )
            session.add(record)
            session.flush()
            session.add(AuditLog(
                entity_id=record.id, entity_type="ClinicianQuery", previous_status=None,
                new_status=answer.verdict or "no_verdict", actor=actor,
                reason=f"Treatment check: {query.department} / {query.plan.text[:120]}",
                audit_metadata={"department": query.department, "treatment": query.plan.text[:120],
                                "verdict": answer.verdict, "cited_statement_ids": cited},
                schema_version="1.0",
            ))
            session.commit()
            return record.id
