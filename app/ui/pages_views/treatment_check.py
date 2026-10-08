"""Ask CKEA - clinician Treatment Check (primary page).

A clinician states the department and the planned treatment; the six agents run and return a
grounded answer: matches current guidance, guidance updated (follow the new version), conflicts,
or no grounded guidance. Decision support only - the treating clinician decides.
"""

import html
from typing import List, Optional

import streamlit as st
from sqlalchemy.orm import Session, sessionmaker

from app.schemas.clinician_query import ClinicianQueryInput, PatientContext, PatientIdentifierError
from app.schemas.treatment_check import ClinicianAnswer, Citation, StepRecord
from app.services.config_service import AppConfig
from app.services.taxonomy import TaxonomyError, get_taxonomy
from app.ui.components import render_page_header
from app.ui.queries import build_clinician_workflow, get_query_history

VERDICT_STYLE = {
    "consistent_with_latest_guidance": ("badge-green", "#15803d"),
    "guidance_updated_follow_new_version": ("badge-amber", "#b45309"),
    "conflicts_with_latest_guidance": ("badge-red", "#b91c1c"),
    "insufficient_grounded_evidence": ("badge-neutral", "#475569"),
    None: ("badge-neutral", "#475569"),
}
AGENT_ORDER = ["Monitoring", "Extraction", "Comparison", "Impact", "Briefing", "Governance"]


def _human(value: str) -> str:
    return str(value).replace("_", " ")


def _split_terms(text: str) -> List[str]:
    return [t.strip() for t in (text or "").split(",") if t.strip()]


def _cite(c: Citation) -> str:
    where = c.section_heading or "section"
    date = f", published {c.published_date}" if c.published_date else ""
    return f"{c.source_title or c.source_identity} - version {c.version}{date} - {where}"


def _quote(c: Citation, label: Optional[str] = None) -> None:
    head = f"**{html.escape(label)}** · " if label else ""
    st.markdown(f"{head}{html.escape(_cite(c))}" + ("" if c.verified else "  \n:red[Not verified - not used for the verdict]"))
    st.markdown(f"> {html.escape(c.excerpt)}")


def render_answer(answer: ClinicianAnswer) -> None:
    badge, color = VERDICT_STYLE.get(answer.verdict, VERDICT_STYLE[None])
    st.markdown(
        f"""
        <div class="ent-card" style="border-left: 5px solid {color};">
            <div class="ent-metric-title">Verdict · {html.escape(answer.department_name)} · {html.escape(answer.treatment)}</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: {color}; margin: 4px 0;">{html.escape(answer.verdict_label)}</div>
            <div style="font-size: 0.85rem; color: #334155;">{html.escape(answer.verdict_basis)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(answer.disclaimer)

    if answer.summary:
        st.markdown("### Summary")
        for line in answer.summary:
            st.markdown(f"- {html.escape(line)}")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("### Latest guidance")
        if answer.latest_guidance:
            for c in answer.latest_guidance:
                _quote(c)
        else:
            st.caption("No current recommendation in the monitored sources matches this treatment.")

        changed = [v for v in answer.what_changed if v.previous or v.latest]
        if changed:
            st.markdown("### What changed between versions")
            for v in changed:
                tag = " · your plan follows the previous version" if v.plan_matches_previous else ""
                st.markdown(f"**{html.escape(v.change_category.replace('_', ' ').title())}** - "
                            f"{html.escape(v.source_title or v.source_identity)}{tag}")
                c_old, c_new = st.columns(2)
                with c_old:
                    if v.previous:
                        st.caption(f"Previous: version {v.previous.version} ({v.previous.published_date})")
                        st.markdown(f"> {html.escape(v.previous.excerpt)}")
                    else:
                        st.caption("Previous: not present")
                with c_new:
                    if v.latest:
                        st.caption(f"Latest: version {v.latest.version} ({v.latest.published_date})")
                        st.markdown(f"> {html.escape(v.latest.excerpt)}")
                    else:
                        st.caption("Latest: removed")
                for d in v.differences:
                    st.caption(f"{d.kind}: {_human(d.before)} -> {_human(d.after)}")

    with right:
        if answer.check_before_prescribing:
            st.markdown("### Check before prescribing")
            for f in answer.check_before_prescribing:
                st.warning(f"{f.applicability_basis or 'Confirm applicability'}")
                st.markdown(f"> {html.escape(f.citation.excerpt)}")
                st.caption(_cite(f.citation))
        if answer.safety_notices:
            st.markdown("### Safety notices")
            for f in answer.safety_notices:
                st.markdown(f"> {html.escape(f.citation.excerpt)}")
                st.caption(f"{_cite(f.citation)} · {f.relation.replace('_', ' ')}")

        st.markdown("### Hospital protocol")
        if not answer.protocol_positions:
            st.caption("No institutional protocol for this department.")
        for p in answer.protocol_positions:
            label = {"aligned": "Aligned with latest guidance", "out_of_date": "Out of date",
                     "not_covered": "Does not cover this treatment"}[p.status]
            st.markdown(f"**{html.escape(p.protocol_id)} {html.escape(p.protocol_version)}** - {label}")
            for item in p.items:
                if item.status == "aligned":
                    continue
                st.caption(item.reason)
                if item.section_text:
                    st.markdown(f"> Protocol: {html.escape(item.section_text)}")
                elif item.latest_citation:
                    st.markdown(f"> Guidance v{html.escape(str(item.latest_citation.version))}: "
                                f"{html.escape(item.latest_citation.excerpt)}")
        for g in answer.governance:
            st.markdown(f"**Governance:** {html.escape(g.message)}")

    st.markdown("### Recommended actions")
    for a in answer.recommended_actions:
        st.markdown(f"- {html.escape(a)}")
    if answer.affected_departments or answer.affected_pathways:
        st.caption("Affected: " + ", ".join(answer.affected_departments + answer.affected_pathways))

    with st.expander(f"All ranked findings ({len(answer.findings)})"):
        for f in answer.findings:
            st.markdown(f"**{f.priority_score}** · {f.kind.replace('_', ' ')} · {f.relation.replace('_', ' ')}"
                        + (" · duplicate of a higher-quality source" if f.novelty == "duplicate_of_existing" else ""))
            st.markdown(f"> {html.escape(f.citation.excerpt)}")
            basis = f.ranking_basis or {}
            st.caption(f"{_cite(f.citation)} · relevance {f.relevance} ({basis.get('relevance', {}).get('rule_id')}) · "
                       f"urgency {f.urgency} ({basis.get('urgency', {}).get('rule_id')}) · source quality {f.source_quality}")
    with st.expander(f"Citations ({len(answer.citations)}) and sources checked"):
        for c in answer.citations:
            st.markdown(f"- {html.escape(_cite(c))}: \"{html.escape(c.excerpt)}\"")
        st.dataframe([{
            "Source": s.title, "Type": s.source_type.replace("_", " "), "Latest": s.latest_version,
            "Published": s.latest_published, "Previous": s.previous_version, "Checked at": s.checked_at[:19],
            "New versions": s.new_versions_ingested, "Error": s.error or "",
        } for s in answer.sources_checked], use_container_width=True, hide_index=True)
    with st.expander("Agent steps"):
        _render_steps(answer.steps)


def _render_steps(steps: List[StepRecord]) -> None:
    for s in steps:
        icon = {"completed": "OK", "failed": "FAILED", "skipped": "SKIPPED"}[s.status]
        st.markdown(f"**{s.agent}** [{icon}] {html.escape(s.summary)}")
        for d in s.details[:8]:
            st.caption(d)


def render_treatment_check_view(session_factory: sessionmaker[Session], config: AppConfig) -> None:
    render_page_header(
        breadcrumb="CLINICAL DECISION SUPPORT › TREATMENT CHECK",
        title="Ask CKEA: is this treatment current?",
        description=("Enter the department and the treatment you plan. CKEA checks the latest versions of the monitored "
                     "guidance, safety notices and publications, and answers only from grounded source text."),
    )
    taxonomy = get_taxonomy()
    departments = {d.name: d.id for d in taxonomy.departments.values()}

    with st.form("treatment_check_form"):
        c1, c2 = st.columns([1, 2])
        department = c1.selectbox("Department", list(departments))
        treatment = c2.text_input("Planned treatment", placeholder="e.g. metformin 500 mg once daily")
        condition = st.text_input("Indication (optional)", placeholder="e.g. low-severity community-acquired pneumonia")
        with st.expander("De-identified patient context (optional)"):
            st.caption("Do not enter names, record numbers, dates of birth, addresses or contact details.")
            k1, k2, k3 = st.columns(3)
            age = k1.selectbox("Age band", ["Not stated", "18-39", "40-64", "65-79", "80+"])
            egfr = k2.selectbox("eGFR band (mL/min/1.73m2)", ["Not stated", ">=60", "45-59", "30-44", "15-29", "<15"])
            pregnancy = k3.selectbox("Pregnancy", ["Not stated", "Not pregnant", "Pregnant"])
            comorbidities = st.text_input("Relevant comorbidities (comma separated)", placeholder="e.g. mechanical heart valve")
            medications = st.text_input("Current medicines (comma separated)", placeholder="e.g. simvastatin 40 mg")
        submitted = st.form_submit_button("Check treatment", type="primary")

    if submitted:
        if not treatment.strip():
            st.warning("Enter the planned treatment.")
        else:
            try:
                raw = ClinicianQueryInput(
                    department=departments[department], treatment=treatment, condition=condition or None,
                    context=PatientContext(
                        age_band=None if age == "Not stated" else age,
                        egfr_band=None if egfr == "Not stated" else egfr,
                        pregnancy={"Not stated": None, "Not pregnant": False, "Pregnant": True}[pregnancy],
                        comorbidities=_split_terms(comorbidities),
                        current_medications=_split_terms(medications),
                    ),
                )
                with st.status("Running the six CKEA agents...", expanded=True) as status:
                    def on_step(step: StepRecord) -> None:
                        status.write(f"**{step.agent}**: {step.summary}")

                    answer = build_clinician_workflow(session_factory, config).run(raw, on_step=on_step)
                    status.update(label=f"Done: {answer.verdict_label}", state="complete", expanded=False)
                st.session_state["treatment_check_answer"] = answer.model_dump(mode="json")
            except PatientIdentifierError as e:
                st.error(f"Not submitted: {e}")
            except (TaxonomyError, ValueError) as e:
                st.error(f"Not submitted: {e}")

    stored = st.session_state.get("treatment_check_answer")
    if stored:
        render_answer(ClinicianAnswer.model_validate(stored))

    history = get_query_history(session_factory)
    if history:
        st.markdown("### Recent treatment checks")
        for row in history:
            label = (f"{str(row['created_at'])[:16]} · {taxonomy.department_name(row['department'])} · "
                     f"{row['treatment']} · {(row['verdict'] or 'no verdict').replace('_', ' ')}")
            if st.button(label, key=f"hist_{row['id']}", use_container_width=True):
                st.session_state["treatment_check_answer"] = row["answer"]
                st.rerun()
