"""Governance page."""

from datetime import (
    datetime,
    timezone,
    timedelta,
)
import html
import streamlit as st
from sqlalchemy.orm import (
    Session,
    sessionmaker,
)
from app.agents.governance_agent import GovernanceAgent
from app.ui.queries import (
    resolve_acting_reviewer,
    get_brief_summaries,
    get_brief_full_payload,
    execute_governance_action,
)
from app.ui.components import (
    format_version,
    render_page_header,
)


def render_governance_view(session_factory: sessionmaker[Session], governance_agent: GovernanceAgent, department=None) -> None:
    """Render Human Governance Decision View as an Enterprise Review Console (Gate G4)."""
    render_page_header(
        breadcrumb="DECISION DESK › GOVERNANCE (G4)",
        title="Clinical Governance Workspace",
        description="Authorized clinicians review evidence briefs and record formal, immutable governance decisions."
    )

    briefs = get_brief_summaries(session_factory, department=department)
    if not briefs:
        st.markdown(
            """
            <div class="ent-card ent-card-info">
                <div class="ent-metric-title">Governance Review Desk</div>
                <div style="font-size: 0.95rem; color: #334155; margin-top: 4px;">
                    No ChangeBriefs currently awaiting governance review.
                </div>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 6px;">
                    Change briefs appear here after passing extraction, comparison, and impact scoring.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    brief_options = [b["brief_id"] for b in briefs]
    selected_brief_id = st.selectbox(
        "Select Brief for Governance Review:",
        options=brief_options,
        format_func=lambda b_id: f"{b_id[:8]}... | Status: {next(b['status'] for b in briefs if b['brief_id'] == b_id)} | Reviewer: {next(b['assigned_reviewer'] for b in briefs if b['brief_id'] == b_id)} | Overdue: {'🚨 YES' if next(b['is_overdue'] for b in briefs if b['brief_id'] == b_id) else 'NO'}",
    )

    current_brief = next(b for b in briefs if b["brief_id"] == selected_brief_id)

    # ------------------------------------------------------------------
    # ASYMMETRIC SPLIT LAYOUT: LEFT ~70% (Dossier) | RIGHT ~30% (Decision Panel)
    # ------------------------------------------------------------------
    col_dossier, col_panel = st.columns([7, 3])

    with col_dossier:
        st.markdown("### Evidence")

        # Card 1: Source Evidence
        st.markdown(
            f"""
            <div class="ent-card" style="border-top: 3px solid #2563eb;">
                <div class="ent-metric-title">External Source Evidence</div>
                <div style="font-size: 0.95rem; font-style: italic; color: #1e293b; margin: 6px 0;">
                    "{html.escape(current_brief['verbatim_recommendation'])}"
                </div>
                <div style="font-size: 0.78rem; color: #475569;">
                    Source: <strong>{html.escape(current_brief['source_identifier'])}</strong> ({html.escape(current_brief['source_type'])})
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Card 2: Hospital Protocol Baseline
        st.markdown(
            f"""
            <div class="ent-card" style="border-top: 3px solid #059669;">
                <div class="ent-metric-title">Institutional Protocol Baseline</div>
                <div style="font-size: 0.85rem; color: #1e293b; margin: 4px 0;">
                    Protocol ID: <strong>{html.escape(str(current_brief['matched_protocol']))}</strong> ({html.escape(format_version(current_brief['protocol_version']))})
                </div>
                <div style="font-size: 0.85rem; color: #475569;">
                    Section: <strong>{html.escape(str(current_brief['protocol_section']))}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Card 3: Difference & Impact
        st.markdown(
            f"""
            <div class="ent-card">
                <div class="ent-metric-title">Clinical Difference & Risk Triage</div>
                <div style="font-size: 0.88rem; color: #1e293b; margin: 4px 0;">
                    <strong>Specific Difference:</strong> {html.escape(current_brief['difference_summary'])}
                </div>
                <div style="font-size: 0.8rem; color: #475569; margin-top: 4px;">
                    Impact Tier: <strong>{html.escape(str(current_brief['tier']))}</strong> | SLA Deadline: <strong>{html.escape(str(current_brief['sla_deadline']))}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Deep-dive expanders
        full_brief = get_brief_full_payload(session_factory, selected_brief_id)
        if full_brief and "structured_payload" in full_brief:
            payload = full_brief["structured_payload"]
            with st.expander("Source Excerpt", expanded=False):
                st.write((payload.get("source_excerpt") or {}).get("source_excerpt") or "Not available.")
            with st.expander("Current Protocol Text", expanded=False):
                current = payload.get("current_protocol") or {}
                st.write(current.get("exact_protocol_text") or current.get("no_match_statement") or "Not available.")

    with col_panel:
        st.markdown("### Reviewer Decision")

        # Status & Overdue check
        if current_brief["is_overdue"]:
            st.markdown(
                """
                <div class="ent-card ent-card-urgent">
                    <div class="ent-metric-title" style="color: #b91c1c;">Review Status: Overdue</div>
                    <div style="font-size: 0.82rem; color: #7f1d1d;">🚨 G5 SLA Escalation Active</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="ent-card">
                    <div class="ent-metric-title">Reviewer & Status</div>
                    <div style="font-size: 0.85rem; color: #0f172a;">Assigned: <strong>{html.escape(current_brief['assigned_reviewer'])}</strong></div>
                    <div style="font-size: 0.78rem; color: #64748b;">Status: {html.escape(str(current_brief['status']).title())}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.form("governance_review_action_form"):
            action = st.radio(
                "Governance Decision:",
                options=["approve", "reject", "defer"],
                format_func=lambda a: {
                    "approve": "Approve",
                    "reject": "Reject",
                    "defer": "Defer",
                }.get(a, a),
                horizontal=True,
            )

            acting, identity_mode, identity_message = resolve_acting_reviewer(
                governance_agent.auth_service, governance_agent.config
            )
            if identity_mode == "demo":
                st.warning(identity_message)
                registered = governance_agent.auth_service.list_reviewers()
                reviewer_id = st.selectbox(
                    "Reviewer (demo identity):",
                    options=[r.reviewer_id for r in registered],
                    format_func=lambda rid: next(f"{r.name} — {r.role}" for r in registered if r.reviewer_id == rid),
                )
            elif acting is not None:
                reviewer_id = acting.reviewer_id
                st.write(f"**Signed in as:** {acting.name} — {acting.role}")
            else:
                reviewer_id = None
                st.error(identity_message)

            rationale = st.text_area(
                "Clinical Rationale (Mandatory):",
                placeholder="Enter clinical rationale, evidence justification, or committee discussion...",
                height=110,
            )

            follow_up_date_input = st.date_input(
                "Deferral Follow-up Date (If Deferring):",
                value=datetime.now(timezone.utc).date() + timedelta(days=14),
            )

            submit_decision = st.form_submit_button("Record Governance Decision", type="primary", use_container_width=True)

            if submit_decision and not reviewer_id:
                st.error("No authenticated, registered reviewer: decision not recorded.")
            elif submit_decision:
                follow_up_dt = None
                if action == "defer":
                    follow_up_dt = datetime.combine(
                        follow_up_date_input, datetime.min.time(), tzinfo=timezone.utc
                    )

                try:
                    result = execute_governance_action(
                        governance_agent=governance_agent,
                        brief_id=selected_brief_id,
                        action=action,
                        reviewer_id=reviewer_id,
                        rationale=rationale,
                        follow_up_date=follow_up_dt,
                    )
                    st.toast(f"Governance decision recorded for {selected_brief_id}")
                    # Never imply protocol update was approved automatically
                    st.success(f"Governance decision recorded: {action.upper()}. Audit log updated.")
                except Exception as e:
                    st.error(f"Governance action rejected: {e}")
