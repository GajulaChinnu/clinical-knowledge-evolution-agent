"""Dashboard page."""

import streamlit as st
from sqlalchemy.orm import (
    Session,
    sessionmaker,
)
from app.ui.queries import (
    get_dashboard_metrics,
    get_source_documents,
)
from app.ui.components import render_page_header


def render_dashboard(session_factory: sessionmaker[Session], department=None) -> None:
    """Render Dashboard overview metrics as an Executive Attention Center."""
    render_page_header(
        breadcrumb="OVERVIEW › ATTENTION CENTER",
        title="Governance & Pipeline Dashboard",
        description="Real-time clinical review triage, SLA compliance, and surveillance operations."
    )
    metrics = get_dashboard_metrics(session_factory)

    # ------------------------------------------------------------------
    # SECTION 1: NEEDS ATTENTION (Top Priority Viewport)
    # ------------------------------------------------------------------
    st.markdown("### Needs Attention")

    col_urgent1, col_urgent2 = st.columns(2)
    with col_urgent1:
        if metrics["overdue_reviews"] > 0:
            st.markdown(
                f"""
                <div class="ent-card ent-card-urgent">
                    <div class="ent-metric-title">Overdue SLA Reviews (G5 Escalated)</div>
                    <div class="ent-metric-val" style="color: #b91c1c;">{metrics['overdue_reviews']}</div>
                    <div class="ent-metric-sub">Immediate committee or CMO review required</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="ent-card ent-card-info">
                    <div class="ent-metric-title">Overdue SLA Reviews (G5)</div>
                    <div class="ent-metric-val" style="color: #15803d;">0</div>
                    <div class="ent-metric-sub">All assigned reviews are within SLA target</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_urgent2:
        if metrics["awaiting_governance"] > 0:
            st.markdown(
                f"""
                <div class="ent-card ent-card-warning">
                    <div class="ent-metric-title">Pending Governance Decisions (G4)</div>
                    <div class="ent-metric-val" style="color: #b45309;">{metrics['awaiting_governance']}</div>
                    <div class="ent-metric-sub">Evidence briefs awaiting clinician sign-off</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="ent-card ent-card-info">
                    <div class="ent-metric-title">Pending Governance Decisions (G4)</div>
                    <div class="ent-metric-val" style="color: #15803d;">0</div>
                    <div class="ent-metric-sub">No briefs currently awaiting formal decision</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    from app.ui.queries import get_detected_changes

    feed = get_detected_changes(session_factory, department=department)[:8]
    st.markdown("### Priority feed (ranked source changes)")
    if feed:
        st.dataframe([{
            "Priority": r["priority"], "Change": r["category"].replace("_", " "), "Source": r["source"],
            "Version": f"{r['previous_version'] or '-'} -> {r['latest_version']}", "Treatments": ", ".join(r["treatments"]),
            "Departments": ", ".join(r["departments"]),
        } for r in feed], use_container_width=True, hide_index=True)
    else:
        st.caption("No ranked changes yet. Run a watchlist check.")

    st.markdown("### Human Review Queues")
    col_q1, col_q2, col_q3 = st.columns(3)
    with col_q1:
        st.markdown(
            f"""
            <div class="ent-card">
                <div class="ent-metric-title">Held at G1 (Extraction)</div>
                <div class="ent-metric-val">{metrics['held_g1']}</div>
                <div class="ent-metric-sub">Low confidence or unextracted sources</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_q2:
        st.markdown(
            f"""
            <div class="ent-card">
                <div class="ent-metric-title">Held at G2 (Comparison)</div>
                <div class="ent-metric-val">{metrics['held_g2']}</div>
                <div class="ent-metric-sub">Ambiguous diffs requiring human check</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_q3:
        st.markdown(
            f"""
            <div class="ent-card">
                <div class="ent-metric-title">Held at G3 (No Match)</div>
                <div class="ent-metric-val">{metrics['held_g3']}</div>
                <div class="ent-metric-sub">New external evidence without baseline</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------
    # SECTION 2: RECENT MONITORED SOURCE CHANGES
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Recent Source Activity")
    docs = get_source_documents(session_factory)
    if docs:
        recent_df = [
            {
                "Source Identifier": d["display_name"],
                "Version": f"v{d['source_version']}",
                "Change Status": d["change_status"].replace("_", " ").title(),
                "Pipeline Status": d["status"].title(),
                "Changes Extracted": d["change_records_count"],
                "Last Retrieved": str(d["last_retrieved"])[:19],
            }
            for d in docs[:5]
        ]
        st.dataframe(
            recent_df,
            use_container_width=True,
            column_config={
                "Source Identifier": st.column_config.TextColumn("Source", width="medium"),
                "Version": st.column_config.TextColumn("Version", width="small"),
                "Change Status": st.column_config.TextColumn("Change", width="small"),
                "Pipeline Status": st.column_config.TextColumn("Pipeline", width="small"),
                "Changes Extracted": st.column_config.NumberColumn("Extracted", format="%d", width="small"),
                "Last Retrieved": st.column_config.TextColumn("Last Retrieved", width="medium"),
            },
        )
    else:
        st.info("No sources have been ingested yet.")

    # ------------------------------------------------------------------
    # SECTION 3: OPERATIONAL & GOVERNANCE SUMMARY
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Operational Metrics")
    col_op1, col_op2, col_op3, col_op4 = st.columns(4)
    with col_op1:
        st.metric("Total Monitored Sources", metrics["total_documents"])
    with col_op2:
        st.metric("Sources Processed", metrics["processed_documents"])
    with col_op3:
        st.metric("Decided Briefs", metrics["decided_briefs"])
    with col_op4:
        st.metric("Closed Briefs", metrics["closed_briefs"])
