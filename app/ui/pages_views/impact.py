"""Impact page."""

import html
import streamlit as st
from sqlalchemy.orm import (
    Session,
    sessionmaker,
)
from app.ui.queries import get_impact_records
from app.ui.components import render_page_header


def render_impact_view(session_factory: sessionmaker[Session]) -> None:
    """Render Deterministic Multidimensional Impact Assessments with compact metric cards."""
    render_page_header(
        breadcrumb="RISK TRIAGE › IMPACT ASSESSMENT",
        title="Deterministic Impact Assessments",
        description="Clinical risk scoring evaluated against YAML rule matrix. Zero LLM calculations."
    )

    impacts = get_impact_records(session_factory)
    if not impacts:
        st.markdown(
            """
            <div class="ent-card ent-card-info">
                <div class="ent-metric-title">Impact Assessment Workspace</div>
                <div style="font-size: 0.95rem; color: #334155; margin-top: 4px;">
                    No impact assessment records found.
                </div>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 6px;">
                    Impact Assessment executes after Protocol Comparison confirms a clinical gap.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for imp in impacts:
        tier_str = str(imp["tier"])
        badge_class = {
            "Critical": "badge-red",
            "High": "badge-amber",
            "Standard": "badge-blue",
            "Low": "badge-green",
        }.get(tier_str, "badge-neutral")

        with st.expander(f"Impact Tier: {tier_str} (Score: {imp['total_score']}/15) — {imp['source_identifier']} vs {imp['matched_protocol_id']}", expanded=True):
            # Top metadata bar
            st.markdown(
                f"""
                <div style="display: flex; gap: 8px; align-items: center; margin-bottom: 10px;">
                    <span class="badge-pill {badge_class}">TIER: {html.escape(tier_str.upper())}</span>
                    <span class="badge-pill badge-neutral">SLA: {html.escape(str(imp['sla_deadline'])[:19])}</span>
                    <span class="badge-pill badge-neutral">PROTOCOL: {html.escape(str(imp['matched_protocol_id']))}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Compact 5-Metric Tiles Row: Urgency, Evidence, Breadth, Total Score, Impact Tier
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.markdown(
                    f"""
                    <div class="ent-card">
                        <div class="ent-metric-title">Urgency</div>
                        <div class="ent-metric-val">{imp['clinical_urgency']}<span style="font-size: 0.9rem; color: #64748b;"> / 5</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f"""
                    <div class="ent-card">
                        <div class="ent-metric-title">Evidence</div>
                        <div class="ent-metric-val">{imp['evidence_strength']}<span style="font-size: 0.9rem; color: #64748b;"> / 5</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f"""
                    <div class="ent-card">
                        <div class="ent-metric-title">Breadth</div>
                        <div class="ent-metric-val">{imp['pathway_breadth']}<span style="font-size: 0.9rem; color: #64748b;"> / 5</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m4:
                st.markdown(
                    f"""
                    <div class="ent-card">
                        <div class="ent-metric-title">Total Score</div>
                        <div class="ent-metric-val">{imp['total_score']}<span style="font-size: 0.9rem; color: #64748b;"> / 15</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m5:
                st.markdown(
                    f"""
                    <div class="ent-card">
                        <div class="ent-metric-title">Impact Tier</div>
                        <div class="ent-metric-val" style="font-size: 1.15rem;">{html.escape(tier_str)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Recommendation Excerpt
            st.markdown(f"**Extracted Recommendation:** \"{imp['verbatim_recommendation']}\"")

            # Expandable Scoring Engine Basis & Traceability
            with st.expander("Scoring Basis & Deterministic Rules", expanded=False):
                st.write(f"**Urgency Basis:** {imp['urgency_basis']}")
                st.write(f"**Evidence Basis:** {imp['evidence_basis']}")
                st.write(f"**Breadth Basis:** {imp['breadth_basis']}")
                st.write(f"**Applied Rule IDs:** `{', '.join(imp['rule_ids'])}`")
                st.write(f"**Scoring YAML Version:** `{imp['scoring_yaml_version']}`")
