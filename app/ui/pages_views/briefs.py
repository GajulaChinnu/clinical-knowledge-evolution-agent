"""Briefs page."""

import html
import streamlit as st
from sqlalchemy.orm import (
    Session,
    sessionmaker,
)
from app.ui.queries import (
    get_brief_summaries,
    get_brief_full_payload,
)
from app.ui.components import (
    render_page_header,
    render_processing_chain,
    render_brief_sections,
)


def render_briefs_view(session_factory: sessionmaker[Session], department=None) -> None:
    """Render 7-Section Change Briefs formatted as Executive Review Documents."""
    render_page_header(
        breadcrumb="GOVERNANCE › CHANGE BRIEFS",
        title="Clinical Change Briefs",
        description="Audit-grade evidence briefs prepared for authorized hospital governance review."
    )

    briefs = get_brief_summaries(session_factory, department=department)
    if not briefs:
        st.markdown(
            """
            <div class="ent-card ent-card-info">
                <div class="ent-metric-title">Briefs Console</div>
                <div style="font-size: 0.95rem; color: #334155; margin-top: 4px;">
                    No ChangeBriefs generated yet.
                </div>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 6px;">
                    Briefs are generated when a confirmed gap passes multidimensional impact scoring.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    selected_brief_id = st.selectbox(
        "Select Change Brief:",
        options=[b["brief_id"] for b in briefs],
        format_func=lambda b_id: f"{b_id[:8]}... | Status: {next(b['status'] for b in briefs if b['brief_id'] == b_id)} | Tier: {next(b['tier'] for b in briefs if b['brief_id'] == b_id)} | Source: {next(b['source_identifier'] for b in briefs if b['brief_id'] == b_id)}",
    )

    if not selected_brief_id:
        return

    current_brief_info = next(b for b in briefs if b["brief_id"] == selected_brief_id)

    # ------------------------------------------------------------------
    # EXECUTIVE REVIEW DOCUMENT HEADER (4 Primary Summary Metrics)
    # ------------------------------------------------------------------
    st.markdown(
        f"""
        <div class="ent-card" style="border-left: 4px solid #1e40af; margin-top: 8px;">
            <div class="ent-metric-title">Executive Governance Dossier</div>
            <div style="display: flex; gap: 16px; flex-wrap: wrap; margin-top: 8px;">
                <div><strong>Change Detected:</strong> {html.escape(current_brief_info['difference_summary'][:60])}</div>
                <div><strong>Impact Tier:</strong> {html.escape(str(current_brief_info['tier']))}</div>
                <div><strong>Protocol Baseline:</strong> {html.escape(str(current_brief_info['matched_protocol']))}</div>
                <div><strong>Review Status:</strong> {html.escape(str(current_brief_info['status']).title())}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_processing_chain(
        current_stage="briefing",
        held_gate="G4",
        pipeline_status=current_brief_info["status"],
    )

    full_brief = get_brief_full_payload(session_factory, selected_brief_id)
    if not full_brief:
        st.error("Failed to load brief payload.")
        return

    payload = full_brief["structured_payload"]

    tab1, tab2, tab3 = st.tabs(["📑 Seven Required Sections", "🌐 Rendered HTML", "📄 Raw Markdown / JSON"])

    with tab1:
        render_brief_sections(payload)

    with tab2:
        if full_brief.get("html_content"):
            st.components.v1.html(full_brief["html_content"], height=600, scrolling=True)
        else:
            st.info("HTML companion file not available.")

    with tab3:
        st.markdown("#### Companion Markdown")
        st.code(full_brief.get("md_content") or "Markdown not available", language="markdown")
        st.markdown("#### Companion JSON Structured Payload")
        st.json(payload)
