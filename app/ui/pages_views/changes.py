"""Changes page."""

import html
from typing import Optional
import streamlit as st
from sqlalchemy.orm import (
    Session,
    sessionmaker,
)
from app.schemas.changes import ChangeStatus
from app.schemas.gaps import GapStatus
from app.services.config_service import (
    AppConfig,
    load_config,
)
from app.ui.queries import (
    get_source_documents,
    get_changes_with_gaps,
)
from app.ui.components import (
    format_version,
    summarize_source_diff,
    render_page_header,
    render_badge,
    render_processing_chain,
)


def render_changes_view(
    session_factory: sessionmaker[Session],
    config: Optional[AppConfig] = None,
) -> None:
    """Render Extracted Changes & Protocol Comparison view with explicit 3-section architecture."""
    cfg = config or load_config()
    render_page_header(
        breadcrumb="CLINICAL ANALYSIS › COMPARISON",
        title="Clinical Recommendations & Protocol Comparison",
        description="Side-by-side verification of external published evidence against institutional hospital baselines."
    )

    changes = get_changes_with_gaps(session_factory, config=cfg)
    if not changes:
        st.markdown(
            """
            <div class="ent-card ent-card-info">
                <div class="ent-metric-title">Comparison Workspace</div>
                <div style="font-size: 0.95rem; color: #334155; margin-top: 4px;">
                    No clinical recommendations are currently undergoing active comparison.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        docs = get_source_documents(session_factory)
        held_g1_docs = [d for d in docs if d["status"] == "held"]
        if held_g1_docs:
            st.warning(
                f"🛑 **Pipeline HELD at Human Gate G1:** {len(held_g1_docs)} monitored source document(s) "
                "could not yield actionable recommendations with sufficient confidence."
            )
            st.markdown(
                """
                **Why has Protocol Comparison not started?**
                - Source retrieval and file normalization succeeded with verified SHA-256 integrity.
                - LLM extraction did **not** produce a reviewable clinical recommendation meeting confidence (>= 0.70) or quote-verification rules.
                - Under CKEA safety policy:
                  - ⏸️ Comparison has **NOT** started
                  - ⏸️ Impact scoring has **NOT** started
                  - ⏸️ No ChangeBrief was created
                  - 🔵 Human extraction review is required at **Gate G1**
                """
            )
        return

    for c in changes:
        status_label = c["change_status"]
        is_held_g1 = status_label == ChangeStatus.HELD_FOR_G1.value
        is_held_g2 = status_label == ChangeStatus.HELD_FOR_G2.value or c["gap_status"] == GapStatus.REVIEW_REQUIRED.value
        is_held_g3 = status_label == ChangeStatus.HELD_FOR_G3.value or c["comparison_result"] == "no_match"

        expander_title = (
            f"Recommendation: {c['verbatim_text'][:70]}... "
            f"[{c['source_type']}: {c['source_identifier']}]"
        )

        with st.expander(expander_title, expanded=True):
            # Traceability Stepper
            gate_tag = "G1" if is_held_g1 else ("G2" if is_held_g2 else ("G3" if is_held_g3 else "G4"))
            render_processing_chain(
                current_stage="comparison" if not is_held_g1 else "extraction",
                held_gate=gate_tag if (is_held_g1 or is_held_g2 or is_held_g3) else None,
                pipeline_status="held" if (is_held_g1 or is_held_g2 or is_held_g3) else "in_progress",
                is_no_match=is_held_g3,
            )

            # ------------------------------------------------------------------
            # 1. SOURCE EVOLUTION (External Lineage Timeline)
            # ------------------------------------------------------------------
            st.markdown("#### 1. Source Evolution (external source, previous → current version)")
            if c.get("previous_source_version"):
                lineage_str = f"{format_version(c['previous_source_version'])} → {format_version(c['source_version'])}"
                changes_summary = summarize_source_diff(c.get("source_diff"))
            else:
                lineage_str = f"{format_version(c.get('source_version'))} (first version seen)"
                changes_summary = "No earlier version to compare against."
            st.markdown(
                f"""
                <div class="lineage-flow-box">
                    <strong>Source versions:</strong> {html.escape(lineage_str)}
                    <span class="badge-pill badge-neutral" style="margin-left: auto;">{html.escape(changes_summary)}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ------------------------------------------------------------------
            # 2. INSTITUTIONAL PROTOCOL COMPARISON (Side-by-Side Cards)
            # ------------------------------------------------------------------
            st.markdown("#### 2. Institutional Protocol Comparison")
            card_left, card_right = st.columns(2)

            with card_left:
                st.markdown(
                    f"""
                    <div class="ent-card" style="border-top: 3px solid #2563eb;">
                        <div class="ent-metric-title">External Evidence (Published)</div>
                        <div style="font-size: 0.95rem; font-style: italic; color: #1e293b; margin: 6px 0;">
                            "{html.escape(c['verbatim_text'])}"
                        </div>
                        <div style="font-size: 0.78rem; color: #475569; margin-top: 8px;">
                            • <strong>Type:</strong> {html.escape(str(c['recommendation_type']))}<br>
                            • <strong>Population:</strong> {html.escape(str(c['target_population']))}<br>
                            • <strong>Intervention:</strong> {html.escape(str(c['intervention']))}<br>
                            • <strong>Grade:</strong> {html.escape(str(c['evidence_grade']))} | <strong>Confidence:</strong> {c['confidence']:.2f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with card_right:
                if not c["has_gap"]:
                    st.info("Institutional comparison pending.")
                elif not c["is_match"] or c["comparison_result"] == "no_match":
                    st.markdown(
                        """
                        <div class="ent-card ent-card-warning" style="border-top: 3px solid #dd6b20;">
                            <div class="ent-metric-title">Institutional Protocol Baseline</div>
                            <div style="font-size: 0.95rem; color: #b45309; margin: 6px 0;">
                                No matching institutional protocol section found in hospital library.
                            </div>
                            <div style="font-size: 0.78rem; color: #78350f;">
                                Gate G3 committee review required to determine if a new protocol must be authored.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                elif not c.get("protocol_resolved", True):
                    st.warning(c["protocol_status_message"])
                else:
                    st.markdown(
                        f"""
                        <div class="ent-card" style="border-top: 3px solid #059669;">
                            <div class="ent-metric-title">Institutional Protocol Baseline</div>
                            <div style="font-size: 0.95rem; font-style: italic; color: #1e293b; margin: 6px 0;">
                                "{html.escape(str(c['protocol_section_text']))}"
                            </div>
                            <div style="font-size: 0.78rem; color: #475569; margin-top: 8px;">
                                • <strong>Protocol:</strong> {html.escape(str(c['matched_protocol_id']))} ({html.escape(format_version(c['matched_protocol_version']))})<br>
                                • <strong>Section:</strong> {html.escape(str(c['protocol_section_id']))} — {html.escape(str(c['protocol_section_heading']))}<br>
                                • <strong>Title:</strong> {html.escape(str(c['protocol_title']))}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # ------------------------------------------------------------------
            # 3. COMPARISON RESULT (Synthesized Gap Finding)
            # ------------------------------------------------------------------
            st.markdown("#### 3. Comparison Result & Gap Synthesis")
            if not c["has_gap"]:
                st.info("Pending comparison execution.")
            elif not c["is_match"] or c["comparison_result"] == "no_match":
                st.error(f"🟠 **No Matching Protocol Section** {render_badge('G3')} — Committee review required.")
            else:
                col_res1, col_res2 = st.columns([1, 2])
                with col_res1:
                    st.write(f"**Result:** `{c['comparison_result'].upper()}`")
                    st.write(f"**Difference Category:** `{c['difference_type']}`")
                    st.write(f"**Confidence:** `{c['comparison_confidence'] if c['comparison_confidence'] is not None else 'N/A'}`")
                    st.write(f"**Contradiction:** `{c['contradiction_status']}`")
                with col_res2:
                    st.markdown("**Specific Clinical Difference:**")
                    st.markdown(f"> {c['specific_difference']}")

                    if is_held_g2:
                        st.warning("🟡 **G2 Safety Gate:** Ambiguous comparison. Human clinical review required.")
                    elif c["comparison_result"] == "gap":
                        st.info("✅ **Confirmed Gap:** Proceeded to deterministic impact scoring.")
                    elif c["comparison_result"] == "no_gap":
                        st.success("✅ **No Material Gap:** Source evidence aligns with hospital protocol.")
