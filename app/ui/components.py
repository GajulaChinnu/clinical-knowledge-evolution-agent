"""Shared presentation components for the CKEA workspace."""

import logging
import html
from pathlib import Path
from typing import (
    Any,
    Dict,
    Optional,
)
import streamlit as st
from app.services.pdf_parser import extract_page_texts

logger = logging.getLogger("ckea.ui.components")


def format_version(version: Optional[str]) -> str:
    """Display a version label once-prefixed ('1.0' -> 'v1.0', 'v1.0' stays 'v1.0')."""
    if not version:
        return "unknown version"
    text = str(version).strip()
    return text if text.lower().startswith("v") else f"v{text}"


def summarize_source_diff(source_diff: Optional[Dict[str, Any]]) -> str:
    """One-line summary of a persisted source-evolution diff."""
    changes = (source_diff or {}).get("changes") or []
    if not changes:
        return "No section-level diff recorded"
    counts: Dict[str, int] = {}
    for change in changes:
        counts[change["change_type"]] = counts.get(change["change_type"], 0) + 1
    return ", ".join(f"{n} {kind}" for kind, n in sorted(counts.items()))


def render_page_header(breadcrumb: str, title: str, description: str) -> None:
    """Render consistent enterprise page header with restrained typography."""
    st.markdown(
        f"""
        <div class="page-breadcrumb">{html.escape(breadcrumb)}</div>
        <h1 class="page-title">{html.escape(title)}</h1>
        <div class="page-description">{html.escape(description)}</div>
        """,
        unsafe_allow_html=True,
    )


def render_badge(gate: str) -> str:
    """Return a standardized markdown badge for a given gate."""
    badges = {
        "G1": "🔹 **G1 — Extraction Quality Review**",
        "G2": "🔹 **G2 — Ambiguous Comparison Review**",
        "G3": "🔹 **G3 — Unmatched Clinical Finding**",
        "G4": "🔸 **G4 — Authorized Governance Sign-off**",
        "G5": "🔺 **G5 — SLA Escalation (Overdue)**"
    }
    return badges.get(gate.upper(), f"**{gate}**")


def render_processing_chain(
    current_stage: str = "monitoring",
    held_gate: Optional[str] = None,
    pipeline_status: str = "held",
    is_no_match: bool = False,
    gate: Optional[str] = None,
    status: Optional[str] = None,
) -> str:
    """Render a compact horizontal pipeline stepper."""
    eff_gate = (gate or held_gate or "").upper()
    eff_status = (status or pipeline_status).lower()
    stage = (current_stage or "").lower()

    stages = ["Ingestion", "Extraction", "Comparison", "Impact", "Brief", "Governance"]

    # Determine the index of the active stage
    if eff_gate == "G1" or (eff_status == "held" and stage == "extraction"):
        active_idx = 1
    elif eff_gate in ["G2", "G3"] or (eff_status == "held" and stage == "comparison"):
        active_idx = 2
    elif eff_gate == "G4" or stage in ("briefing", "governance", "draft", "assigned", "in_review") or eff_status == "pending":
        active_idx = 5
    elif stage in ("closure", "closed", "decided") or eff_status == "complete":
        active_idx = 6  # all done
    else:
        try:
            active_idx = stages.index(stage.capitalize())
        except ValueError:
            active_idx = 0

    steps = []
    for i, s in enumerate(stages):
        if i < active_idx:
            steps.append(f"✅ ~{s}~")
        elif i == active_idx:
            gate_label = f" ({eff_gate})" if eff_gate else ""
            steps.append(f"**🟢 {s}{gate_label}**")
        else:
            steps.append(f"⏸️ {s}")

    chain_md = " ➔ ".join(steps)
    if eff_gate:
        chain_md += f"  \n\n*Held at:* {render_badge(eff_gate)}"

    st.markdown(chain_md)
    return chain_md


def render_header() -> None:
    """Render compact enterprise clinical governance boundary notice."""
    st.markdown(
        """
        <div class="gov-notice-strip">
            <span style="margin-right: 8px; font-size: 1rem;">🛡️</span>
            <div><strong>The agent prepares. Clinicians decide.</strong> CKEA monitors guidance, detects changes, compares them with
            hospital protocols and answers from grounded source text. It never modifies a protocol or makes a clinical or governance decision.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_gate_legend() -> None:
    """Render compact safety gate reference in sidebar."""
    st.sidebar.markdown(
        """
        <div style="margin-top: 1rem; border-top: 1px solid #e2e8f0; padding-top: 1rem;">
            <div style="font-size: 0.7rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.6rem;">Safety Gates</div>
            <div style="display: flex; flex-direction: column; gap: 0.4rem; font-size: 0.8rem;">
                <div style="display: flex; align-items: center; gap: 0.4rem;">
                    <span class="badge-pill badge-neutral" style="min-width: 28px; text-align: center; padding: 0.1rem 0.3rem;">G1</span>
                    <span style="color: #475569; font-weight: 500;">Extraction Review</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.4rem;">
                    <span class="badge-pill badge-neutral" style="min-width: 28px; text-align: center; padding: 0.1rem 0.3rem;">G2</span>
                    <span style="color: #475569; font-weight: 500;">Comparison Review</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.4rem;">
                    <span class="badge-pill badge-neutral" style="min-width: 28px; text-align: center; padding: 0.1rem 0.3rem;">G3</span>
                    <span style="color: #475569; font-weight: 500;">Unmatched Protocol</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.4rem;">
                    <span class="badge-pill badge-amber" style="min-width: 28px; text-align: center; padding: 0.1rem 0.3rem;">G4</span>
                    <span style="color: #475569; font-weight: 500;">Governance Decision</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.4rem;">
                    <span class="badge-pill badge-red" style="min-width: 28px; text-align: center; padding: 0.1rem 0.3rem;">G5</span>
                    <span style="color: #475569; font-weight: 500;">SLA Escalation</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_upload_result_card(res: Dict[str, Any]) -> None:
    """Render the structured outcome card after PDF or URL upload and processing."""
    st.markdown("---")
    input_type = res.get("input_type", "URL" if res.get("source_url") else "PDF")

    # Bot-check / Content challenge handling
    if not res.get("usable_clinical_content", True):
        st.error(f"❌ [{input_type}] Ingestion Blocked: Browser Challenge / Bot-Check Detected")
        st.warning(
            "🛑 **Source Content Validation Policy Violation**\n\n"
            f"**Reason:** {res.get('result_message')}\n\n"
            "- **Retrieved (HTTP Connection):** YES\n"
            "- **Usable Clinical Content:** NO\n"
            "- **Action Taken:** Challenge page rejected before normalization. Zero challenge data sent to extraction or LLM.\n"
            "- **Anti-Bot Policy:** CKEA strictly does not bypass bot protection challenges or store cookie wall pages."
        )
        return

    gate = res.get("gate_status")
    success = res.get("success", True)
    msg = res.get("result_message", "")
    change_status = res.get("change_status", "new_source")
    source_ver = res.get("source_version", "1.0")
    pipe_status = res.get("pipeline_status", "Held").capitalize()

    st.markdown("#### Ingestion & Processing Outcome")

    # Status notice treatment:
    # Important: "No source change detected" must NOT look like a green success alert. It is a normal informational state.
    is_no_change = "no source change" in msg.lower() or change_status == "no_change"

    if not success:
        st.error(f"❌ [{input_type}] {msg}")
        if res.get("error"):
            st.caption(f"Details: {res.get('error')}")
    elif is_no_change:
        st.info(f"ℹ️ [{input_type}] {msg}")
    elif gate == "G1":
        st.info(f"🔵 **G1 Safety Gate:** {msg}")
    elif gate == "G2":
        st.warning(f"🟡 **G2 Safety Gate:** {msg}")
    elif gate == "G3":
        st.warning(f"🟠 **G3 Safety Gate:** {msg}")
    elif gate == "G4":
        st.info(f"🔴 **G4 Safety Gate:** {msg}")
    else:
        st.success(f"✅ [{input_type}] {msg}")

    # Compact Badges Strip: RETRIEVAL STATUS | PIPELINE STATUS | SOURCE VERSION | CHANGE STATUS | GATE
    retrieval_status = res.get("ingestion_status", "Complete")
    gate_badge_label = gate if gate else "None"
    change_readable = change_status.replace("_", " ").title()

    st.markdown(
        f"""
        <div style="display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0 12px 0;">
            <span class="badge-pill badge-neutral">RETRIEVAL: {html.escape(retrieval_status)}</span>
            <span class="badge-pill badge-blue">PIPELINE: {html.escape(pipe_status)}</span>
            <span class="badge-pill badge-neutral">VERSION: v{html.escape(str(source_ver))}</span>
            <span class="badge-pill badge-neutral">CHANGE: {html.escape(change_readable)}</span>
            <span class="badge-pill {'badge-amber' if gate else 'badge-neutral'}">GATE: {html.escape(gate_badge_label)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    resolved_url = res.get("resolved_source_url")
    if resolved_url:
        st.caption(f"🔗 **Resolved Full-Text URL:** [{resolved_url}]({resolved_url})")
    elif res.get("source_url"):
        st.caption(f"🔗 **Original Source URL:** `{res['source_url']}`")

    if res.get("title"):
        st.markdown(f"**Document Title:** {res['title']}")

    # Render processing chain stepper
    render_processing_chain(
        current_stage=res.get("current_stage", "extraction"),
        held_gate=res.get("gate_status"),
        pipeline_status=res.get("pipeline_status", "held"),
        is_no_match=(res.get("gate_status") == "G3"),
    )

    # Secondary details in compact expander
    with st.expander("Technical Provenance & Document Details", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Document ID", (res.get("document_id") or "N/A")[:12] + "...")
        with c2:
            st.metric("Content Type", res.get("content_classification", "Web Page"))
        with c3:
            st.metric("Extracted Size", f"{res.get('extracted_text_size', 0):,} B")
        with c4:
            st.metric("SHA-256", (res.get("sha256_hash", "")[:12] + "...") if res.get("sha256_hash") else "N/A")

        if gate == "G1":
            st.info(
                "🛑 **Why paused at G1?** "
                f"{res.get('pipeline_reason') or 'No actionable clinical recommendation could be confidently extracted.'} "
                "Source is saved with verified SHA-256 integrity, but extraction requires authorized human review."
            )

    saved_path_str = res.get("saved_path")
    if saved_path_str and Path(saved_path_str).exists():
        try:
            page_texts = extract_page_texts(Path(saved_path_str))
            if page_texts:
                with st.expander("📄 Document Text Preview", expanded=False):
                    for pt in page_texts[:2]:
                        st.markdown(f"**Page {pt.page_number}**")
                        snippet = pt.text[:600] + ("..." if len(pt.text) > 600 else "")
                        st.text(snippet if snippet.strip() else "[Empty or non-text page]")
        except Exception as e:
            logger.debug("Read-only preview generation skipped: %s", e)


def render_brief_sections(payload: Dict[str, Any]) -> None:
    """Render the seven required Change Brief sections from the StructuredBriefPayload schema."""
    what = payload.get("what_changed") or {}
    current = payload.get("current_protocol") or {}
    diff = payload.get("specific_difference") or {}
    impact = payload.get("impact_assessment") or {}
    workflows = payload.get("affected_workflows") or {}
    excerpt = payload.get("source_excerpt") or {}

    with st.expander("1. What Changed (external source)", expanded=True):
        st.write(f"**Verbatim Recommendation:** {what.get('recommendation_text')}")
        st.write(f"**Source:** `{what.get('source_identifier')}` · page {what.get('page')} · {what.get('section')}")
        st.write(f"**Target Population:** {what.get('target_population')}")
        st.write(f"**Intervention:** {what.get('intervention')}")
        st.write(f"**Evidence Grade:** {what.get('evidence_grade') or 'Not stated in source'}")
        st.write(f"**Extraction Confidence:** {what.get('extraction_confidence')}")

    with st.expander("2. Current Institutional Protocol", expanded=True):
        if current.get("is_match"):
            st.write(
                f"**Protocol:** `{current.get('protocol_id')}` version `{current.get('protocol_version')}` · "
                f"section `{current.get('section_id')}` — {current.get('section_heading')}"
            )
            st.info(current.get("exact_protocol_text") or "Protocol text not available.")
        else:
            st.warning(current.get("no_match_statement") or "No matching protocol section.")

    with st.expander("3. Specific Difference", expanded=True):
        st.write(f"**Difference Type:** `{diff.get('difference_type')}` · **Result:** `{diff.get('comparison_result')}`")
        st.write(f"**Comparison Confidence:** {diff.get('comparison_confidence')}")
        st.write(diff.get("specific_difference"))

    with st.expander("4. Impact Assessment", expanded=False):
        if impact.get("is_complete"):
            st.write(
                f"**Tier:** `{impact.get('tier')}` · **Score:** `{impact.get('total_score')}/15` · "
                f"**SLA:** {impact.get('sla_hours')}h (deadline `{impact.get('sla_deadline')}`) · "
                f"**Route:** {impact.get('routing_target')}"
            )
            st.write(f"- Clinical urgency {impact.get('clinical_urgency')}/5 — {impact.get('urgency_basis')}")
            st.write(f"- Evidence strength {impact.get('evidence_strength')}/5 — {impact.get('evidence_basis')}")
            st.write(f"- Pathway breadth {impact.get('pathway_breadth')}/5 — {impact.get('breadth_basis')}")
            st.caption(f"Rules {impact.get('rule_ids')} · scoring version {impact.get('scoring_yaml_version')}")
        else:
            st.warning(f"Impact incomplete — no tier or SLA assigned. {impact.get('incomplete_reason') or ''}")

    with st.expander("5. Affected Workflows", expanded=False):
        if workflows.get("is_available"):
            st.write(", ".join(workflows.get("affected_workflows") or []) or "None listed.")
            if workflows.get("workflow_summary"):
                st.write(workflows["workflow_summary"])
        else:
            st.warning(workflows.get("unavailability_reason") or "Workflow information unavailable.")

    with st.expander("6. Proposed Review Actions", expanded=False):
        for item in payload.get("proposed_actions") or []:
            st.markdown(f"{item.get('step')}. **{item.get('action')}** — {item.get('description')}")

    with st.expander("7. Source Excerpt", expanded=False):
        st.info(
            f"**{excerpt.get('source_identifier')}** · page {excerpt.get('page')} · {excerpt.get('section')}\n\n"
            f"> {excerpt.get('source_excerpt')}"
        )
