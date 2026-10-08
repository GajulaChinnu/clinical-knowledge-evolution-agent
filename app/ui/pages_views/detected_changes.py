"""Detected Changes page: source evolution (previous -> latest version), ranked, with a Filtered tab."""

import html
from typing import Optional

import streamlit as st
from sqlalchemy.orm import Session, sessionmaker

from app.services.config_service import AppConfig
from app.services.reviewer_authorization import ReviewerAuthorizationService
from app.ui.components import render_page_header
from app.ui.queries import get_detected_changes, restore_filtered_change

CATEGORY_BADGE = {
    "contraindication_added": "badge-red", "safety_warning": "badge-red", "withdrawn": "badge-amber",
    "dose_change": "badge-amber", "threshold_change": "badge-amber", "new_recommendation": "badge-blue",
    "revised_recommendation": "badge-blue", "contraindication_removed": "badge-neutral",
    "new_evidence": "badge-neutral", "no_practice_change": "badge-neutral",
}


def _render_change(row: dict) -> None:
    badge = CATEGORY_BADGE.get(row["category"], "badge-neutral")
    label = row["category"].replace("_", " ").upper()
    st.markdown(
        f"<span class='badge-pill {badge}'>{html.escape(label)}</span> "
        f"<span class='badge-pill badge-neutral'>priority {row['priority']}</span> "
        f"<strong>{html.escape(row['source'])}</strong> "
        f"<span style='color:#64748b'>({html.escape(', '.join(row['treatments']) or 'no treatment')})</span>",
        unsafe_allow_html=True,
    )
    c_old, c_new = st.columns(2)
    with c_old:
        if row["previous_text"]:
            st.caption(f"Previous: version {row['previous_version']} ({row['previous_date']})")
            st.markdown(f"> {html.escape(row['previous_text'])}")
        else:
            st.caption("Previous: not present (new content)")
    with c_new:
        if row["latest_text"]:
            st.caption(f"Latest: version {row['latest_version']} ({row['latest_date']})")
            st.markdown(f"> {html.escape(row['latest_text'])}")
        else:
            st.caption("Latest: removed")
    for d in row["attribute_changes"]:
        st.caption(f"{d['kind']}: {d['before']} -> {d['after']}")
    basis = row["ranking_basis"]
    st.caption(
        f"Urgency {row['urgency']} ({basis.get('urgency', {}).get('rule_id')}) · relevance {row['relevance']} · "
        f"source quality {row['source_quality']} · novelty {row['novelty']}"
        + (" (duplicate of a higher-quality source)" if row["duplicate_of"] else "")
        + f" · departments {', '.join(row['departments'])}"
    )
    st.divider()


def render_detected_changes_view(session_factory: sessionmaker[Session], config: AppConfig,
                                 department: Optional[str] = None) -> None:
    render_page_header(
        breadcrumb="SURVEILLANCE › DETECTED CHANGES",
        title="Detected Changes (source evolution)",
        description=("What changed in each monitored source between its previous and latest version, ranked by "
                     "urgency, relevance, source quality and novelty. Comparison with hospital protocols is separate."),
    )
    active = get_detected_changes(session_factory, department=department)
    filtered = get_detected_changes(session_factory, department=department, filtered=True)
    tab_active, tab_filtered = st.tabs([f"Ranked changes ({len(active)})", f"Filtered ({len(filtered)})"])
    with tab_active:
        if not active:
            st.info("No changes yet. Run a watchlist check from 'Watchlist & Sources'.")
        for row in active:
            _render_change(row)
    with tab_filtered:
        st.caption("Filtered items are kept and can be restored by an authorised reviewer (audited).")
        reviewers = [r.reviewer_id for r in ReviewerAuthorizationService(registry_path=config.reviewer_registry_path).list_reviewers()]
        for row in filtered:
            _render_change(row)
            st.caption(f"Filter reason: {row['filter_reason']}")
            with st.form(f"restore_{row['id']}"):
                c1, c2, c3 = st.columns([1, 2, 1])
                reviewer = c1.selectbox("Reviewer (demo identity)", reviewers, key=f"rv_{row['id']}")
                reason = c2.text_input("Reason", key=f"rs_{row['id']}")
                if c3.form_submit_button("Restore"):
                    try:
                        restore_filtered_change(session_factory, config, row["id"], reviewer, reason)
                        st.success("Restored to the ranked queue.")
                        st.rerun()
                    except (PermissionError, ValueError) as e:
                        st.error(str(e))
