"""Audit page."""

import streamlit as st
from sqlalchemy.orm import (
    Session,
    sessionmaker,
)
from app.ui.queries import get_audit_trail
from app.ui.components import render_page_header


def render_audit_view(session_factory: sessionmaker[Session]) -> None:
    """Render Immutable Governance Audit Trail as an Enterprise Data Grid."""
    render_page_header(
        breadcrumb="COMPLIANCE › AUDIT LOG",
        title="Immutable Governance Audit Trail",
        description="Cryptographically auditable and append-only record of all system events and human decisions."
    )

    logs = get_audit_trail(session_factory)
    if not logs:
        st.info("No audit events recorded.")
        return

    df = []
    for log in logs:
        df.append({
            "Timestamp": str(log["timestamp"])[:19],
            "Actor": log["actor"],
            "Action": log["action"].upper(),
            "Entity Type": log["entity_type"].title(),
            "Entity ID": (log["entity_id"] or "")[:12] + "...",
            "Transition": f"{log['previous_status'] or 'None'} ➔ {log['new_status']}",
            "Reason / Rationale": log["reason"] or "N/A",
        })

    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "Timestamp": st.column_config.TextColumn("Timestamp", width="medium"),
            "Actor": st.column_config.TextColumn("Actor", width="small"),
            "Action": st.column_config.TextColumn("Action", width="small"),
            "Entity Type": st.column_config.TextColumn("Entity Type", width="small"),
            "Entity ID": st.column_config.TextColumn("Entity ID"),
            "Transition": st.column_config.TextColumn("State Transition", width="medium"),
            "Reason / Rationale": st.column_config.TextColumn("Rationale / Details", width="large"),
        },
    )
