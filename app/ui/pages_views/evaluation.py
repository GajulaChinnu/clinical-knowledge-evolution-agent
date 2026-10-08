"""Evaluation page."""

import streamlit as st
from app.ui.queries import load_evaluation_reports
from app.ui.components import render_page_header


def render_evaluation_reports_view() -> None:
    """Render Phase 12 Evaluation Framework Reports."""
    render_page_header(
        breadcrumb="VERIFICATION › EVALUATION",
        title="Evaluation Framework Reports",
        description="Read-only inspection of reproducible evaluation results across the 20-document synthetic corpus."
    )

    reports = load_evaluation_reports()
    report_names = [k for k in reports.keys() if reports[k] is not None]

    if not report_names:
        st.info("No evaluation reports found in data/evaluation/reports/.")
        return

    selected_report = st.selectbox("Select Evaluation Report:", options=report_names)

    if selected_report:
        rep_data = reports[selected_report]
        st.json(rep_data)
