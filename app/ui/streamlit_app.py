"""Clinical Knowledge Evolution Agent (CKEA) - Streamlit Governance & Demonstration UI.

Phase 13: Final End-to-End Application Integration & Human Governance Interface.

CORE SAFETY PRINCIPLE:
"System prepares evidence and review material. Authorized clinicians make the final decision."
The application NEVER makes automated clinical decisions, approves, rejects, defers, or closes
briefs without explicit human action.

Entry point only: layout and navigation. Data access lives in app.ui.queries, shared
widgets in app.ui.components, page renderers in app.ui.pages_views, styles in app.ui.theme.
"""

import sys
from pathlib import Path

# Allow `streamlit run app/ui/streamlit_app.py` from the project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402

from app.agents.governance_agent import GovernanceAgent  # noqa: E402
from app.services.config_service import load_config  # noqa: E402
from app.ui.components import *  # noqa: E402,F401,F403  (re-exported for callers/tests)
from app.ui.components import render_gate_legend, render_header  # noqa: E402
from app.ui.queries import *  # noqa: E402,F401,F403  (re-exported for callers/tests)
from app.ui.queries import get_db_session_factory  # noqa: E402
from app.ui.pages_views.dashboard import render_dashboard  # noqa: E402
from app.ui.pages_views.sources import render_sources_view  # noqa: E402
from app.ui.pages_views.changes import render_changes_view  # noqa: E402
from app.ui.pages_views.impact import render_impact_view  # noqa: E402
from app.ui.pages_views.briefs import render_briefs_view  # noqa: E402
from app.ui.pages_views.governance import render_governance_view  # noqa: E402
from app.ui.pages_views.audit import render_audit_view  # noqa: E402
from app.ui.pages_views.evaluation import render_evaluation_reports_view  # noqa: E402
from app.ui.theme import inject_theme  # noqa: E402

__all__ = ['execute_governance_action', 'format_version', 'get_audit_trail', 'get_brief_full_payload', 'get_brief_summaries', 'get_changes_with_gaps', 'get_dashboard_metrics', 'get_db_session_factory', 'get_impact_records', 'get_source_documents', 'handle_source_pdf_upload', 'handle_source_url_upload', 'load_evaluation_reports', 'render_audit_view', 'render_badge', 'render_brief_sections', 'render_briefs_view', 'render_changes_view', 'render_dashboard', 'render_evaluation_reports_view', 'render_gate_legend', 'render_governance_view', 'render_header', 'render_impact_view', 'render_page_header', 'render_processing_chain', 'render_sources_view', 'render_upload_result_card', 'resolve_acting_reviewer', 'resolve_protocol_section_details', 'summarize_source_diff']


def main() -> None:
    """Streamlit application entry point."""
    st.set_page_config(
        page_title="CKEA - Clinical Knowledge Evolution Agent",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject modern enterprise CSS
    inject_theme()

    # Render compact global clinical governance boundary notice
    render_header()

    config = load_config()
    session_factory = get_db_session_factory(config)
    governance_agent = GovernanceAgent(session_factory=session_factory, config=config)

    # Compact Enterprise Product Sidebar Header
    st.sidebar.markdown(
        """
        <div style="padding: 0.2rem 0 0.6rem 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 0.6rem;">
            <div style="font-size: 0.95rem; font-weight: 700; color: #0f172a; letter-spacing: -0.01em;">CKEA Governance</div>
            <div style="font-size: 0.72rem; color: #64748b; font-weight: 500;">Clinical Knowledge Evolution</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pages = [
        "Dashboard",
        "Sources",
        "Clinical Changes",
        "Impact",
        "Change Briefs",
        "Governance",
        "Audit Log",
        "Evaluation",
    ]
    # ?page=<slug> deep links (e.g. ?page=governance) so reviewers can share a view.
    slugs = {p.lower().replace(" ", "-"): p for p in pages}
    requested = slugs.get(str(st.query_params.get("page", "")).lower(), "Dashboard")
    menu = st.sidebar.radio(
        "Navigation",
        options=pages,
        index=pages.index(requested),
        label_visibility="collapsed",
    )
    st.query_params["page"] = menu.lower().replace(" ", "-")

    # Compact safety gate legend at bottom of sidebar
    render_gate_legend()

    if menu == "Dashboard":
        render_dashboard(session_factory)
    elif menu == "Sources":
        render_sources_view(session_factory, config=config)
    elif menu == "Clinical Changes":
        render_changes_view(session_factory)
    elif menu == "Impact":
        render_impact_view(session_factory)
    elif menu == "Change Briefs":
        render_briefs_view(session_factory)
    elif menu == "Governance":
        render_governance_view(session_factory, governance_agent)
    elif menu == "Audit Log":
        render_audit_view(session_factory)
    elif menu == "Evaluation":
        render_evaluation_reports_view()


if __name__ == "__main__":
    main()
