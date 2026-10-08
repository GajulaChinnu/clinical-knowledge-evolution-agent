"""Watchlist page: the defined set of monitored sources and a manual surveillance run."""

import streamlit as st
from sqlalchemy.orm import Session, sessionmaker

from app.services.config_service import AppConfig
from app.ui.components import render_page_header
from app.ui.queries import get_watchlist_rows, run_watchlist_surveillance


def render_watchlist_view(session_factory: sessionmaker[Session], config: AppConfig) -> None:
    render_page_header(
        breadcrumb="SURVEILLANCE › WATCHLIST",
        title="Monitored Source Watchlist",
        description=("Guidelines, safety notices and publications CKEA monitors (config/watchlist.yaml). "
                     "A check ingests any newer version, indexes its statements and ranks what changed."),
    )
    rows = get_watchlist_rows(session_factory)
    c1, c2 = st.columns([3, 1])
    c1.caption(f"{len(rows)} sources. All corpus content is SYNTHETIC.")
    if c2.button("Check all sources now", type="primary", use_container_width=True):
        with st.spinner("Checking watchlist sources..."):
            results = run_watchlist_surveillance(session_factory, config)
        st.session_state["watchlist_results"] = results
        st.rerun()

    st.dataframe([{k: v for k, v in r.items() if k != "id"} for r in rows], use_container_width=True, hide_index=True)

    results = st.session_state.get("watchlist_results")
    if results:
        st.markdown("### Last check")
        st.dataframe(results, use_container_width=True, hide_index=True)
        failed = [r for r in results if r["Result"] != "OK"]
        if failed:
            st.error(f"{len(failed)} source(s) failed; see Result column and the ingestion failure log.")
