"""Sources page."""

import logging
import html
from typing import Optional
import streamlit as st
from sqlalchemy.orm import (
    Session,
    sessionmaker,
)
from app.services.config_service import (
    AppConfig,
    load_config,
)
from app.services.source_documents import UnreadableSourceError
from app.services.source_ingestion_service import UnknownSourceError
from app.services.url_ingestion_service import URLIngestionError
from app.ui.queries import (
    get_source_documents,
    handle_source_pdf_upload,
    handle_source_url_upload,
)
from app.ui.components import (
    format_version,
    render_page_header,
    render_upload_result_card,
)

logger = logging.getLogger("ckea.ui.sources")


def render_sources_view(
    session_factory: sessionmaker[Session],
    config: Optional[AppConfig] = None,
) -> None:
    """Render Document and Monitoring inspection view with inventory prioritization."""
    render_page_header(
        breadcrumb="SURVEILLANCE › SOURCES",
        title="Clinical Source Documents",
        description="Continuous surveillance of external clinical guidelines, literature URLs, and protocol evidence."
    )
    cfg = config or load_config()

    docs = get_source_documents(session_factory)

    # ------------------------------------------------------------------
    # COMPACT TOOLBAR & CONTROLS
    # ------------------------------------------------------------------
    col_tool1, col_tool2, col_tool3 = st.columns([2, 3, 2])
    with col_tool1:
        st.markdown(
            f"""
            <div style="padding-top: 6px;">
                <span class="badge-pill badge-neutral">Total Sources: <strong>{len(docs)}</strong></span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_tool2:
        search_kw = st.text_input(
            "Filter sources",
            placeholder="Search by identifier or source...",
            label_visibility="collapsed",
            key="source_search_input",
        )
    with col_tool3:
        show_add = st.checkbox("Add clinical source", value=False, key="toggle_add_source")

    # ------------------------------------------------------------------
    # ADD CLINICAL SOURCE EXPANDABLE CARD
    # ------------------------------------------------------------------
    if show_add:
        with st.container():
            st.markdown(
                """
                <div class="ent-card" style="margin-top: 8px;">
                    <div class="ent-metric-title" style="margin-bottom: 8px;">Add Clinical Source Document</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            tab_url, tab_pdf = st.tabs(["From URL", "Upload PDF"])

            with tab_url:
                st.caption(
                    "Any HTTP(S) URL. PDFs are downloaded directly; web pages are retrieved through "
                    "Jina Reader. Sites requiring login or bot checks are rejected, never bypassed."
                )
                col_u1, col_u2 = st.columns([4, 1])
                with col_u1:
                    url_input = st.text_input(
                        "Clinical Source Document URL",
                        placeholder="https://example.org/guidelines/diabetes_protocol.pdf or web URL",
                        label_visibility="collapsed",
                        key="source_url_input",
                    )
                with col_u2:
                    fetch_clicked = st.button("Fetch Source", type="primary", use_container_width=True, key="fetch_url_btn")

                if fetch_clicked:
                    if not url_input or not url_input.strip():
                        st.warning("Please provide a valid HTTP or HTTPS clinical source URL.")
                    else:
                        with st.spinner("Fetching URL and processing through pipeline..."):
                            try:
                                res = handle_source_url_upload(
                                    url=url_input.strip(),
                                    session_factory=session_factory,
                                    config=cfg,
                                )
                                st.session_state["upload_result_info"] = res
                            except (URLIngestionError, ValueError) as ue:
                                st.error(f"URL Ingestion Error: {ue}")
                            except Exception as e:
                                logger.exception("Unexpected error in URL ingestion handler: %s", e)
                                st.error("URL retrieval succeeded, but processing failed. Please check system logs.")

            with tab_pdf:
                col_p1, col_p2 = st.columns([3, 1])
                with col_p1:
                    uploaded_file = st.file_uploader(
                        "Choose PDF",
                        type=["pdf"],
                        accept_multiple_files=False,
                        label_visibility="collapsed",
                        key="source_pdf_file_uploader",
                    )
                with col_p2:
                    upload_clicked = st.button("Process Document", type="primary", use_container_width=True, key="upload_process_btn")

                # Source identity is never the filename: the reviewer states whether this PDF
                # starts a new source or is a new version of an existing one.
                existing_sources = {
                    d["source_identifier"]: d for d in get_source_documents(session_factory)
                }
                NEW_SOURCE = "New source"
                target_source = st.selectbox(
                    "Source",
                    options=[NEW_SOURCE] + list(existing_sources),
                    format_func=lambda sid: sid if sid == NEW_SOURCE else (
                        f"New version of: {existing_sources[sid].get('title') or sid} "
                        f"({format_version(existing_sources[sid]['source_version'])})"
                    ),
                    key="upload_target_source",
                )

                if upload_clicked:
                    if uploaded_file is None:
                        st.warning("Please select a PDF file to upload.")
                    else:
                        with st.spinner("Processing PDF through clinical pipeline..."):
                            try:
                                res = handle_source_pdf_upload(
                                    uploaded_name=uploaded_file.name,
                                    uploaded_bytes=uploaded_file.getvalue(),
                                    session_factory=session_factory,
                                    config=cfg,
                                    existing_source_identifier=None if target_source == NEW_SOURCE else target_source,
                                )
                                st.session_state["upload_result_info"] = res
                            except UnreadableSourceError as ue:
                                st.error(f"Unreadable PDF (recorded as an ingestion failure): {ue}")
                            except UnknownSourceError as ue:
                                st.error(str(ue))
                            except ValueError as ve:
                                st.error(f"Validation Error: {ve}")
                            except Exception as e:
                                logger.exception("Unexpected error in upload handler: %s", e)
                                st.error("Upload succeeded, but processing failed. Please check system logs.")

        if "upload_result_info" in st.session_state:
            render_upload_result_card(st.session_state["upload_result_info"])

    # ------------------------------------------------------------------
    # PRIMARY: SOURCE INVENTORY TABLE (Above Fold)
    # ------------------------------------------------------------------
    st.markdown("### Source Inventory")
    if not docs:
        st.info("No clinical sources currently in the database. Use '➕ Add Clinical Source' above to ingest a guideline.")
        return

    # Filter docs by search keyword if provided
    filtered_docs = docs
    if search_kw and search_kw.strip():
        kw = search_kw.strip().lower()
        filtered_docs = [
            d for d in docs
            if kw in d["source_identifier"].lower() or kw in str(d["source_path"]).lower() or kw in str(d["display_name"]).lower()
        ]

    df_rows = []
    for d in filtered_docs:
        meta_type = "PDF" if str(d.get("source_path", "")).lower().endswith(".pdf") else "URL"
        chg_text = d["change_status"].replace("_", " ").title()
        df_rows.append({
            "Source": d["display_name"],
            "Type": meta_type,
            "Version": f"v{d['source_version']}",
            "Change": chg_text,
            "Pipeline": d["status"].title(),
            "Gate": "G1" if d["status"] == "held" else "None",
            "Last Retrieved": str(d["last_retrieved"])[:16],
            "id": d["id"],
        })

    st.dataframe(
        df_rows,
        use_container_width=True,
        column_config={
            "Source": st.column_config.TextColumn("Source", width="large"),
            "Type": st.column_config.TextColumn("Type", width="small"),
            "Version": st.column_config.TextColumn("Version", width="small"),
            "Change": st.column_config.TextColumn("Change Status", width="medium"),
            "Pipeline": st.column_config.TextColumn("Pipeline", width="small"),
            "Gate": st.column_config.TextColumn("Gate", width="small"),
            "Last Retrieved": st.column_config.TextColumn("Last Retrieved", width="medium"),
            "id": None,  # Hide internal primary key from table
        },
    )

    # ------------------------------------------------------------------
    # SECONDARY: DETAIL VIEW & PROVENANCE
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Source Detail & Provenance")

    doc_ids = [d["id"] for d in filtered_docs]
    selected_id = st.selectbox(
        "Select source to inspect provenance:",
        options=doc_ids,
        format_func=lambda x: next(f"{d.get('title') or d['source_identifier']} ({format_version(d['source_version'])})" for d in filtered_docs if d["id"] == x),
        key="source_detail_selector",
    )

    if selected_id:
        doc_item = next(d for d in filtered_docs if d["id"] == selected_id)

        # Level 1 Hierarchy: PRIMARY (Source title, Version, Change status, Pipeline status, Gate)
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        with col_p1:
            st.metric("Source", doc_item["display_name"][:40], help=f"Stable identity: {doc_item['source_identifier']}")
        with col_p2:
            st.metric("Current Version", f"v{doc_item['source_version']}")
        with col_p3:
            st.metric("Change Status", doc_item["change_status"].replace("_", " ").title())
        with col_p4:
            st.metric("Pipeline Gate", "G1 (Held)" if doc_item["status"] == "held" else doc_item["status"].title())

        # Lineage Timeline
        prev_ver_id = doc_item.get("previous_source_version_id")
        lineage_label = (
            f"{format_version(doc_item.get('previous_source_version'))} → {format_version(doc_item['source_version'])}"
            if prev_ver_id else f"{format_version(doc_item['source_version'])} (first version seen)"
        )
        st.markdown(
            f"""
            <div class="lineage-flow-box">
                <strong>Source Lineage:</strong> {html.escape(lineage_label)}
                <span class="badge-pill badge-neutral" style="margin-left: auto;">Status: {html.escape(doc_item['change_status'].replace('_', ' ').title())}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Level 2 Hierarchy: SECONDARY (Collapsible Technical Provenance)
        with st.expander("Technical Provenance & Integrity", expanded=False):
            t1, t2 = st.columns(2)
            with t1:
                st.write(f"**Document ID:** `{doc_item['id']}`")
                st.write(f"**Storage Path:** `{doc_item['source_path']}`")
                st.write(f"**First Seen:** `{str(doc_item['first_seen'])[:19]}`")
                st.write(f"**Last Retrieved:** `{str(doc_item['last_retrieved'])[:19]}`")
                st.write(f"**Retrieval Method:** `{doc_item['retrieval_provider']}`")
                if doc_item.get("routing_decision"):
                    st.write(f"**Routing Decision:** `{doc_item['routing_decision']}`")
                if doc_item.get("resolved_source_url"):
                    st.write(f"**Resolved URL:** `{doc_item['resolved_source_url']}`")
                for warning in doc_item.get("provider_warnings") or []:
                    st.warning(f"Retrieval provider warning: {warning}")

        source_diff = doc_item.get("source_diff") or {}
        if source_diff.get("changes"):
            with st.expander("Source Evolution (previous version → this version)", expanded=False):
                st.caption(
                    "What changed in the external source itself. This is separate from the "
                    "institutional protocol comparison."
                )
                for change in source_diff["changes"]:
                    kind = change["change_type"].upper()
                    heading = change["heading"]
                    if change.get("old_heading"):
                        heading = f"{change['old_heading']} → {heading}"
                    st.markdown(f"**{kind}** · {html.escape(heading)}")
                    if change["change_type"] == "removed":
                        st.warning("Removed from the source; not re-extracted. Review whether the protocol relied on it.")
                        st.text(change["old_text"][:1500])
                    elif change["change_type"] == "modified":
                        c_old, c_new = st.columns(2)
                        c_old.text(change["old_text"][:1500])
                        c_new.text(change["new_text"][:1500])
                    else:
                        st.text(change["new_text"][:1500])
            with t2:
                st.write(f"**Current SHA-256:** `{doc_item['sha256_hash']}`")
                prev_sha = doc_item.get("previous_sha256_hash")
                st.write(f"**Previous SHA-256:** `{prev_sha if prev_sha else 'None (First Ingestion)'}`")
                st.write(f"**Extracted Change Records:** `{doc_item['change_records_count']}`")
