"""Jobs / Logs tab — view ingested document stats."""

from __future__ import annotations

import streamlit as st


def render():
    st.subheader("Jobs & Logs")

    from brain.config import get_settings
    from brain.stores.docstore import DocStore

    settings = get_settings()
    docstore = DocStore(settings.docstore.path)

    docs = docstore.list_documents()
    total_chunks = sum(len(docstore.get_chunks_for_doc(d.doc_id)) for d in docs)

    col1, col2, col3 = st.columns(3)
    col1.metric("Documents", len(docs))
    col2.metric("Total Chunks", total_chunks)

    try:
        from brain.stores.vectorstore import VectorStore
        vs = VectorStore()
        col3.metric("Vectors", vs.count())
        vs.close()
    except Exception:
        col3.metric("Vectors", "N/A")

    docstore.close()

    st.divider()
    st.markdown("**Recent ingestions:**")

    if not docs:
        st.info("No documents ingested yet.")
        return

    rows = []
    for d in docs[:20]:
        chunk_count = len(docstore.get_chunks_for_doc(d.doc_id)) if False else "—"
        rows.append({
            "Title": d.title,
            "Type": d.doc_type.value,
            "Ingested": str(d.ingested_at)[:19],
            "Hash": d.content_hash[:12] + "..." if d.content_hash else "",
        })

    st.dataframe(rows, use_container_width=True)
