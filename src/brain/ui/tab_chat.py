"""Chat tab — ask questions, see answers with citations."""

from __future__ import annotations

import asyncio

import streamlit as st


def _get_stores():
    """Lazy-init stores in session state."""
    if "docstore" not in st.session_state:
        from brain.config import get_settings
        from brain.stores.docstore import DocStore
        from brain.stores.vectorstore import VectorStore

        settings = get_settings()
        st.session_state.docstore = DocStore(settings.docstore.path)
        st.session_state.vectorstore = VectorStore()
    return st.session_state.docstore, st.session_state.vectorstore


def render():
    st.subheader("Chat with your knowledge base")

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "citations" in msg:
                with st.expander("Sources"):
                    for i, c in enumerate(msg["citations"], 1):
                        source = c.get("title") or c.get("source_path", "")
                        st.caption(f"[D{i}] {source} — score: {c.get('score', 0):.3f}")
                        st.text(c.get("snippet", "")[:200])
            if "graph_facts" in msg and msg["graph_facts"]:
                with st.expander("Graph Facts"):
                    for i, gf in enumerate(msg["graph_facts"], 1):
                        st.caption(
                            f"[G{i}] {gf['source_node']} → {gf['target_node']}: {gf['fact']}"
                        )

    # Chat input
    from brain.config import get_settings
    settings = get_settings()

    use_graph = st.toggle("Enable graph-enhanced retrieval", value=False, key="chat_graph_toggle")
    top_k = st.slider("Retrieved chunks", 1, 20, settings.chat.top_k, key="chat_top_k")

    if prompt := st.chat_input("Ask your second brain..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                docstore, vectorstore = _get_stores()

                async def _run():
                    from brain.chat.engine import chat

                    graph_client = None
                    if use_graph:
                        from brain.graph.client import GraphClient
                        graph_client = GraphClient()
                        await graph_client.initialize()

                    answer, citations, graph_facts = await chat(
                        prompt,
                        vectorstore,
                        docstore,
                        graph_client=graph_client,
                        top_k=top_k,
                    )

                    if graph_client:
                        await graph_client.close()
                    return answer, citations, graph_facts

                answer, citations, graph_facts = asyncio.run(_run())

            st.markdown(answer)

            citation_dicts = [
                {
                    "title": c.title,
                    "source_path": c.source_path,
                    "snippet": c.snippet,
                    "score": c.score,
                }
                for c in citations
            ]
            if citation_dicts:
                with st.expander("Sources"):
                    for i, c in enumerate(citation_dicts, 1):
                        source = c.get("title") or c.get("source_path", "")
                        st.caption(f"[D{i}] {source} — score: {c.get('score', 0):.3f}")
                        st.text(c.get("snippet", "")[:200])

            if graph_facts:
                with st.expander("Graph Facts"):
                    for i, gf in enumerate(graph_facts, 1):
                        st.caption(
                            f"[G{i}] {gf['source_node']} → {gf['target_node']}: {gf['fact']}"
                        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "citations": citation_dicts,
            "graph_facts": graph_facts,
        })
