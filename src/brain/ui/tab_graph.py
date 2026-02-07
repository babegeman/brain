"""Graph tab — search the knowledge graph, view stats, clear data."""

from __future__ import annotations

import asyncio

import streamlit as st

CONFIRM_PHRASE = "delete all graph data"


def render():
    st.subheader("Knowledge Graph")

    from brain.config import get_settings
    settings = get_settings()

    st.markdown(
        f"**Neo4j**: `{settings.neo4j.uri}` — "
        f"Open [Neo4j Browser](http://localhost:7474) to explore the full graph."
    )

    # Graph stats
    _show_stats()

    st.divider()

    # Graph search
    query = st.text_input("Search the graph for entities/facts:", key="graph_query")
    num_results = st.slider("Number of results", 1, 30, settings.graph.num_results, key="graph_num_results")

    if query and st.button("Search Graph", key="graph_search_btn"):
        _search_graph(query, num_results)

    st.divider()

    # Danger zone — clear graph
    _render_clear_section()


def _show_stats():
    from brain.graph.client import GraphClient

    async def _run():
        gc = GraphClient()
        return await gc.get_stats()

    try:
        stats = asyncio.run(_run())
        col1, col2 = st.columns(2)
        col1.metric("Nodes", stats["nodes"])
        col2.metric("Relationships", stats["relationships"])
    except Exception as e:
        st.warning(f"Could not fetch graph stats: {e}")


def _search_graph(query: str, num_results: int):
    from brain.graph.client import GraphClient

    async def _run():
        gc = GraphClient()
        await gc.initialize()
        facts = await gc.search(query, num_results=num_results)
        await gc.close()
        return facts

    with st.spinner("Searching knowledge graph..."):
        try:
            facts = asyncio.run(_run())
        except Exception as e:
            st.error(f"Graph search failed: {e}")
            return

    if not facts:
        st.info("No graph facts found for this query.")
        return

    st.markdown(f"**{len(facts)} facts found:**")
    for i, f in enumerate(facts, 1):
        st.markdown(
            f"**{i}.** {f['source_node']} → {f['target_node']}: {f['fact']}"
        )


def _render_clear_section():
    with st.expander("Danger Zone", expanded=False):
        st.warning("This will permanently delete **all** nodes and relationships from the knowledge graph.")
        phrase = st.text_input(
            f'Type `{CONFIRM_PHRASE}` to enable the delete button:',
            key="graph_clear_phrase",
        )

        disabled = phrase != CONFIRM_PHRASE
        if st.button("Clear Entire Graph", key="graph_clear_btn", type="primary", disabled=disabled):
            _clear_graph()


def _clear_graph():
    from brain.graph.client import GraphClient

    async def _run():
        gc = GraphClient()
        return await gc.clear_graph()

    with st.spinner("Clearing graph..."):
        try:
            result = asyncio.run(_run())
        except Exception as e:
            st.error(f"Failed to clear graph: {e}")
            return

    st.success(
        f"Graph cleared: {result['deleted_nodes']} nodes, "
        f"{result['deleted_relationships']} relationships deleted."
    )
