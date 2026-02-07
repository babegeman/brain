"""Settings tab — view and edit configuration, check health."""

from __future__ import annotations

import asyncio

import streamlit as st


def render():
    st.subheader("Settings")

    from brain.config import get_settings

    settings = get_settings()

    (
        tab_llm,
        tab_retrieval,
        tab_ingestion,
        tab_graph,
        tab_prompts,
        tab_connections,
        tab_health,
    ) = st.tabs([
        "LLM Profiles",
        "Retrieval",
        "Ingestion",
        "Graph",
        "Prompts",
        "Connections",
        "Health Check",
    ])

    with tab_llm:
        _render_llm_section(settings)

    with tab_retrieval:
        _render_retrieval_section(settings)

    with tab_ingestion:
        _render_ingestion_section(settings)

    with tab_graph:
        _render_graph_section(settings)

    with tab_prompts:
        _render_prompts_section(settings)

    with tab_connections:
        _render_connections_section(settings)

    with tab_health:
        _render_health_section()


# ---------------------------------------------------------------------------
# Sub-tab renderers
# ---------------------------------------------------------------------------


def _render_llm_section(settings):
    st.markdown("#### Active Profile")

    profile_names = list(settings.profiles.keys())
    current_idx = (
        profile_names.index(settings.active_profile)
        if settings.active_profile in profile_names
        else 0
    )

    selected = st.selectbox("Profile", profile_names, index=current_idx, key="cfg_active_profile")
    profile = settings.profiles.get(selected, settings.llm)

    st.markdown("#### Model Settings")
    col1, col2 = st.columns(2)

    with col1:
        chat_model = st.text_input("Chat model", value=profile.chat_model, key="cfg_chat_model")
        embed_model = st.text_input("Embed model", value=profile.embed_model, key="cfg_embed_model")
        embed_dim = st.number_input(
            "Embed dimension", value=profile.embed_dim, min_value=1, key="cfg_embed_dim"
        )
        vision_model = st.text_input(
            "Vision model", value=profile.vision_model, key="cfg_vision_model"
        )

    with col2:
        graph_extract_model = st.text_input(
            "Graph extract model", value=profile.graph_extract_model, key="cfg_graph_extract_model"
        )
        graph_extract_small_model = st.text_input(
            "Graph extract (small)",
            value=profile.graph_extract_small_model,
            key="cfg_graph_extract_small_model",
        )
        temperature = st.slider(
            "Temperature", 0.0, 2.0, profile.temperature, step=0.05, key="cfg_temperature"
        )
        max_tokens = st.number_input(
            "Max tokens", value=profile.max_tokens, min_value=1, max_value=32768, key="cfg_max_tokens"
        )

    if st.button("Save LLM Settings", key="save_llm"):
        from brain.config import save_user_config

        save_user_config({
            "active_profile": selected,
            "profiles": {
                selected: {
                    "chat_model": chat_model,
                    "embed_model": embed_model,
                    "embed_dim": embed_dim,
                    "vision_model": vision_model,
                    "graph_extract_model": graph_extract_model,
                    "graph_extract_small_model": graph_extract_small_model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            },
        })
        st.success(f"Saved LLM settings for profile '{selected}'.")
        st.rerun()


def _render_retrieval_section(settings):
    st.markdown("#### Chat Retrieval Defaults")
    st.caption("These are defaults. The Chat tab sliders override them per-session.")

    top_k = st.slider("Default top_k (chunks)", 1, 30, settings.chat.top_k, key="cfg_top_k")
    graph_results = st.slider(
        "Default graph results", 1, 30, settings.chat.graph_results, key="cfg_graph_results"
    )
    snippet_length = st.number_input(
        "Snippet length (chars)",
        value=settings.chat.snippet_length,
        min_value=50,
        max_value=2000,
        key="cfg_snippet_length",
    )

    if st.button("Save Retrieval Settings", key="save_retrieval"):
        from brain.config import save_user_config

        save_user_config({
            "chat": {
                "top_k": top_k,
                "graph_results": graph_results,
                "snippet_length": snippet_length,
            }
        })
        st.success("Saved retrieval settings.")
        st.rerun()


def _render_ingestion_section(settings):
    st.markdown("#### Chunker Settings")

    chunk_size = st.number_input(
        "Chunk size (chars)",
        value=settings.chunker.chunk_size,
        min_value=64,
        max_value=8192,
        step=64,
        key="cfg_chunk_size",
    )
    chunk_overlap = st.number_input(
        "Chunk overlap (chars)",
        value=settings.chunker.chunk_overlap,
        min_value=0,
        max_value=chunk_size // 2,
        step=16,
        key="cfg_chunk_overlap",
    )

    st.markdown("#### Watched Folders")
    folders_text = st.text_area(
        "One folder per line",
        value="\n".join(settings.watched_folders),
        key="cfg_watched_folders",
    )

    if st.button("Save Ingestion Settings", key="save_ingestion"):
        from brain.config import save_user_config

        folders = [f.strip() for f in folders_text.strip().split("\n") if f.strip()]
        save_user_config({
            "chunker": {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            },
            "watched_folders": folders,
        })
        st.success("Saved ingestion settings.")
        st.rerun()


def _render_graph_section(settings):
    st.markdown("#### Knowledge Graph Defaults")

    group_id = st.text_input(
        "Graph group ID", value=settings.graph.group_id, key="cfg_group_id"
    )
    num_results = st.slider(
        "Default graph search results", 1, 50, settings.graph.num_results, key="cfg_graph_num_results"
    )

    if st.button("Save Graph Settings", key="save_graph"):
        from brain.config import save_user_config

        save_user_config({
            "graph": {
                "group_id": group_id,
                "num_results": num_results,
            }
        })
        st.success("Saved graph settings.")
        st.rerun()


def _render_prompts_section(settings):
    st.markdown("#### System Prompts")
    st.caption("These are sent as the system message to the LLM for each chat query.")

    system_prompt = st.text_area(
        "Document-only system prompt",
        value=settings.prompts.system_prompt,
        height=200,
        key="cfg_system_prompt",
    )
    system_prompt_fusion = st.text_area(
        "Fusion (doc + graph) system prompt",
        value=settings.prompts.system_prompt_fusion,
        height=200,
        key="cfg_system_prompt_fusion",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save Prompts", key="save_prompts"):
            from brain.config import save_user_config

            save_user_config({
                "prompts": {
                    "system_prompt": system_prompt,
                    "system_prompt_fusion": system_prompt_fusion,
                }
            })
            st.success("Saved prompt settings.")
            st.rerun()
    with col2:
        if st.button("Reset to Defaults", key="reset_prompts"):
            from brain.config import PromptsConfig, save_user_config

            defaults = PromptsConfig()
            save_user_config({
                "prompts": {
                    "system_prompt": defaults.system_prompt,
                    "system_prompt_fusion": defaults.system_prompt_fusion,
                }
            })
            st.success("Prompts reset to defaults.")
            st.rerun()


def _render_connections_section(settings):
    st.markdown("#### Connection Details")
    st.info("Connection settings are read-only here. Edit config.yaml directly to change them.")
    st.code(f"Neo4j URI:     {settings.neo4j.uri}")
    st.code(f"Neo4j User:    {settings.neo4j.user}")
    st.code(f"Qdrant Path:   {settings.qdrant.path}")
    st.code(f"Qdrant Coll:   {settings.qdrant.collection}")
    st.code(f"Docstore Path: {settings.docstore.path}")


def _render_health_section():
    if st.button("Run Health Check", key="settings_doctor"):
        from brain.cli.doctor import _run_checks

        with st.spinner("Running health checks..."):
            results = asyncio.run(_run_checks())

        for r in results:
            icon = "PASS" if r["ok"] else "FAIL"
            st.markdown(f"**[{icon}] {r['check']}**: {r['detail']}")
