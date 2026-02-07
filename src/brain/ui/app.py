"""Streamlit entry point — Second Brain GUI."""

import streamlit as st

st.set_page_config(
    page_title="Second Brain",
    page_icon="🧠",
    layout="wide",
)

st.title("Second Brain")

tab_chat, tab_ingest, tab_graph, tab_jobs, tab_settings = st.tabs(
    ["Chat", "Ingest", "Graph", "Jobs / Logs", "Settings"]
)

with tab_chat:
    from brain.ui.tab_chat import render as render_chat
    render_chat()

with tab_ingest:
    from brain.ui.tab_ingest import render as render_ingest
    render_ingest()

with tab_graph:
    from brain.ui.tab_graph import render as render_graph
    render_graph()

with tab_jobs:
    from brain.ui.tab_jobs import render as render_jobs
    render_jobs()

with tab_settings:
    from brain.ui.tab_settings import render as render_settings
    render_settings()
