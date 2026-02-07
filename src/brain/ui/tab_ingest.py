"""Ingest tab — upload files, trigger folder ingestion, view documents.

Features a real-time log display that streams pipeline messages
(normalisation, chunking, embedding batches, graph extraction)
into an expandable Streamlit status container.
"""

from __future__ import annotations

import asyncio
import logging
import re
import tempfile
import time
from pathlib import Path

import streamlit as st


# ---------------------------------------------------------------------------
# Streamlit-aware log handler
# ---------------------------------------------------------------------------


class _StreamlitLogHandler(logging.Handler):
    """Captures ``brain.*`` log records and streams them into Streamlit.

    * Updates a scrolling code block with recent log lines.
    * Parses structured messages to drive a progress bar
      (embedding batches and graph-extraction chunks).
    """

    def __init__(
        self,
        log_container,
        progress_bar=None,
        max_lines: int = 50,
    ):
        super().__init__(level=logging.INFO)
        self.log_container = log_container
        self.progress_bar = progress_bar
        self.lines: list[str] = []
        self.max_lines = max_lines
        self._graph_re = re.compile(r"Graph extraction episode (\d+)/(\d+)")
        self._embed_re = re.compile(r"Embedding batch (\d+)/(\d+)")
        self._last_update = 0.0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.lines.append(msg)
            if len(self.lines) > self.max_lines:
                self.lines = self.lines[-self.max_lines :]

            # Throttle the log-area refresh to avoid flooding the websocket
            now = time.monotonic()
            if now - self._last_update >= 0.3:
                self.log_container.code("\n".join(self.lines))
                self._last_update = now

            # Drive the progress bar from structured messages
            if self.progress_bar:
                m = self._graph_re.search(msg)
                if m:
                    cur, tot = int(m.group(1)), int(m.group(2))
                    self.progress_bar.progress(
                        cur / tot,
                        text=f"Graph: chunk {cur}/{tot}",
                    )
                else:
                    m = self._embed_re.search(msg)
                    if m:
                        cur, tot = int(m.group(1)), int(m.group(2))
                        self.progress_bar.progress(
                            cur / tot,
                            text=f"Embedding: batch {cur}/{tot}",
                        )
        except Exception:
            pass  # never let a handler crash the app

    def flush(self) -> None:
        """Force a final UI update."""
        if self.lines:
            self.log_container.code("\n".join(self.lines))


# ---------------------------------------------------------------------------
# Public render
# ---------------------------------------------------------------------------


def render():
    st.subheader("Ingest Documents")

    from brain.config import get_settings

    settings = get_settings()

    col1, col2 = st.columns(2)

    # --- File upload ---
    with col1:
        st.markdown("**Upload files**")
        uploaded = st.file_uploader(
            "Choose files to ingest",
            accept_multiple_files=True,
            type=[
                "docx", "pptx", "txt", "md", "csv",
                "json", "yaml", "yml", "png", "jpg", "jpeg",
            ],
        )

    # --- Watched folders ---
    with col2:
        st.markdown("**Watched folders**")
        if settings.watched_folders:
            for folder in settings.watched_folders:
                st.code(folder)
        else:
            st.caption("No watched folders configured.")

    # --- Options ---
    opt1, opt2, _ = st.columns(3)
    with opt1:
        graph_enabled = st.checkbox("Extract knowledge graph", key="ingest_graph")
    with opt2:
        force_reingest = st.checkbox("Force re-ingest", key="ingest_force")

    # --- Action buttons ---
    btn1, btn2 = st.columns(2)
    with btn1:
        if uploaded and st.button("Ingest uploaded files", key="ingest_upload"):
            _ingest_uploads(uploaded, graph=graph_enabled, force=force_reingest)
    with btn2:
        if settings.watched_folders and st.button(
            "Ingest watched folders", key="ingest_watched"
        ):
            _ingest_watched_folders(graph=graph_enabled, force=force_reingest)

    st.divider()

    # --- Document list ---
    st.markdown("**Ingested Documents**")
    _show_documents()


# ---------------------------------------------------------------------------
# Ingestion helpers
# ---------------------------------------------------------------------------


def _attach_log_handler(handler: _StreamlitLogHandler) -> logging.Logger:
    """Attach *handler* to the ``brain`` logger and return the logger."""
    logger = logging.getLogger("brain")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def _ingest_uploads(uploaded_files, *, graph: bool = False, force: bool = False):
    """Ingest uploaded files with a live progress display."""
    from brain.config import get_settings
    from brain.ingest.pipeline import ingest_file
    from brain.stores.docstore import DocStore
    from brain.stores.vectorstore import VectorStore

    settings = get_settings()
    docstore = DocStore(settings.docstore.path)
    vectorstore = VectorStore()

    with st.status("Ingesting uploaded files...", expanded=True) as status:
        file_progress = st.progress(0, text="Preparing...")
        phase_progress = st.progress(0, text="")
        log_area = st.empty()

        handler = _StreamlitLogHandler(log_area, progress_bar=phase_progress)
        handler.setFormatter(logging.Formatter("%(levelname)-5s %(message)s"))
        logger = _attach_log_handler(handler)

        async def _run():
            graph_client = None
            if graph:
                from brain.graph.client import GraphClient

                graph_client = GraphClient()
                await graph_client.initialize()

            try:
                results = []
                n = len(uploaded_files)
                for i, uf in enumerate(uploaded_files):
                    file_progress.progress(
                        i / n, text=f"Processing {uf.name} ({i + 1}/{n})"
                    )
                    phase_progress.progress(0, text="")

                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=Path(uf.name).suffix
                    ) as tmp:
                        tmp.write(uf.getvalue())
                        tmp_path = Path(tmp.name)

                    doc = await ingest_file(
                        tmp_path, docstore, vectorstore, graph_client, force=force
                    )
                    if doc:
                        results.append(doc)
                    file_progress.progress(
                        (i + 1) / n, text=f"Done: {uf.name} ({i + 1}/{n})"
                    )
                return results
            finally:
                if graph_client:
                    await graph_client.close()

        try:
            results = asyncio.run(_run())
        except Exception as e:
            status.update(label=f"Error: {e}", state="error", expanded=True)
            results = None
        finally:
            handler.flush()
            logger.removeHandler(handler)
            vectorstore.close()
            docstore.close()

        if results is None:
            return
        if results:
            titles = ", ".join(d.title for d in results)
            status.update(
                label=f"Ingested {len(results)} file(s): {titles}",
                state="complete",
                expanded=False,
            )
        else:
            status.update(
                label="No new files to ingest",
                state="complete",
                expanded=False,
            )


def _ingest_watched_folders(*, graph: bool = False, force: bool = False):
    """Ingest all watched folders with a live progress display."""
    from brain.config import get_settings
    from brain.ingest.pipeline import ingest_folder
    from brain.stores.docstore import DocStore
    from brain.stores.vectorstore import VectorStore

    settings = get_settings()
    docstore = DocStore(settings.docstore.path)
    vectorstore = VectorStore()

    with st.status("Ingesting watched folders...", expanded=True) as status:
        folder_progress = st.progress(0, text="Scanning folders...")
        phase_progress = st.progress(0, text="")
        log_area = st.empty()

        handler = _StreamlitLogHandler(log_area, progress_bar=phase_progress)
        handler.setFormatter(logging.Formatter("%(levelname)-5s %(message)s"))
        logger = _attach_log_handler(handler)

        async def _run():
            graph_client = None
            if graph:
                from brain.graph.client import GraphClient

                graph_client = GraphClient()
                await graph_client.initialize()

            try:
                all_docs = []
                folders = [f for f in settings.watched_folders if Path(f).exists()]
                n = max(len(folders), 1)
                for i, folder in enumerate(folders):
                    name = Path(folder).name
                    folder_progress.progress(
                        i / n, text=f"Processing {name} ({i + 1}/{len(folders)})"
                    )
                    phase_progress.progress(0, text="")

                    docs = await ingest_folder(
                        Path(folder), docstore, vectorstore, graph_client, force=force
                    )
                    all_docs.extend(docs)
                    folder_progress.progress(
                        (i + 1) / n, text=f"Done: {name} ({i + 1}/{len(folders)})"
                    )
                return all_docs
            finally:
                if graph_client:
                    await graph_client.close()

        try:
            results = asyncio.run(_run())
        except Exception as e:
            status.update(label=f"Error: {e}", state="error", expanded=True)
            results = None
        finally:
            handler.flush()
            logger.removeHandler(handler)
            vectorstore.close()
            docstore.close()

        if results is None:
            return
        if results:
            status.update(
                label=f"Ingested {len(results)} document(s) from watched folders",
                state="complete",
                expanded=False,
            )
        else:
            status.update(
                label="No new documents found in watched folders",
                state="complete",
                expanded=False,
            )


# ---------------------------------------------------------------------------
# Document list
# ---------------------------------------------------------------------------


def _show_documents():
    """Display a table of all ingested documents."""
    from brain.config import get_settings
    from brain.stores.docstore import DocStore

    settings = get_settings()
    docstore = DocStore(settings.docstore.path)
    docs = docstore.list_documents()
    docstore.close()

    if not docs:
        st.info("No documents ingested yet.")
        return

    rows = []
    for d in docs:
        rows.append({
            "Title": d.title,
            "Type": d.doc_type.value,
            "Path": d.source_path,
            "Ingested": str(d.ingested_at)[:19],
        })
    st.dataframe(rows, use_container_width=True)
