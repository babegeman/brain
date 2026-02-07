"""Ingestion pipeline — orchestrates normalize → chunk → store → embed."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path

from brain import llm
from brain.config import get_settings
from brain.graph.client import GraphClient
from brain.ingest.chunker import chunk_document
from brain.ingest.normalizers import NORMALIZERS, get_normalizer
from brain.models import Document
from brain.stores.docstore import DocStore
from brain.stores.vectorstore import VectorStore

log = logging.getLogger(__name__)


async def ingest_file(
    path: Path,
    docstore: DocStore,
    vectorstore: VectorStore | None = None,
    graph_client: GraphClient | None = None,
    *,
    force: bool = False,
) -> Document | None:
    """Ingest a single file: normalize → chunk → store → embed → graph.

    Args:
        path: Path to the file.
        docstore: DocStore instance for text/metadata persistence.
        vectorstore: Optional VectorStore for embedding storage.
        force: Re-ingest even if the file hash hasn't changed.

    Returns:
        The Document if ingested, None if skipped.
    """
    path = path.resolve()
    suffix = path.suffix.lower()
    normalizer = get_normalizer(path)

    if normalizer is None:
        log.warning("Unsupported file type: %s", suffix)
        return None

    # Check if file has changed since last ingestion
    if not force:
        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if docstore.is_unchanged(str(path), content_hash):
            log.info("Skipping unchanged file: %s", path.name)
            return None

    log.info("Ingesting: %s", path.name)

    # Normalize (async for images, sync for everything else)
    if asyncio.iscoroutinefunction(normalizer):
        doc = await normalizer(path)
    else:
        doc = normalizer(path)

    # Check for re-ingestion: if this path was previously ingested,
    # reuse the doc_id and clean up old chunks/vectors
    existing = docstore.get_document_by_path(str(path))
    if existing:
        doc.doc_id = existing.doc_id
        docstore.delete_doc_chunks(existing.doc_id)
        if vectorstore:
            vectorstore.delete_by_doc_id(existing.doc_id)

    # Chunk
    cfg = get_settings().chunker
    chunks = chunk_document(
        doc_id=doc.doc_id,
        text=doc.text,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        metadata={"source_path": doc.source_path, "doc_type": doc.doc_type.value},
    )

    # Store text + metadata
    docstore.upsert_document(doc)
    docstore.upsert_chunks(chunks)

    # Embed and store vectors
    if vectorstore and chunks:
        texts = [c.text for c in chunks]
        embeddings = await llm.embed(texts)
        vectorstore.upsert_chunks(chunks, embeddings)
        log.info("Embedded %d chunks for %s", len(chunks), path.name)

    # Graph extraction (after vector store, so retrieval works even if graph fails)
    if graph_client and graph_client.is_initialized:
        try:
            await graph_client.add_document(doc)
        except Exception:
            log.exception("Graph extraction failed for %s (non-fatal)", path.name)

    log.info("Ingested %s: %d chunks", path.name, len(chunks))
    return doc


async def ingest_folder(
    folder: Path,
    docstore: DocStore,
    vectorstore: VectorStore | None = None,
    graph_client: GraphClient | None = None,
    *,
    force: bool = False,
) -> list[Document]:
    """Ingest all supported files in a folder recursively.

    Returns list of ingested Documents (skipped files are excluded).
    """
    folder = folder.resolve()
    docs: list[Document] = []

    supported_files = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in NORMALIZERS
    )

    for path in supported_files:
        doc = await ingest_file(path, docstore, vectorstore, graph_client, force=force)
        if doc:
            docs.append(doc)

    return docs
