"""Retriever — vector search + optional graph search with citation building."""

from __future__ import annotations

from brain import llm
from brain.config import get_settings
from brain.graph.client import GraphClient
from brain.models import Citation
from brain.stores.docstore import DocStore
from brain.stores.vectorstore import VectorStore


async def retrieve_chunks(
    query: str,
    vectorstore: VectorStore,
    docstore: DocStore,
    *,
    top_k: int | None = None,
) -> list[Citation]:
    """Embed query, search Qdrant, and build Citation objects."""
    cfg = get_settings()
    if top_k is None:
        top_k = cfg.chat.top_k
    snippet_length = cfg.chat.snippet_length

    query_emb = (await llm.embed([query]))[0]
    results = vectorstore.search(query_emb, top_k=top_k)

    citations: list[Citation] = []
    for r in results:
        doc = docstore.get_document(r["doc_id"])
        citations.append(
            Citation(
                doc_id=r["doc_id"],
                chunk_id=r["chunk_id"],
                source_path=doc.source_path if doc else "",
                title=doc.title if doc else "",
                chunk_index=r.get("index", 0),
                snippet=r["text"][:snippet_length],
                score=r["score"],
            )
        )
    return citations


async def retrieve_fusion(
    query: str,
    vectorstore: VectorStore,
    docstore: DocStore,
    graph_client: GraphClient,
    *,
    top_k: int | None = None,
    graph_results: int | None = None,
) -> tuple[list[Citation], list[dict]]:
    """Fusion retrieval: vector chunks + graph facts.

    Returns:
        citations: list of Citation from vector search
        graph_facts: list of dicts from Graphiti search
    """
    cfg = get_settings()
    if graph_results is None:
        graph_results = cfg.chat.graph_results

    citations = await retrieve_chunks(query, vectorstore, docstore, top_k=top_k)

    graph_facts = []
    if graph_client.is_initialized:
        graph_facts = await graph_client.search(
            query, num_results=graph_results
        )

    return citations, graph_facts
