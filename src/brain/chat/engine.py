"""Chat engine — DocRAG + optional GraphRAG fusion with citations."""

from __future__ import annotations

from collections.abc import AsyncIterator

from brain import llm
from brain.chat.retriever import retrieve_chunks, retrieve_fusion
from brain.config import get_settings
from brain.graph.client import GraphClient
from brain.models import Citation
from brain.stores.docstore import DocStore
from brain.stores.vectorstore import VectorStore


def _build_context(
    citations: list[Citation],
    graph_facts: list[dict] | None = None,
) -> str:
    """Format citations and graph facts as a numbered context block."""
    parts = []
    for i, c in enumerate(citations, 1):
        source = c.title or c.source_path
        parts.append(f"[D{i}] ({source}):\n{c.snippet}")

    if graph_facts:
        for i, gf in enumerate(graph_facts, 1):
            parts.append(
                f"[G{i}] {gf['source_node']} → {gf['target_node']}: {gf['fact']}"
            )

    return "\n\n".join(parts)


async def chat(
    query: str,
    vectorstore: VectorStore,
    docstore: DocStore,
    *,
    graph_client: GraphClient | None = None,
    history: list[dict] | None = None,
    top_k: int | None = None,
    stream: bool = False,
) -> tuple[str | AsyncIterator[str], list[Citation], list[dict]]:
    """DocRAG + GraphRAG fusion chat.

    Returns:
        (answer_text, citations, graph_facts) when stream=False
        (async_iterator, citations, graph_facts) when stream=True
    """
    cfg = get_settings()
    graph_facts: list[dict] = []

    if graph_client and graph_client.is_initialized:
        citations, graph_facts = await retrieve_fusion(
            query, vectorstore, docstore, graph_client, top_k=top_k
        )
        system = cfg.prompts.system_prompt_fusion
    else:
        citations = await retrieve_chunks(query, vectorstore, docstore, top_k=top_k)
        system = cfg.prompts.system_prompt

    context = _build_context(citations, graph_facts)

    messages = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append(
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    )

    answer = await llm.complete(messages, stream=stream)
    return answer, citations, graph_facts
