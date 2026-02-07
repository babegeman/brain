"""brain chat — ask questions against the knowledge base."""

from __future__ import annotations

import asyncio
import json
from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel


def chat_cmd(
    query: Annotated[str, typer.Argument(help="Question to ask your second brain")],
    top_k: Annotated[
        int | None, typer.Option("--top-k", "-k", help="Number of chunks to retrieve")
    ] = None,
    use_graph: Annotated[
        bool, typer.Option("--graph/--no-graph", help="Enable graph-enhanced retrieval")
    ] = False,
):
    """Chat with your second brain."""
    from brain.chat.engine import chat
    from brain.cli.app import is_json
    from brain.config import get_settings
    from brain.stores.docstore import DocStore
    from brain.stores.vectorstore import VectorStore

    settings = get_settings()
    if top_k is None:
        top_k = settings.chat.top_k
    docstore = DocStore(settings.docstore.path)
    vectorstore = VectorStore()
    graph_client = None

    async def _run():
        nonlocal graph_client
        gc = None
        if use_graph:
            from brain.graph.client import GraphClient

            gc = GraphClient()
            await gc.initialize()

        answer, citations, graph_facts = await chat(
            query, vectorstore, docstore, graph_client=gc, top_k=top_k
        )
        return answer, citations, graph_facts, gc

    try:
        answer, citations, graph_facts, graph_client = asyncio.run(_run())

        if is_json():
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "answer": answer,
                        "citations": [
                            {
                                "doc_id": c.doc_id,
                                "chunk_id": c.chunk_id,
                                "title": c.title,
                                "source_path": c.source_path,
                                "chunk_index": c.chunk_index,
                                "snippet": c.snippet,
                                "score": c.score,
                            }
                            for c in citations
                        ],
                        "graph_facts": graph_facts,
                    },
                    indent=2,
                )
            )
        else:
            console = Console()
            console.print(Panel(Markdown(answer), title="Answer", border_style="green"))

            if citations:
                console.print("\n[bold]Document Sources:[/bold]")
                for i, c in enumerate(citations, 1):
                    source = c.title or c.source_path
                    console.print(f"  [D{i}] {source} (score: {c.score:.3f})")

            if graph_facts:
                console.print("\n[bold]Graph Facts:[/bold]")
                for i, gf in enumerate(graph_facts, 1):
                    console.print(
                        f"  [G{i}] {gf['source_node']} → {gf['target_node']}: {gf['fact']}"
                    )
    finally:
        if graph_client:
            asyncio.run(graph_client.close())
        vectorstore.close()
        docstore.close()
