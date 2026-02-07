"""brain ingest — ingest files or folders into the second brain."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table


def ingest_cmd(
    path: Annotated[Path, typer.Argument(help="File or folder to ingest")],
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Re-ingest even if unchanged")
    ] = False,
    graph: Annotated[
        bool, typer.Option("--graph", "-g", help="Extract knowledge graph via Neo4j/Graphiti")
    ] = False,
):
    """Ingest documents into the second brain."""
    from brain.cli.app import is_json
    from brain.config import get_settings
    from brain.ingest.pipeline import ingest_file, ingest_folder
    from brain.stores.docstore import DocStore
    from brain.stores.vectorstore import VectorStore

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    settings = get_settings()
    docstore = DocStore(settings.docstore.path)
    vectorstore = VectorStore()

    async def _run():
        graph_client = None
        if graph:
            from brain.graph.client import GraphClient

            graph_client = GraphClient()
            await graph_client.initialize()

        try:
            resolved = path.resolve()
            if resolved.is_dir():
                return await ingest_folder(
                    resolved, docstore, vectorstore, graph_client, force=force
                )
            elif resolved.is_file():
                doc = await ingest_file(
                    resolved, docstore, vectorstore, graph_client, force=force
                )
                return [doc] if doc else []
            else:
                typer.echo(f"Path not found: {resolved}", err=True)
                raise typer.Exit(code=1)
        finally:
            if graph_client:
                await graph_client.close()

    try:
        docs = asyncio.run(_run())

        if is_json():
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "ingested": len(docs),
                        "documents": [
                            {
                                "doc_id": d.doc_id,
                                "title": d.title,
                                "source_path": d.source_path,
                                "doc_type": d.doc_type.value,
                                "chunks": len(docstore.get_chunks_for_doc(d.doc_id)),
                            }
                            for d in docs
                        ],
                    },
                    indent=2,
                )
            )
        else:
            console = Console()
            if not docs:
                console.print("[yellow]No new documents to ingest.[/yellow]")
                return

            table = Table(title=f"Ingested {len(docs)} document(s)")
            table.add_column("Title")
            table.add_column("Type")
            table.add_column("Chunks", justify="right")
            table.add_column("Path")

            for d in docs:
                n_chunks = len(docstore.get_chunks_for_doc(d.doc_id))
                table.add_row(d.title, d.doc_type.value, str(n_chunks), d.source_path)

            console.print(table)
    finally:
        vectorstore.close()
        docstore.close()
