"""brain graph — knowledge graph management commands."""

from __future__ import annotations

import asyncio
from typing import Annotated

import typer

CONFIRM_PHRASE = "delete all graph data"

app = typer.Typer(
    name="graph",
    help="Knowledge graph management commands.",
    no_args_is_help=True,
)


@app.command(name="stats")
def stats_cmd():
    """Show node and relationship counts in the knowledge graph."""
    from brain.graph.client import GraphClient

    async def _run():
        gc = GraphClient()
        return await gc.get_stats()

    stats = asyncio.run(_run())
    typer.echo(f"Nodes: {stats['nodes']}")
    typer.echo(f"Relationships: {stats['relationships']}")


@app.command(name="clear")
def clear_cmd(
    confirm: Annotated[
        str,
        typer.Option(
            "--confirm",
            help=f'Type "{CONFIRM_PHRASE}" to confirm deletion.',
        ),
    ] = "",
):
    """Delete ALL nodes and relationships from the knowledge graph.

    Requires --confirm "delete all graph data" to proceed.
    """
    from brain.graph.client import GraphClient

    if confirm != CONFIRM_PHRASE:
        typer.echo("Safety check failed.", err=True)
        typer.echo(
            f'To clear the graph, run:\n\n'
            f'  brain graph clear --confirm "{CONFIRM_PHRASE}"',
            err=True,
        )
        raise typer.Exit(code=1)

    async def _run():
        gc = GraphClient()
        return await gc.clear_graph()

    result = asyncio.run(_run())
    typer.echo(
        f"Graph cleared: {result['deleted_nodes']} nodes, "
        f"{result['deleted_relationships']} relationships deleted."
    )
