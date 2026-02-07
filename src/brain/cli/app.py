"""brain CLI — Typer entrypoint with global options."""

from __future__ import annotations

import os
from typing import Annotated, Optional

import typer

app = typer.Typer(
    name="brain",
    help="Second Brain CLI — ingest, chat, and debug your knowledge base.",
    no_args_is_help=True,
)

# Global state shared across subcommands
_state: dict = {"json": False}


def is_json() -> bool:
    """Check if --json output mode is active."""
    return _state["json"]


@app.callback()
def main(
    json_output: Annotated[
        bool, typer.Option("--json", help="Output as JSON (agent-friendly)")
    ] = False,
    profile: Annotated[
        Optional[str], typer.Option("--profile", "-p", help="Override active LLM profile")
    ] = None,
    root: Annotated[
        Optional[str], typer.Option("--root", help="Override project root directory")
    ] = None,
):
    """Global options applied before any subcommand."""
    _state["json"] = json_output
    if root:
        os.environ["BRAIN_ROOT"] = root
    if profile:
        os.environ["BRAIN_ACTIVE_PROFILE"] = profile


# Register subcommands -------------------------------------------------------

from brain.cli.chat_cmd import chat_cmd  # noqa: E402
from brain.cli.doctor import doctor_cmd  # noqa: E402
from brain.cli.graph_cmd import app as graph_app  # noqa: E402
from brain.cli.ingest_cmd import ingest_cmd  # noqa: E402

app.command(name="chat", help="Chat with your second brain.")(chat_cmd)
app.command(name="doctor", help="Check system health and connectivity.")(doctor_cmd)
app.add_typer(graph_app, name="graph", help="Knowledge graph management.")
app.command(name="ingest", help="Ingest documents into the second brain.")(ingest_cmd)
