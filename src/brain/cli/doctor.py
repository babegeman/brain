"""brain doctor — validate config, connectivity, and credentials."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from rich.console import Console
from rich.table import Table


async def _check_config() -> tuple[bool, str]:
    """Verify config loads without error."""
    try:
        from brain.config import get_settings
        settings = get_settings()
        profile = settings.llm
        return True, f"profile={settings.active_profile}, chat={profile.chat_model}"
    except Exception as e:
        return False, str(e)


async def _check_neo4j() -> tuple[bool, str]:
    """Test bolt connection to Neo4j."""
    try:
        from neo4j import AsyncGraphDatabase
        from brain.config import get_settings
        cfg = get_settings().neo4j
        driver = AsyncGraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))
        async with driver.session() as session:
            result = await session.run("RETURN 1 AS n")
            record = await result.single()
            assert record["n"] == 1
        await driver.close()
        return True, f"connected to {cfg.uri}"
    except Exception as e:
        return False, str(e)


async def _check_llm() -> tuple[bool, str]:
    """Call LiteLLM with a trivial prompt."""
    try:
        from brain.llm import complete
        answer = await complete(
            [{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=10,
        )
        return True, f"got response ({len(answer)} chars)"
    except Exception as e:
        return False, str(e)


async def _check_embed() -> tuple[bool, str]:
    """Call LiteLLM embedding with a trivial input."""
    try:
        from brain.llm import embed
        vectors = await embed(["test"])
        dim = len(vectors[0])
        return True, f"dim={dim}"
    except Exception as e:
        return False, str(e)


async def _check_data_dirs() -> tuple[bool, str]:
    """Verify data directories are writable."""
    try:
        from brain.config import get_settings
        settings = get_settings()
        dirs = [
            Path(settings.qdrant.path),
            Path(settings.docstore.path).parent,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        for folder in settings.watched_folders:
            Path(folder).mkdir(parents=True, exist_ok=True)
        return True, "all data directories writable"
    except Exception as e:
        return False, str(e)


async def _check_env_vars() -> tuple[bool, str]:
    """Check that required environment variables are set."""
    missing = []
    for var in ["ANTHROPIC_API_KEY", "VOYAGE_API_KEY"]:
        if not os.environ.get(var):
            missing.append(var)
    if missing:
        return False, f"missing: {', '.join(missing)}"
    return True, "ANTHROPIC_API_KEY, VOYAGE_API_KEY set"


async def _run_checks() -> list[dict]:
    """Run all checks and return results."""
    checks = [
        ("Config", _check_config),
        ("Env Vars", _check_env_vars),
        ("Data Dirs", _check_data_dirs),
        ("Neo4j", _check_neo4j),
        ("LLM (chat)", _check_llm),
        ("Embeddings", _check_embed),
    ]
    results = []
    for name, check_fn in checks:
        ok, detail = await check_fn()
        results.append({"check": name, "ok": ok, "detail": detail})
    return results


def doctor_cmd():
    """Check system health and connectivity."""
    from brain.cli.app import is_json

    results = asyncio.run(_run_checks())

    if is_json():
        print(json.dumps(results, indent=2))
        return

    console = Console()
    table = Table(title="brain doctor", show_lines=True)
    table.add_column("Check", style="bold")
    table.add_column("Status")
    table.add_column("Detail")

    all_ok = True
    for r in results:
        status = "[green]PASS[/green]" if r["ok"] else "[red]FAIL[/red]"
        if not r["ok"]:
            all_ok = False
        table.add_row(r["check"], status, r["detail"])

    console.print(table)
    if all_ok:
        console.print("\n[bold green]All checks passed.[/bold green]")
    else:
        console.print("\n[bold yellow]Some checks failed. See details above.[/bold yellow]")
        raise typer.Exit(code=1)


# Avoid circular import — typer is imported in app.py
import typer  # noqa: E402
