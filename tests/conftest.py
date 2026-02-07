"""Shared test fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Ensure tests run from the project root so config.default.yaml is found
PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture(autouse=True)
def _set_project_root(monkeypatch, tmp_path):
    """Point BRAIN_ROOT at the project root and use tmp_path for data."""
    monkeypatch.setenv("BRAIN_ROOT", str(PROJECT_ROOT))
    monkeypatch.setenv("BRAIN_QDRANT__PATH", str(tmp_path / "qdrant"))
    monkeypatch.setenv("BRAIN_DOCSTORE__PATH", str(tmp_path / "docstore.db"))

    # Reset settings cache between tests
    from brain.config import reset_settings
    reset_settings()
