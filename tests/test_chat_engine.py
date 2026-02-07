"""Tests for brain.chat.engine and brain.chat.retriever."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from brain.chat.engine import _build_context, chat
from brain.models import Citation


@pytest.fixture
def sample_citations():
    return [
        Citation(
            doc_id="d1",
            chunk_id="c1",
            source_path="/tmp/doc.txt",
            title="Test Doc",
            chunk_index=0,
            snippet="This is a test snippet about AI.",
            score=0.95,
        ),
        Citation(
            doc_id="d2",
            chunk_id="c2",
            source_path="/tmp/other.txt",
            title="Other Doc",
            chunk_index=1,
            snippet="Another piece of relevant info.",
            score=0.82,
        ),
    ]


def test_build_context(sample_citations):
    ctx = _build_context(sample_citations)
    assert "[D1]" in ctx
    assert "[D2]" in ctx
    assert "Test Doc" in ctx
    assert "Other Doc" in ctx
    assert "test snippet" in ctx


def test_build_context_with_graph_facts(sample_citations):
    graph_facts = [
        {"source_node": "AI", "target_node": "ML", "fact": "AI includes ML"},
    ]
    ctx = _build_context(sample_citations, graph_facts)
    assert "[G1]" in ctx
    assert "AI" in ctx
    assert "ML" in ctx


@pytest.mark.asyncio
async def test_chat_calls_retrieve_and_complete(sample_citations):
    """Chat engine should retrieve chunks then call LLM."""
    with (
        patch("brain.chat.engine.retrieve_chunks", new_callable=AsyncMock) as mock_retrieve,
        patch("brain.chat.engine.llm.complete", new_callable=AsyncMock) as mock_complete,
    ):
        mock_retrieve.return_value = sample_citations
        mock_complete.return_value = "The answer is 42. [1][2]"

        # Pass None for vectorstore/docstore since retrieve is mocked
        answer, citations, graph_facts = await chat(
            "What is the meaning of life?",
            vectorstore=None,
            docstore=None,
        )

    assert answer == "The answer is 42. [1][2]"
    assert graph_facts == []
    assert len(citations) == 2
    mock_retrieve.assert_called_once()
    mock_complete.assert_called_once()

    # Verify the messages passed to complete include context and system prompt
    messages = mock_complete.call_args.args[0]
    assert any("Context:" in m["content"] for m in messages if m["role"] == "user")
    system_msg = next(m for m in messages if m["role"] == "system")
    assert "knowledge base" in system_msg["content"]
