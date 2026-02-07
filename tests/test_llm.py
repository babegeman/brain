"""Tests for brain.llm — mocked LiteLLM calls."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from brain.llm import complete, embed, transcribe_image


@pytest.fixture
def mock_completion():
    """Mock litellm.acompletion to return a fake response."""
    choice = MagicMock()
    choice.message.content = "Hello from mock"
    response = MagicMock()
    response.choices = [choice]

    with patch("brain.llm.litellm.acompletion", new_callable=AsyncMock) as mock:
        mock.return_value = response
        yield mock


@pytest.fixture
def mock_embedding():
    """Mock litellm.aembedding to return fake vectors."""
    response = MagicMock()
    response.data = [
        {"embedding": [0.1, 0.2, 0.3]},
        {"embedding": [0.4, 0.5, 0.6]},
    ]

    with patch("brain.llm.litellm.aembedding", new_callable=AsyncMock) as mock:
        mock.return_value = response
        yield mock


@pytest.mark.asyncio
async def test_complete_uses_config_model(mock_completion):
    result = await complete([{"role": "user", "content": "hi"}])
    assert result == "Hello from mock"

    call_kwargs = mock_completion.call_args
    assert "anthropic" in call_kwargs.kwargs["model"]


@pytest.mark.asyncio
async def test_complete_allows_model_override(mock_completion):
    await complete([{"role": "user", "content": "hi"}], model="openai/gpt-4o")

    call_kwargs = mock_completion.call_args
    assert call_kwargs.kwargs["model"] == "openai/gpt-4o"


@pytest.mark.asyncio
async def test_embed_returns_vectors(mock_embedding):
    vectors = await embed(["hello", "world"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 3

    call_kwargs = mock_embedding.call_args
    assert "voyage" in call_kwargs.kwargs["model"]


@pytest.mark.asyncio
async def test_transcribe_image_formats_base64(mock_completion):
    fake_bytes = b"\x89PNG\r\n\x1a\n"
    result = await transcribe_image(fake_bytes)
    assert result == "Hello from mock"

    call_kwargs = mock_completion.call_args
    messages = call_kwargs.kwargs["messages"]
    assert messages[0]["content"][1]["type"] == "image_url"
    assert "base64" in messages[0]["content"][1]["image_url"]["url"]
