"""Tests for brain.graph.bridge — config translation to Graphiti clients."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from brain.config import LLMProfile
from brain.graph.bridge import (
    _parse_provider_model,
    build_graphiti_embedder,
    build_graphiti_llm_client,
)


def test_parse_provider_model_with_prefix():
    assert _parse_provider_model("anthropic/claude-3-5-haiku") == ("anthropic", "claude-3-5-haiku")
    assert _parse_provider_model("openai/gpt-4o") == ("openai", "gpt-4o")
    assert _parse_provider_model("voyage/voyage-3.5-lite") == ("voyage", "voyage-3.5-lite")
    assert _parse_provider_model("ollama/llama3.1:8b") == ("ollama", "llama3.1:8b")


def test_parse_provider_model_without_prefix():
    assert _parse_provider_model("gpt-4o") == ("openai", "gpt-4o")


def test_parse_provider_model_bedrock():
    provider, model = _parse_provider_model("bedrock/anthropic.claude-3-sonnet")
    assert provider == "bedrock"
    assert model == "anthropic.claude-3-sonnet"


def test_build_llm_client_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    profile = LLMProfile(
        graph_extract_model="anthropic/claude-3-5-haiku-20241022",
        graph_extract_small_model="anthropic/claude-3-5-haiku-20241022",
    )
    client = build_graphiti_llm_client(profile)
    from graphiti_core.llm_client.anthropic_client import AnthropicClient
    assert isinstance(client, AnthropicClient)


def test_build_llm_client_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    profile = LLMProfile(
        graph_extract_model="openai/gpt-4o",
        graph_extract_small_model="openai/gpt-4o-mini",
    )
    client = build_graphiti_llm_client(profile)
    from graphiti_core.llm_client.openai_client import OpenAIClient
    assert isinstance(client, OpenAIClient)


def test_build_llm_client_unsupported_raises():
    profile = LLMProfile(
        graph_extract_model="bedrock/anthropic.claude-3-sonnet",
        graph_extract_small_model="bedrock/anthropic.claude-3-haiku",
    )
    with pytest.raises(ValueError, match="Unsupported"):
        build_graphiti_llm_client(profile)


def test_build_embedder_voyage(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    profile = LLMProfile(
        embed_model="voyage/voyage-3.5-lite",
        embed_dim=1024,
    )
    embedder = build_graphiti_embedder(profile)
    from graphiti_core.embedder.voyage import VoyageAIEmbedder
    assert isinstance(embedder, VoyageAIEmbedder)


def test_build_embedder_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    profile = LLMProfile(
        embed_model="openai/text-embedding-3-small",
        embed_dim=1536,
    )
    embedder = build_graphiti_embedder(profile)
    from graphiti_core.embedder.openai import OpenAIEmbedder
    assert isinstance(embedder, OpenAIEmbedder)


def test_build_embedder_unsupported_raises():
    profile = LLMProfile(
        embed_model="bedrock/amazon.titan-embed",
        embed_dim=1024,
    )
    with pytest.raises(ValueError, match="Unsupported"):
        build_graphiti_embedder(profile)
