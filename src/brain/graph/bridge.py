"""Bridge between our config and Graphiti's LLM/embedder configuration.

Translates LiteLLM-style model strings (e.g. "anthropic/claude-3-5-haiku-20241022")
into Graphiti's native client objects (AnthropicClient, VoyageAIEmbedder, etc.).
"""

from __future__ import annotations

import os

from graphiti_core.cross_encoder.client import CrossEncoderClient

from brain.config import LLMProfile, get_settings


class NoOpCrossEncoder(CrossEncoderClient):
    """A no-op cross-encoder that returns passages with uniform scores.

    Used to avoid the default OpenAI reranker (which requires an API key)
    when we only use Graphiti's basic search() method.
    """

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        return [(p, 1.0) for p in passages]


def _parse_provider_model(litellm_model: str) -> tuple[str, str]:
    """Split 'provider/model-name' into ('provider', 'model-name').

    Examples:
        'anthropic/claude-3-5-haiku-20241022' -> ('anthropic', 'claude-3-5-haiku-20241022')
        'openai/gpt-4o' -> ('openai', 'gpt-4o')
        'gpt-4o' -> ('openai', 'gpt-4o')
    """
    if "/" in litellm_model:
        provider, model = litellm_model.split("/", 1)
        return provider.lower(), model
    return "openai", litellm_model


def build_graphiti_llm_client(profile: LLMProfile | None = None):
    """Create a Graphiti LLMClient from our config."""
    from graphiti_core.llm_client import LLMConfig

    if profile is None:
        profile = get_settings().llm

    provider, model = _parse_provider_model(profile.graph_extract_model)
    _, small_model = _parse_provider_model(profile.graph_extract_small_model)

    if provider == "anthropic":
        from graphiti_core.llm_client.anthropic_client import AnthropicClient

        return AnthropicClient(
            config=LLMConfig(
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
                model=model,
                small_model=small_model,
            ),
        )

    if provider == "openai":
        from graphiti_core.llm_client.openai_client import OpenAIClient

        return OpenAIClient(
            config=LLMConfig(
                api_key=os.environ.get("OPENAI_API_KEY"),
                model=model,
                small_model=small_model,
            ),
        )

    if provider in ("ollama", "ollama_chat"):
        from graphiti_core.llm_client.openai_client import OpenAIClient

        return OpenAIClient(
            config=LLMConfig(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
                model=model,
                small_model=small_model,
            ),
        )

    if provider == "groq":
        from graphiti_core.llm_client.groq_client import GroqClient

        return GroqClient(
            config=LLMConfig(
                api_key=os.environ.get("GROQ_API_KEY"),
                model=model,
                small_model=small_model,
            ),
        )

    raise ValueError(
        f"Unsupported Graphiti LLM provider: '{provider}'. "
        f"Supported: anthropic, openai, ollama, groq. "
        f"For Bedrock, run `litellm --port 4000 --model {profile.graph_extract_model}` "
        f"and set graph_extract_model to 'openai/<model>' with OPENAI_API_KEY and "
        f"base_url pointed at the proxy."
    )


def build_graphiti_embedder(profile: LLMProfile | None = None):
    """Create a Graphiti EmbedderClient from our config."""
    if profile is None:
        profile = get_settings().llm

    provider, model = _parse_provider_model(profile.embed_model)

    if provider == "voyage":
        from graphiti_core.embedder.voyage import VoyageAIEmbedder, VoyageAIEmbedderConfig

        return VoyageAIEmbedder(
            config=VoyageAIEmbedderConfig(
                embedding_model=model,
                api_key=os.environ.get("VOYAGE_API_KEY"),
            ),
        )

    if provider == "openai":
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

        return OpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                api_key=os.environ.get("OPENAI_API_KEY"),
                embedding_model=model,
                embedding_dim=profile.embed_dim,
            ),
        )

    if provider in ("ollama", "ollama_chat"):
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

        return OpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
                embedding_model=model,
                embedding_dim=profile.embed_dim,
            ),
        )

    raise ValueError(
        f"Unsupported Graphiti embedding provider: '{provider}'. "
        f"Supported: voyage, openai, ollama."
    )
