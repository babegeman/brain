"""Pydantic Settings with YAML profile support.

Priority (highest first): env vars > .env > config.yaml > config.default.yaml
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict, YamlConfigSettingsSource


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class LLMProfile(BaseModel):
    """One named LLM configuration profile."""

    chat_model: str = "anthropic/claude-sonnet-4-5-20250929"
    embed_model: str = "voyage/voyage-4-lite"
    embed_dim: int = 1024
    graph_extract_model: str = "anthropic/claude-haiku-4-5-20251001"
    graph_extract_small_model: str = "anthropic/claude-haiku-4-5-20251001"
    vision_model: str = "anthropic/claude-sonnet-4-5-20250929"
    temperature: float = 0.1
    max_tokens: int = 4096


class Neo4jConfig(BaseModel):
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "memgraph"


class QdrantConfig(BaseModel):
    path: str = "./data/qdrant"
    collection: str = "brain_chunks"


class DocstoreConfig(BaseModel):
    path: str = "./data/docstore.db"


class ChunkerConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 64


class ChatConfig(BaseModel):
    """Defaults for chat retrieval. UI sliders override per-session."""

    top_k: int = 5
    graph_results: int = 10
    snippet_length: int = 300


class GraphConfig(BaseModel):
    """Knowledge graph defaults."""

    group_id: str = "brain"
    num_results: int = 10
    episode_size: int = 100_000  # max chars per Graphiti episode (~25K tokens)
    max_coroutines: int = 3  # max parallel LLM calls inside Graphiti (default 20 is too aggressive)
    episode_delay: int = 15  # seconds to wait between episodes for rate limit cooldown


class PromptsConfig(BaseModel):
    """System prompts used by the chat engine."""

    system_prompt: str = (
        "You are a helpful assistant with access to a personal knowledge base.\n"
        "Answer the user's question based on the provided context excerpts.\n"
        "Always cite your sources using [D1], [D2], etc. for document excerpts.\n"
        "If the context doesn't contain enough information, say so clearly."
    )
    system_prompt_fusion: str = (
        "You are a helpful assistant with access to a personal knowledge base.\n"
        "You have two sources of information:\n"
        "1. Document excerpts (cited as [D1], [D2], etc.)\n"
        "2. Knowledge graph facts (cited as [G1], [G2], etc.)\n\n"
        "Answer the user's question using BOTH sources when relevant.\n"
        "Always cite your sources with the appropriate notation."
    )


# ---------------------------------------------------------------------------
# Main settings
# ---------------------------------------------------------------------------

def _yaml_files() -> list[Path]:
    """Return YAML config file paths relative to the project root."""
    root = Path(os.environ.get("BRAIN_ROOT", "."))
    files = [root / "config.default.yaml"]
    user_cfg = root / "config.yaml"
    if user_cfg.exists():
        files.append(user_cfg)
    return files


class Settings(BaseSettings):
    """Application settings loaded from YAML + env vars."""

    model_config = SettingsConfigDict(
        env_prefix="BRAIN_",
        env_nested_delimiter="__",
    )

    active_profile: str = "default"
    profiles: dict[str, LLMProfile] = {}
    neo4j: Neo4jConfig = Neo4jConfig()
    qdrant: QdrantConfig = QdrantConfig()
    docstore: DocstoreConfig = DocstoreConfig()
    watched_folders: list[str] = ["./data/watched"]
    chunker: ChunkerConfig = ChunkerConfig()
    chat: ChatConfig = ChatConfig()
    graph: GraphConfig = GraphConfig()
    prompts: PromptsConfig = PromptsConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(
                settings_cls,
                yaml_file=_yaml_files(),
            ),
        )

    @property
    def llm(self) -> LLMProfile:
        """Return the currently active LLM profile."""
        if self.active_profile not in self.profiles:
            available = ", ".join(self.profiles.keys()) or "(none)"
            raise KeyError(
                f"Profile '{self.active_profile}' not found. Available: {available}"
            )
        return self.profiles[self.active_profile]


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_settings(**kwargs: Any) -> Settings:
    """Lazy singleton for settings. Call reset_settings() to reload."""
    root = Path(os.environ.get("BRAIN_ROOT", "."))
    load_dotenv(root / ".env", override=False)
    return Settings(**kwargs)


def reset_settings() -> None:
    """Clear the settings cache so the next get_settings() reloads from disk."""
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (mutates base)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def save_user_config(overrides: dict) -> Path:
    """Write user overrides to config.yaml and reset the settings cache.

    Deep-merges *overrides* into the existing config.yaml (if any) so
    keys not present in *overrides* are preserved.  After writing, the
    settings singleton is cleared so the next ``get_settings()`` call
    picks up the new values.
    """
    import yaml

    root = Path(os.environ.get("BRAIN_ROOT", "."))
    config_path = root / "config.yaml"

    existing: dict = {}
    if config_path.exists():
        with open(config_path) as f:
            existing = yaml.safe_load(f) or {}

    _deep_merge(existing, overrides)

    with open(config_path, "w") as f:
        yaml.dump(existing, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    reset_settings()
    return config_path
