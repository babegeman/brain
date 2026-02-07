"""Tests for brain.config."""

from __future__ import annotations

import pytest

from brain.config import (
    ChatConfig,
    GraphConfig,
    PromptsConfig,
    Settings,
    _deep_merge,
    get_settings,
    reset_settings,
    save_user_config,
)


def test_settings_loads_default_yaml():
    """config.default.yaml should load into Settings without error."""
    settings = get_settings()
    assert settings.active_profile == "default"
    assert "default" in settings.profiles


def test_active_profile_resolves():
    """The active profile should resolve to an LLMProfile."""
    settings = get_settings()
    profile = settings.llm
    assert "anthropic" in profile.chat_model
    assert "voyage" in profile.embed_model
    assert profile.embed_dim == 1024


def test_neo4j_defaults():
    settings = get_settings()
    assert settings.neo4j.uri == "bolt://localhost:7687"
    assert settings.neo4j.password == "memgraph"


def test_missing_profile_raises(monkeypatch):
    """Requesting a non-existent profile should raise KeyError."""
    reset_settings()
    monkeypatch.setenv("BRAIN_ACTIVE_PROFILE", "nonexistent")
    reset_settings()
    settings = get_settings()
    with pytest.raises(KeyError, match="nonexistent"):
        _ = settings.llm


def test_env_var_override_profile(monkeypatch):
    """BRAIN_ACTIVE_PROFILE env var should override the YAML value."""
    reset_settings()
    monkeypatch.setenv("BRAIN_ACTIVE_PROFILE", "openai")
    reset_settings()
    settings = get_settings()
    assert settings.active_profile == "openai"
    assert "openai" in settings.llm.chat_model


def test_all_profiles_have_required_fields():
    """Every profile in config should have all LLMProfile fields."""
    settings = get_settings()
    for name, profile in settings.profiles.items():
        assert profile.chat_model, f"{name}: missing chat_model"
        assert profile.embed_model, f"{name}: missing embed_model"
        assert profile.embed_dim > 0, f"{name}: invalid embed_dim"
        assert profile.graph_extract_model, f"{name}: missing graph_extract_model"


# --- New config model tests ---


def test_chat_config_defaults():
    cfg = ChatConfig()
    assert cfg.top_k == 5
    assert cfg.graph_results == 10
    assert cfg.snippet_length == 300


def test_graph_config_defaults():
    cfg = GraphConfig()
    assert cfg.group_id == "brain"
    assert cfg.num_results == 10


def test_prompts_config_defaults():
    cfg = PromptsConfig()
    assert "[D1]" in cfg.system_prompt
    assert "[G1]" in cfg.system_prompt_fusion


def test_settings_has_new_config_sections():
    """Settings should include chat, graph, and prompts sections."""
    settings = get_settings()
    assert settings.chat.top_k == 5
    assert settings.graph.group_id == "brain"
    assert "knowledge base" in settings.prompts.system_prompt


def test_deep_merge_flat():
    base = {"a": 1, "b": 2}
    override = {"b": 3, "c": 4}
    result = _deep_merge(base, override)
    assert result == {"a": 1, "b": 3, "c": 4}


def test_deep_merge_nested():
    base = {"x": {"a": 1, "b": 2}, "y": 10}
    override = {"x": {"b": 99}, "z": 42}
    result = _deep_merge(base, override)
    assert result == {"x": {"a": 1, "b": 99}, "y": 10, "z": 42}


def test_save_user_config(tmp_path, monkeypatch):
    """save_user_config should write config.yaml and reset the cache."""
    monkeypatch.setenv("BRAIN_ROOT", str(tmp_path))
    # Copy config.default.yaml so settings can load
    import shutil
    from pathlib import Path

    src = Path(__file__).parent.parent / "config.default.yaml"
    shutil.copy(src, tmp_path / "config.default.yaml")

    reset_settings()
    path = save_user_config({"chat": {"top_k": 42}})
    assert path.exists()

    import yaml

    with open(path) as f:
        data = yaml.safe_load(f)
    assert data["chat"]["top_k"] == 42

    # Settings should reflect the new value
    settings = get_settings()
    assert settings.chat.top_k == 42


def test_save_user_config_merges(tmp_path, monkeypatch):
    """Successive calls should merge, not overwrite."""
    monkeypatch.setenv("BRAIN_ROOT", str(tmp_path))
    import shutil
    from pathlib import Path

    src = Path(__file__).parent.parent / "config.default.yaml"
    shutil.copy(src, tmp_path / "config.default.yaml")

    reset_settings()
    save_user_config({"chat": {"top_k": 7}})
    save_user_config({"graph": {"group_id": "test"}})

    import yaml

    with open(tmp_path / "config.yaml") as f:
        data = yaml.safe_load(f)
    assert data["chat"]["top_k"] == 7
    assert data["graph"]["group_id"] == "test"
