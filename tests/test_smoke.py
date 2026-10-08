"""Smoke tests for settings loading and text models."""

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from helix import config
from helix.config import load_settings
from helix.models import ChatModelAdapter, DeterministicTextModel

ENV_VARS = [
    "HELIX_ENV",
    "HELIX_MODEL_PROVIDER",
    "HELIX_MODEL_NAME",
    "HELIX_MODEL_TEMPERATURE",
    "HELIX_LOG_LEVEL",
    "OPENAI_API_KEY",
]


@pytest.fixture
def clean_env(monkeypatch):
    """Clear Helix env vars and stop load_settings from reading the local .env."""
    monkeypatch.setattr(config, "load_dotenv", lambda: None)
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_load_settings_defaults(clean_env):
    settings = load_settings()
    assert settings.model_provider == "deterministic"
    assert settings.model_name == "deterministic-v1"
    assert settings.model_temperature == 0.0
    assert settings.log_level == "INFO"


def test_load_settings_rejects_unknown_provider(clean_env):
    clean_env.setenv("HELIX_MODEL_PROVIDER", "nope")
    with pytest.raises(ValueError, match="'deterministic' or 'openai'"):
        load_settings()


def test_load_settings_openai_requires_key(clean_env):
    clean_env.setenv("HELIX_MODEL_PROVIDER", "openai")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        load_settings()


def test_load_settings_openai_with_key(clean_env):
    clean_env.setenv("HELIX_MODEL_PROVIDER", "OpenAI")
    clean_env.setenv("OPENAI_API_KEY", "test-key")
    assert load_settings().model_provider == "openai"


def test_deterministic_model_collapses_whitespace():
    result = DeterministicTextModel().generate("hello   world\n")
    assert result == "Workspace ready: hello world"


@pytest.mark.parametrize(
    "content",
    ["plain", [{"type": "text", "text": "plain"}]],
)
def test_chat_adapter_returns_text(content):
    adapter = ChatModelAdapter.__new__(ChatModelAdapter)
    adapter._model = GenericFakeChatModel(messages=iter([AIMessage(content=content)]))
    assert adapter.generate("hi") == "plain"
