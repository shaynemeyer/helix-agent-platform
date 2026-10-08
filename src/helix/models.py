"""Provider-neutral text model interface and implementations."""

from __future__ import annotations

from typing import Protocol

from helix.config import AppSettings


class TextModel(Protocol):
    """Small interface used by the graph instead of a provider-specific class."""

    def generate(self, prompt: str) -> str:
        """Return text for a prompt."""


class DeterministicTextModel:
    """Predictable local substitute used by tests and smoke checks."""

    def generate(self, prompt: str) -> str:
        normalized = " ".join(prompt.split())
        return f"Workspace ready: {normalized}"


class ChatModelAdapter:
    """Adapter around a LangChain chat-model integration."""

    def __init__(self, settings: AppSettings) -> None:
        from langchain.chat_models import init_chat_model

        self._model = init_chat_model(
            settings.model_name,
            model_provider=settings.model_provider,
            temperature=settings.model_temperature,
        )

    def generate(self, prompt: str) -> str:
        return self._model.invoke(prompt).text


def build_text_model(settings: AppSettings) -> TextModel:
    """Construct the configured text model without changing graph code."""
    if settings.model_provider == "deterministic":
        return DeterministicTextModel()
    return ChatModelAdapter(settings)
