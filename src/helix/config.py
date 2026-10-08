"""Environment-backed application settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, cast

from dotenv import load_dotenv

ProviderName = Literal["deterministic", "openai"]


@dataclass(frozen=True, slots=True)
class AppSettings:
    environment: str
    model_provider: ProviderName
    model_name: str
    model_temperature: float
    log_level: str


def load_settings() -> AppSettings:
    """Load and validate settings from .env and the process environment."""
    load_dotenv()

    provider_value = os.getenv("HELIX_MODEL_PROVIDER", "deterministic").lower()
    if provider_value not in ["deterministic", "openai"]:
        raise ValueError("HELIX_MODEL_PROVIDER must be 'deterministic' or 'openai'.")

    provider = cast(ProviderName, provider_value)
    settings = AppSettings(
        environment=os.getenv("HELIX_ENV", "development"),
        model_provider=provider,
        model_name=os.getenv("HELIX_MODEL_NAME", "deterministic-v1"),
        model_temperature=float(os.getenv("HELIX_MODEL_TEMPERATURE", "0")),
        log_level=os.getenv("HELIX_LOG_LEVEL", "INFO").upper(),
    )

    if settings.model_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required when HELIX_MODEL_PROVIDER=openai.")
    return settings
