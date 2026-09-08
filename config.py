"""
config.py
---------
Central configuration loader for RIME.

All settings are read from environment variables (populated from .env by
python-dotenv). No module other than this file and ExternalServiceManager
should ever read environment variables directly, to keep secrets contained.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load .env from the project root (one level up from this file if installed
# as a package; same directory when run directly).
_env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_env_path, override=False)


class RimeSettings(BaseSettings):
    """Rime TTS integration — all values must be pinned at submission time."""

    api_key: str = Field(default="", alias="RIME_API_KEY")
    model_id: str = Field(default="mist", alias="RIME_MODEL_ID")
    speaker: str = Field(default="lagoon", alias="RIME_SPEAKER")
    language: str = Field(default="en-US", alias="RIME_LANGUAGE")
    audio_format: str = Field(default="pcm_16000", alias="RIME_AUDIO_FORMAT")
    endpoint: str = Field(
        default="wss://users.rime.ai/v1/rime-tts", alias="RIME_ENDPOINT"
    )
    http_endpoint: str = Field(
        default="https://users.rime.ai/v1/rime-tts", alias="RIME_HTTP_ENDPOINT"
    )

    model_config = {"env_file": ".env", "extra": "ignore", "populate_by_name": True, "protected_namespaces": ()}


class LLMSettings(BaseSettings):
    """Local-first LLM + optional external fallback."""

    ollama_base_url: str = Field(
        default="http://localhost:11434", alias="OLLAMA_BASE_URL"
    )
    ollama_model: str = Field(default="gemma:1b", alias="OLLAMA_MODEL")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")

    model_config = {"env_file": ".env", "extra": "ignore", "populate_by_name": True, "protected_namespaces": ()}


class SpeakerSettings(BaseSettings):
    """Speaker identification thresholds (TRD §3.3)."""

    confident_threshold: float = Field(
        default=0.80, alias="SPEAKER_MATCH_CONFIDENT_THRESHOLD"
    )
    tentative_threshold: float = Field(
        default=0.60, alias="SPEAKER_MATCH_TENTATIVE_THRESHOLD"
    )

    model_config = {"env_file": ".env", "extra": "ignore", "populate_by_name": True, "protected_namespaces": ()}


class STTSettings(BaseSettings):
    """Whisper local STT configuration."""

    model_size: str = Field(default="base", alias="WHISPER_MODEL_SIZE")

    model_config = {"env_file": ".env", "extra": "ignore", "populate_by_name": True, "protected_namespaces": ()}


class SafetySettings(BaseSettings):
    """Emergency detection and feature flags."""

    emergency_confidence_threshold: float = Field(
        default=0.85, alias="EMERGENCY_CONFIDENCE_THRESHOLD"
    )
    enable_privacy_mode: bool = Field(default=True, alias="ENABLE_PRIVACY_MODE")
    enable_emergency_detection: bool = Field(
        default=True, alias="ENABLE_EMERGENCY_DETECTION"
    )
    enable_proactive_tasks: bool = Field(default=False, alias="ENABLE_PROACTIVE_TASKS")

    model_config = {"env_file": ".env", "extra": "ignore", "populate_by_name": True, "protected_namespaces": ()}


class ExternalAPISettings(BaseSettings):
    """Third-party live-data services — only read via ExternalServiceManager."""

    weather_api_key: str = Field(default="", alias="WEATHER_API_KEY")
    weather_base_url: str = Field(
        default="https://api.openweathermap.org/data/2.5",
        alias="WEATHER_BASE_URL",
    )
    transport_api_key: str = Field(default="", alias="TRANSPORT_API_KEY")
    transport_base_url: str = Field(
        default="https://api.example-transit.com/v1", alias="TRANSPORT_BASE_URL"
    )

    model_config = {"env_file": ".env", "extra": "ignore", "populate_by_name": True, "protected_namespaces": ()}


class AppSettings(BaseSettings):
    """Top-level application settings."""

    host: str = Field(default="0.0.0.0", alias="APP_HOST")
    port: int = Field(default=8080, alias="APP_PORT")
    secret_key: str = Field(default="change_me", alias="SECRET_KEY")
    database_path: str = Field(default="./rime.sqlite", alias="DATABASE_PATH")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    model_config = {"env_file": ".env", "extra": "ignore", "populate_by_name": True, "protected_namespaces": ()}


class Settings:
    """Aggregate settings object — the single point of access for all config."""

    def __init__(self) -> None:
        self.app = AppSettings()
        self.rime = RimeSettings()
        self.llm = LLMSettings()
        self.speaker = SpeakerSettings()
        self.stt = STTSettings()
        self.safety = SafetySettings()
        self.external = ExternalAPISettings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached singleton settings object."""
    return Settings()
