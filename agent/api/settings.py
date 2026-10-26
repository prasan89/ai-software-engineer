"""Application settings loaded from environment variables.

All secrets are read from env vars. Never hard-code a key here.
See .env.example for the full list of supported variables.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Meta API
    meta_api_key: str = Field(default="", alias="META_API_KEY")
    meta_default_model: str = Field(default="muse-spark-1.3", alias="META_DEFAULT_MODEL")

    # Budget controls
    provider_budget_usd: float = Field(default=5.0, alias="PROVIDER_BUDGET_USD")
    fallback_provider_enabled: bool = Field(default=False, alias="FALLBACK_PROVIDER_ENABLED")
    fallback_budget_usd: float = Field(default=0.0, alias="FALLBACK_BUDGET_USD")

    # GitHub
    github_token: str = Field(default="", alias="GITHUB_TOKEN")

    # Server
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    log_level: str = Field(default="info", alias="LOG_LEVEL")

    # Cache
    cache_max_age_seconds: float = Field(default=86_400, alias="CACHE_MAX_AGE_SECONDS")


def get_settings() -> Settings:
    return Settings()
