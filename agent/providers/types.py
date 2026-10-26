"""Shared Pydantic types for the provider abstraction layer."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Role(StrEnum):
    system = "system"
    user = "user"
    assistant = "assistant"


class Message(BaseModel):
    role: Role
    content: str


class ModelCapability(StrEnum):
    chat = "chat"
    tool_calling = "tool_calling"
    code_generation = "code_generation"
    long_context = "long_context"  # >128k tokens


class LLMRequest(BaseModel):
    model_override: str | None = None
    messages: list[Message]
    max_tokens: int = 4096
    temperature: float = 0.2
    stop: list[str] | None = None
    # Prompt caching (Meta API, verified 2026-10-26 via docs/cookbook/prompt-caching)
    # prompt_cache_key: optional stable routing hint — same key → same backend → higher hit rate.
    # Caching is AUTOMATIC; this field improves hit rate, it does not enable/disable caching.
    # prompt_cache_retention: "in_memory" (default) or "24h"
    prompt_cache_key: str | None = None
    prompt_cache_retention: str | None = None  # "in_memory" | "24h"
    extra: dict[str, Any] = Field(default_factory=dict)


class UsageStats(BaseModel):
    """Token usage as reported by the provider.

    Fields:
        prompt_tokens: input tokens billed.
        completion_tokens: output tokens billed.
        cached_tokens: prompt tokens served from provider-side cache.
            Verified field path: usage.prompt_tokens_details.cached_tokens
            (Meta Model API docs/cookbook/prompt-caching, retrieved 2026-10-26).
            Value is 0 when the provider does not report it.
    """

    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMResponse(BaseModel):
    content: str
    model: str
    usage: UsageStats
    finish_reason: str = "stop"
    raw: dict[str, Any] = Field(default_factory=dict, exclude=True)


class ProviderInfo(BaseModel):
    name: str
    base_url: str
    auth_header: str
    docs_url: str


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ProviderError(Exception):
    """Non-retryable provider error (4xx that is not rate-limit)."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class RateLimitError(ProviderError):
    """Provider returned a rate-limit response (429)."""


class BudgetExceededError(Exception):
    """Configured budget cap reached; refusing to make further calls."""
