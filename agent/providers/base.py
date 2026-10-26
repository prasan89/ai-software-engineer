"""Provider-neutral LLM interface.

All provider adapters must implement BaseLLMProvider. No workflow code
imports a concrete adapter directly; it receives an instance via dependency
injection.
"""

from __future__ import annotations

import abc
from typing import Any

from agent.providers.types import (
    LLMRequest,
    LLMResponse,
    ModelCapability,
    ProviderInfo,
)


class BaseLLMProvider(abc.ABC):
    """Minimal contract every LLM provider adapter must satisfy."""

    @property
    @abc.abstractmethod
    def info(self) -> ProviderInfo:
        """Static metadata about this provider."""

    @abc.abstractmethod
    async def chat(self, request: LLMRequest) -> LLMResponse:
        """Send a chat-completions request and return a typed response.

        Raises:
            ProviderError: on non-retryable API errors.
            RateLimitError: when the provider signals rate limiting.
            BudgetExceededError: when the configured budget cap is reached.
        """

    @abc.abstractmethod
    def supports(self, capability: ModelCapability) -> bool:
        """Return True if the current model supports the given capability."""

    @abc.abstractmethod
    def estimated_cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Best-effort cost estimate for a call of the given token counts."""

    @abc.abstractmethod
    def model_id(self) -> str:
        """Return the model identifier string used in API requests."""

    # Optional: providers that support tool/function calling override this.
    async def chat_with_tools(
        self,
        request: LLMRequest,
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        raise NotImplementedError(f"{self.__class__.__name__} does not support tool calling")
