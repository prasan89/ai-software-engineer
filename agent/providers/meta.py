"""Meta Model API adapter.

Verified against Meta Model API documentation retrieved 2026-10-26:
  https://dev.meta.ai/docs/ and https://dev.meta.ai/docs/pricing-rate-limits

Base URL:        https://api.meta.ai/v1
Auth header:     Authorization: Bearer <MODEL_API_KEY>
Endpoint:        POST /v1/chat/completions  (OpenAI-compatible)
Usage response:  usage.prompt_tokens, usage.completion_tokens, usage.cached_tokens
Context window:  1,048,576 tokens (all models below)

Model identifiers (verified 2026-10-26):
    MUSE_WONDER_1_0          = "muse-wonder-1.0"
    MUSE_SPARK_1_3           = "muse-spark-1.3"
    MUSE_SPARK_1_3_CONTRIBUTOR = "muse-spark-1.3-contributor"

Pricing per 1M tokens (verified 2026-10-26):
    muse-wonder-1.0:              cached $0.20 / input $4.00 / output $20.00
    muse-spark-1.3 standard:      cached $0.15 / input $1.25 / output $4.25
    muse-spark-1.3 contributor:   cached $0.002 / input $0.10 / output $0.20

Rate limits (verified 2026-10-26):
    standard:    3,000 RPM / 4,000,000 TPM
    contributor: 150 RPM  / 3,000,000 TPM

UNRESOLVED: The exact request field name to enable/control prompt caching is not
documented in the public docs as of 2026-10-26. The response field ``cached_tokens``
is confirmed. Caching behaviour and request field must be verified before
implementing prefix-cache optimization.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from agent.providers.base import BaseLLMProvider
from agent.providers.types import (
    BudgetExceededError,
    LLMRequest,
    LLMResponse,
    Message,
    ModelCapability,
    ProviderError,
    ProviderInfo,
    RateLimitError,
    UsageStats,
)

# ---------------------------------------------------------------------------
# Verified model identifiers — do not change without re-verifying docs
# ---------------------------------------------------------------------------

MUSE_WONDER_1_0 = "muse-wonder-1.0"
MUSE_SPARK_1_3 = "muse-spark-1.3"
MUSE_SPARK_1_3_CONTRIBUTOR = "muse-spark-1.3-contributor"

# Models suitable for code/SWE tasks (capability routing, see ModelRouter)
CODE_MODELS = frozenset({MUSE_WONDER_1_0, MUSE_SPARK_1_3, MUSE_SPARK_1_3_CONTRIBUTOR})

_CAPABILITIES: dict[str, frozenset[ModelCapability]] = {
    MUSE_WONDER_1_0: frozenset(
        {
            ModelCapability.chat,
            ModelCapability.code_generation,
            ModelCapability.tool_calling,
            ModelCapability.long_context,
        }
    ),
    MUSE_SPARK_1_3: frozenset(
        {
            ModelCapability.chat,
            ModelCapability.code_generation,
            ModelCapability.tool_calling,
            ModelCapability.long_context,
        }
    ),
    MUSE_SPARK_1_3_CONTRIBUTOR: frozenset(
        {
            ModelCapability.chat,
            ModelCapability.code_generation,
            ModelCapability.long_context,
        }
    ),
}

# ---------------------------------------------------------------------------
# Pricing (USD per 1M tokens, verified 2026-10-26)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Pricing:
    cached_input_per_m: float
    input_per_m: float
    output_per_m: float


_MODEL_PRICING: dict[str, _Pricing] = {
    MUSE_WONDER_1_0: _Pricing(0.20, 4.00, 20.00),
    MUSE_SPARK_1_3: _Pricing(0.15, 1.25, 4.25),
    MUSE_SPARK_1_3_CONTRIBUTOR: _Pricing(0.002, 0.10, 0.20),
}

_BASE_URL = "https://api.meta.ai/v1"
_CHAT_PATH = "/chat/completions"

_META_PROVIDER_INFO = ProviderInfo(
    name="meta",
    base_url=_BASE_URL,
    auth_header="Authorization",
    docs_url="https://dev.meta.ai/docs/",
)


class MetaProvider(BaseLLMProvider):
    """Adapter for Meta Model API (OpenAI-compatible chat completions).

    Args:
        model: One of the verified model identifier constants above.
        api_key: Meta MODEL_API_KEY. Reads META_API_KEY env var by default.
        budget_usd: Hard spend cap. Raises BudgetExceededError when exceeded.
        http_client: Injected for testing; a new client is created otherwise.
    """

    def __init__(
        self,
        model: str = MUSE_SPARK_1_3,
        api_key: str | None = None,
        budget_usd: float | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("META_API_KEY", "")
        if not resolved_key:
            raise ProviderError(
                "META_API_KEY is not set. Add it to your .env file "
                "(see .env.example). Never commit a real key."
            )
        self._model = model
        self._api_key = resolved_key
        self._budget_usd = budget_usd
        self._spent_usd = 0.0
        self._http_client = http_client
        self._owns_client = http_client is None

    # ------------------------------------------------------------------
    # BaseLLMProvider interface
    # ------------------------------------------------------------------

    @property
    def info(self) -> ProviderInfo:
        return _META_PROVIDER_INFO

    def model_id(self) -> str:
        return self._model

    def supports(self, capability: ModelCapability) -> bool:
        return capability in _CAPABILITIES.get(self._model, frozenset())

    def estimated_cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        pricing = _MODEL_PRICING.get(self._model)
        if pricing is None:
            return 0.0
        return (
            prompt_tokens / 1_000_000 * pricing.input_per_m
            + completion_tokens / 1_000_000 * pricing.output_per_m
        )

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def chat(self, request: LLMRequest) -> LLMResponse:
        self._check_budget(request)
        model = request.model_override or self._model
        payload = _build_payload(model, request)

        raw = await self._post(_CHAT_PATH, payload)
        response = _parse_response(raw)

        cost = self.estimated_cost_usd(
            response.usage.prompt_tokens, response.usage.completion_tokens
        )
        self._spent_usd += cost
        return response

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_budget(self, _request: LLMRequest) -> None:
        if self._budget_usd is not None and self._spent_usd >= self._budget_usd:
            raise BudgetExceededError(
                f"Budget cap of ${self._budget_usd:.4f} USD reached "
                f"(spent ${self._spent_usd:.4f}). "
                "Increase PROVIDER_BUDGET_USD or set FALLBACK_PROVIDER_ENABLED=true "
                "with an explicit FALLBACK_BUDGET_USD to allow fallback."
            )

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        client = self._http_client or httpx.AsyncClient(base_url=_BASE_URL, timeout=120)
        try:
            resp = await client.post(
                path,
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        finally:
            if self._owns_client and self._http_client is None:
                await client.aclose()

        if resp.status_code == 429:
            raise RateLimitError("Meta API rate limit hit", status_code=429)
        if resp.status_code >= 400:
            raise ProviderError(
                f"Meta API error {resp.status_code}: {resp.text[:200]}",
                status_code=resp.status_code,
            )
        return resp.json()  # type: ignore[no-any-return]

    @property
    def spent_usd(self) -> float:
        return self._spent_usd


# ---------------------------------------------------------------------------
# Pure helpers (no side effects, easy to unit test)
# ---------------------------------------------------------------------------


def _build_payload(model: str, request: LLMRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [_message_to_dict(m) for m in request.messages],
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
    }
    if request.stop:
        payload["stop"] = request.stop
    # Prompt caching: optional routing hint — caching is automatic per Meta API docs.
    # Verified fields: prompt_cache_key, prompt_cache_retention (2026-10-26)
    if request.prompt_cache_key is not None:
        payload["prompt_cache_key"] = request.prompt_cache_key
    if request.prompt_cache_retention is not None:
        payload["prompt_cache_retention"] = request.prompt_cache_retention
    payload.update(request.extra)
    return payload


def _message_to_dict(message: Message) -> dict[str, str]:
    return {"role": message.role.value, "content": message.content}


def _parse_response(raw: dict[str, Any]) -> LLMResponse:
    choice = raw["choices"][0]
    usage_raw = raw.get("usage", {})
    # cached_tokens is nested under usage.prompt_tokens_details.cached_tokens
    # Verified: Meta API docs/cookbook/prompt-caching, retrieved 2026-10-26
    details = usage_raw.get("prompt_tokens_details", {})
    cached = details.get("cached_tokens", 0)
    usage = UsageStats(
        prompt_tokens=usage_raw.get("prompt_tokens", 0),
        completion_tokens=usage_raw.get("completion_tokens", 0),
        cached_tokens=cached,
    )
    return LLMResponse(
        content=choice["message"]["content"],
        model=raw.get("model", ""),
        usage=usage,
        finish_reason=choice.get("finish_reason", "stop"),
        raw=raw,
    )
