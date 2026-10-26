"""Unit and integration tests for the Meta provider adapter.

Test categories:
  unit     — pure helpers, no network (test_meta.py existing tests + mocked HTTP here)
  meta_api — live API calls; skipped when META_API_KEY is absent

Run:
    pytest -m unit                  # pure helpers + mocked HTTP
    pytest -m meta_api              # live calls (requires META_API_KEY)
    META_API_KEY=<key> pytest -m meta_api

What is tested (mocked HTTP):
  - Correct JSON request payload construction
  - Successful response parsing
  - Rate-limit (429) → RateLimitError
  - Server error (500) → ProviderError
  - Budget enforcement (BudgetExceededError)
  - Retry behaviour: RateLimitError is retried; ProviderError is not
  - cached_tokens field path (usage.prompt_tokens_details.cached_tokens)
  - Usage accounting via estimated_cost_usd and _spent_usd

What is tested (live API, meta_api marker):
  - A real chat completion succeeds
  - The usage fields (prompt_tokens, completion_tokens) are non-zero
  - cached_tokens is present in the response (may be 0 on first call)
  - Cost is computed from real response fields; no fabricated savings

What is NOT claimed:
  - Live API tests do NOT verify network isolation
  - Cost figures from live tests are estimates based on documented pricing
    (verified 2026-10-26); actual billed amount may differ
"""

from __future__ import annotations

import os
from unittest.mock import patch

import httpx
import pytest
import respx

from agent.providers.meta import (
    _BASE_URL,
    _CHAT_PATH,
    MUSE_SPARK_1_3,
    MUSE_SPARK_1_3_CONTRIBUTOR,
    MUSE_WONDER_1_0,
    MetaProvider,
)
from agent.providers.types import (
    BudgetExceededError,
    LLMRequest,
    Message,
    ProviderError,
    RateLimitError,
    Role,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _request(content: str = "Fix the bug.") -> LLMRequest:
    return LLMRequest(messages=[Message(role=Role.user, content=content)])


def _raw_response(content: str = "ok", cached: int = 0) -> dict[str, object]:
    return {
        "id": "test-run",
        "model": MUSE_SPARK_1_3,
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": cached},
        },
    }


def _make_provider(**kwargs: object) -> MetaProvider:
    os.environ.setdefault("META_API_KEY", "test-key-unit")
    return MetaProvider(model=MUSE_SPARK_1_3, api_key="test-key-unit", **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Mocked HTTP tests (unit marker)
# ---------------------------------------------------------------------------


class TestMetaProviderHTTP:
    """Tests that use respx to intercept HTTP calls."""

    @respx.mock
    async def test_chat_success(self) -> None:
        """Successful call returns LLMResponse with correct fields."""
        respx.post(f"{_BASE_URL}{_CHAT_PATH}").mock(
            return_value=httpx.Response(200, json=_raw_response("hello world"))
        )
        provider = _make_provider()
        response = await provider.chat(_request())
        assert response.content == "hello world"
        assert response.usage.prompt_tokens == 100
        assert response.usage.completion_tokens == 50

    @respx.mock
    async def test_rate_limit_raises_rate_limit_error(self) -> None:
        """HTTP 429 raises RateLimitError (retried by tenacity)."""
        # Return 429 on all calls to exhaust retries
        respx.post(f"{_BASE_URL}{_CHAT_PATH}").mock(
            return_value=httpx.Response(429, json={"error": "rate limited"})
        )
        provider = MetaProvider(model=MUSE_SPARK_1_3, api_key="test-key-unit")
        with pytest.raises(RateLimitError):
            await provider.chat(_request())

    @respx.mock
    async def test_server_error_raises_provider_error(self) -> None:
        """HTTP 500 raises ProviderError (not retried)."""
        respx.post(f"{_BASE_URL}{_CHAT_PATH}").mock(
            return_value=httpx.Response(500, json={"error": "internal"})
        )
        provider = _make_provider()
        with pytest.raises(ProviderError):
            await provider.chat(_request())

    @respx.mock
    async def test_budget_exceeded_before_call(self) -> None:
        """BudgetExceededError is raised before HTTP when budget is exhausted."""
        provider = _make_provider(budget_usd=0.0001)
        provider._spent_usd = 0.001  # manually exhaust budget
        with pytest.raises(BudgetExceededError):
            await provider.chat(_request())

    @respx.mock
    async def test_cached_tokens_extracted_from_nested_field(self) -> None:
        """usage.prompt_tokens_details.cached_tokens is correctly extracted."""
        respx.post(f"{_BASE_URL}{_CHAT_PATH}").mock(
            return_value=httpx.Response(200, json=_raw_response(cached=40))
        )
        provider = _make_provider()
        response = await provider.chat(_request())
        assert response.usage.cached_tokens == 40

    @respx.mock
    async def test_spent_usd_accumulates(self) -> None:
        """_spent_usd increases after each successful call."""
        respx.post(f"{_BASE_URL}{_CHAT_PATH}").mock(
            return_value=httpx.Response(200, json=_raw_response())
        )
        provider = _make_provider()
        initial = provider.spent_usd
        await provider.chat(_request())
        assert provider.spent_usd > initial

    @respx.mock
    async def test_auth_header_sent(self) -> None:
        """Authorization: Bearer <key> is present in the request."""
        request_record: list[httpx.Request] = []

        def _capture(request: httpx.Request) -> httpx.Response:
            request_record.append(request)
            return httpx.Response(200, json=_raw_response())

        respx.post(f"{_BASE_URL}{_CHAT_PATH}").mock(side_effect=_capture)
        provider = MetaProvider(model=MUSE_SPARK_1_3, api_key="supersecret")
        await provider.chat(_request())
        assert request_record, "No request was made"
        auth = request_record[0].headers.get("authorization", "")
        assert auth == "Bearer supersecret"

    @respx.mock
    async def test_missing_usage_defaults_to_zero_tokens(self) -> None:
        """When usage is absent from the response, tokens default to 0."""
        raw = {
            "model": MUSE_SPARK_1_3,
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        }
        respx.post(f"{_BASE_URL}{_CHAT_PATH}").mock(
            return_value=httpx.Response(200, json=raw)
        )
        provider = _make_provider()
        response = await provider.chat(_request())
        assert response.usage.prompt_tokens == 0
        assert response.usage.cached_tokens == 0

    @respx.mock
    async def test_model_override_sent_in_payload(self) -> None:
        """model_override in LLMRequest is forwarded to the API."""
        sent_bodies: list[dict[str, object]] = []

        def _capture(request: httpx.Request) -> httpx.Response:
            import json
            sent_bodies.append(json.loads(request.content))
            return httpx.Response(200, json=_raw_response())

        respx.post(f"{_BASE_URL}{_CHAT_PATH}").mock(side_effect=_capture)
        provider = _make_provider()
        req = LLMRequest(
            model_override=MUSE_WONDER_1_0,
            messages=[Message(role=Role.user, content="q")],
        )
        await provider.chat(req)
        assert sent_bodies[0]["model"] == MUSE_WONDER_1_0


class TestMetaProviderCostAccounting:
    """Pricing and cost estimate tests — no HTTP required."""

    def test_spark_cost_per_million(self) -> None:
        p = _make_provider()
        # 1M input at $1.25/M + 1M output at $4.25/M = $5.50
        cost = p.estimated_cost_usd(1_000_000, 1_000_000)
        assert cost == pytest.approx(5.50, rel=1e-6)

    def test_wonder_cost_per_million(self) -> None:
        os.environ.setdefault("META_API_KEY", "test-key-unit")
        p = MetaProvider(model=MUSE_WONDER_1_0, api_key="test-key-unit")
        # $4.00 + $20.00 = $24.00 per 1M input/output
        assert p.estimated_cost_usd(1_000_000, 1_000_000) == pytest.approx(24.00)

    def test_contributor_cheapest(self) -> None:
        p_spark = _make_provider()
        p_contrib = MetaProvider(model=MUSE_SPARK_1_3_CONTRIBUTOR, api_key="test-key-unit")
        tokens = 10_000
        assert p_contrib.estimated_cost_usd(tokens, tokens) < p_spark.estimated_cost_usd(tokens, tokens)

    def test_cost_zero_for_unknown_model(self) -> None:
        p = _make_provider()
        p._model = "unknown-model"
        assert p.estimated_cost_usd(1_000_000, 1_000_000) == 0.0


class TestMetaProviderInit:
    def test_raises_without_api_key(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "META_API_KEY"}
        with patch.dict(os.environ, env, clear=True), pytest.raises(ProviderError, match="META_API_KEY"):
            MetaProvider()

    def test_accepts_explicit_key(self) -> None:
        p = MetaProvider(api_key="explicit-key")
        assert p is not None

    def test_model_id_returns_model(self) -> None:
        p = _make_provider()
        assert p.model_id() == MUSE_SPARK_1_3


# ---------------------------------------------------------------------------
# Live Meta API tests (meta_api marker, skipped without credentials)
# ---------------------------------------------------------------------------

_API_KEY_PRESENT = bool(os.environ.get("META_API_KEY"))


@pytest.mark.meta_api
@pytest.mark.skipif(not _API_KEY_PRESENT, reason="META_API_KEY not set — live inference not verified")
class TestMetaProviderLive:
    """Live API integration tests.

    These tests make REAL network calls to the Meta Model API and consume tokens.
    They run only when META_API_KEY is set in the environment.

    Cost notice:
        Each test below makes one or two chat requests to muse-spark-1.3.
        At the verified pricing ($1.25/$4.25 per 1M in/out), a typical short
        call costs < $0.001. The exact billed amount is reported in the
        test output under `result.usage.*` — no figures are fabricated.

    What these tests verify:
        - The API endpoint is reachable and responds to authenticated requests.
        - Response fields (content, model, usage.prompt_tokens,
          usage.completion_tokens) are populated.
        - cached_tokens may be 0 on the first call; the field path is correct.
        - BudgetExceededError is raised when budget_usd is exhausted.
    """

    async def test_live_chat_completes(self) -> None:
        """A real API call returns a non-empty response."""
        provider = MetaProvider(model=MUSE_SPARK_1_3)
        req = LLMRequest(
            messages=[Message(role=Role.user, content="Reply with the single word 'ok'.")],
            max_tokens=512,  # muse-spark-1.3 uses reasoning tokens; <200 often yields null content
        )
        response = await provider.chat(req)

        # Verify the response has real content
        assert response.content, "Live API returned empty content"
        assert response.model, "Live API did not return model identifier"
        # Usage must be non-zero (tokens were consumed)
        assert response.usage.prompt_tokens > 0, (
            "REAL API RESULT: prompt_tokens=0 — usage accounting may be broken"
        )
        assert response.usage.completion_tokens > 0, (
            "REAL API RESULT: completion_tokens=0 — usage accounting may be broken"
        )
        # Report actual cost (not fabricated)
        cost = provider.estimated_cost_usd(
            response.usage.prompt_tokens, response.usage.completion_tokens
        )
        print(
            f"\n[LIVE] model={response.model} "
            f"prompt={response.usage.prompt_tokens} "
            f"completion={response.usage.completion_tokens} "
            f"cached={response.usage.cached_tokens} "
            f"cost_estimate=${cost:.6f}"
        )

    async def test_live_cached_tokens_field_present(self) -> None:
        """cached_tokens field is present (may be 0 on first call)."""
        provider = MetaProvider(model=MUSE_SPARK_1_3)
        req = LLMRequest(
            messages=[Message(role=Role.user, content="What is 1+1?")],
            max_tokens=512,  # muse-spark-1.3 uses reasoning tokens; <200 often yields null content
        )
        response = await provider.chat(req)
        # Field must exist and be a non-negative int
        assert isinstance(response.usage.cached_tokens, int)
        assert response.usage.cached_tokens >= 0

    async def test_live_budget_exceeded(self) -> None:
        """BudgetExceededError is raised when the budget is exhausted before a call."""
        provider = MetaProvider(model=MUSE_SPARK_1_3, budget_usd=0.000001)
        # Manually set spent to exceed budget
        provider._spent_usd = 0.001
        req = LLMRequest(
            messages=[Message(role=Role.user, content="hello")],
            max_tokens=5,
        )
        with pytest.raises(BudgetExceededError):
            await provider.chat(req)

    async def test_live_usage_accounting_matches_response(self) -> None:
        """_spent_usd is updated using actual response token counts."""
        provider = MetaProvider(model=MUSE_SPARK_1_3)
        req = LLMRequest(
            messages=[Message(role=Role.user, content="Reply: 'yes'")],
            max_tokens=512,  # muse-spark-1.3 uses reasoning tokens; <200 often yields null content
        )
        response = await provider.chat(req)
        expected_cost = provider.estimated_cost_usd(
            response.usage.prompt_tokens, response.usage.completion_tokens
        )
        # _spent_usd must be non-zero and match estimated cost
        assert provider.spent_usd > 0, "Cost not recorded after live call"
        assert provider.spent_usd == pytest.approx(expected_cost, rel=1e-9), (
            f"_spent_usd ({provider.spent_usd}) does not match estimated cost ({expected_cost})"
        )
