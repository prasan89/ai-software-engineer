"""Unit tests for provider types and Meta adapter pure helpers."""

from __future__ import annotations

import pytest

from agent.providers.meta import (
    MUSE_SPARK_1_3,
    MUSE_SPARK_1_3_CONTRIBUTOR,
    MUSE_WONDER_1_0,
    _build_payload,
    _parse_response,
)
from agent.providers.types import (
    LLMRequest,
    Message,
    Role,
)

pytestmark = pytest.mark.unit


def _request(model_override: str | None = None) -> LLMRequest:
    return LLMRequest(
        model_override=model_override,
        messages=[Message(role=Role.user, content="Fix the bug.")],
    )


class TestBuildPayload:
    def test_model_from_request(self) -> None:
        payload = _build_payload(MUSE_SPARK_1_3, _request())
        assert payload["model"] == MUSE_SPARK_1_3

    def test_messages_serialized(self) -> None:
        payload = _build_payload(MUSE_SPARK_1_3, _request())
        assert payload["messages"] == [{"role": "user", "content": "Fix the bug."}]

    def test_stop_omitted_when_none(self) -> None:
        payload = _build_payload(MUSE_SPARK_1_3, _request())
        assert "stop" not in payload

    def test_stop_included_when_set(self) -> None:
        req = LLMRequest(
            messages=[Message(role=Role.user, content="q")],
            stop=["<END>"],
        )
        payload = _build_payload(MUSE_SPARK_1_3, req)
        assert payload["stop"] == ["<END>"]


class TestParseResponse:
    def _raw(self, content: str = "ok", cached: int = 0) -> dict:
        return {
            "id": "run-1",
            "model": MUSE_SPARK_1_3,
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                # Verified nested path: usage.prompt_tokens_details.cached_tokens
                # (Meta API docs/cookbook/prompt-caching, 2026-10-26)
                "prompt_tokens_details": {"cached_tokens": cached},
            },
        }

    def test_content_extracted(self) -> None:
        resp = _parse_response(self._raw("hello"))
        assert resp.content == "hello"

    def test_usage_populated(self) -> None:
        resp = _parse_response(self._raw(cached=30))
        assert resp.usage.prompt_tokens == 100
        assert resp.usage.completion_tokens == 50
        assert resp.usage.cached_tokens == 30

    def test_total_tokens(self) -> None:
        resp = _parse_response(self._raw())
        assert resp.usage.total_tokens == 150

    def test_missing_usage_defaults_to_zero(self) -> None:
        raw = {
            "model": MUSE_SPARK_1_3,
            "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
        }
        resp = _parse_response(raw)
        assert resp.usage.prompt_tokens == 0
        assert resp.usage.cached_tokens == 0

    def test_prompt_cache_key_included_in_payload(self) -> None:
        req = LLMRequest(
            messages=[Message(role=Role.user, content="q")],
            prompt_cache_key="stable-system-prompt-v1",
            prompt_cache_retention="24h",
        )
        payload = _build_payload(MUSE_SPARK_1_3, req)
        assert payload["prompt_cache_key"] == "stable-system-prompt-v1"
        assert payload["prompt_cache_retention"] == "24h"

    def test_prompt_cache_key_omitted_when_none(self) -> None:
        payload = _build_payload(MUSE_SPARK_1_3, _request())
        assert "prompt_cache_key" not in payload
        assert "prompt_cache_retention" not in payload


class TestEstimatedCost:
    def test_wonder_cost(self) -> None:
        import os

        from agent.providers.meta import MetaProvider

        os.environ["META_API_KEY"] = "test-key"
        p = MetaProvider(model=MUSE_WONDER_1_0)
        cost = p.estimated_cost_usd(1_000_000, 1_000_000)
        assert cost == pytest.approx(4.00 + 20.00)

    def test_contributor_is_cheapest(self) -> None:
        from agent.providers.meta import MetaProvider

        p_wonder = MetaProvider(model=MUSE_WONDER_1_0)
        p_contrib = MetaProvider(model=MUSE_SPARK_1_3_CONTRIBUTOR)
        tokens = 100_000
        assert p_contrib.estimated_cost_usd(tokens, tokens) < p_wonder.estimated_cost_usd(
            tokens, tokens
        )
