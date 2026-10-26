"""Unit tests for CacheMetrics."""

from __future__ import annotations

import pytest

from agent.cost_optimizer.metrics import CacheMetrics

pytestmark = pytest.mark.unit


class TestCacheMetrics:
    def test_initial_state(self) -> None:
        m = CacheMetrics()
        assert m.cache_hits == 0
        assert m.cache_hit_rate == 0.0
        assert m.total_tokens_saved == 0
        assert m.total_cost_saved_usd == 0.0

    def test_cache_hit_rate_calculation(self) -> None:
        m = CacheMetrics()
        m.record_cache_hit()
        m.record_cache_hit()
        m.record_cache_miss()
        assert m.cache_hit_rate == pytest.approx(2 / 3)

    def test_tokens_and_cost_accumulate(self) -> None:
        m = CacheMetrics()
        m.record_cache_hit(tokens_saved=1000, cost_saved_usd=0.01)
        m.record_fix_reuse(tokens_saved=5000, cost_saved_usd=0.05)
        assert m.total_tokens_saved == 6000
        assert m.total_cost_saved_usd == pytest.approx(0.06)

    def test_report_keys(self) -> None:
        report = CacheMetrics().report()
        expected = {
            "cache_hit_rate",
            "cache_hits",
            "cache_misses",
            "verified_fix_reuses",
            "tokens_saved_total",
            "cost_saved_usd_total",
            "checkpoint_resumptions",
            "stages_skipped_by_checkpoint",
        }
        assert set(report.keys()) == expected

    def test_checkpoint_resumption(self) -> None:
        m = CacheMetrics()
        m.record_checkpoint_resumption(stages_skipped=3)
        assert m.checkpoint_resumptions == 1
        assert m.stages_skipped_by_checkpoint == 3
