"""Unit tests for CacheInvalidator."""

from __future__ import annotations

import time

import pytest

from agent.engineering_knowledge.cache import (
    AGENT_VERSION,
    CacheInvalidator,
    CacheKey,
    CacheStage,
    InvalidationReason,
)

pytestmark = pytest.mark.unit

MODEL = "muse-spark-1.3"
FP = "a" * 64
TREE = "b" * 64


def _key(stage: CacheStage = CacheStage.root_cause, tree: str = TREE) -> CacheKey:
    return CacheKey(
        fingerprint_hex=FP,
        tree_hash=tree,
        stage=stage,
        model_id=MODEL,
        agent_version=AGENT_VERSION,
    )


class TestCacheHit:
    def test_store_and_hit(self) -> None:
        cache = CacheInvalidator()
        key = _key()
        cache.store(key, {"analysis": "result"})
        result = cache.lookup(key)
        assert result.hit
        assert result.entry is not None
        assert result.entry.value == {"analysis": "result"}

    def test_hit_increments_hit_count(self) -> None:
        cache = CacheInvalidator()
        key = _key()
        cache.store(key, "v")
        cache.lookup(key)
        cache.lookup(key)
        assert cache.lookup(key).entry.hit_count == 3  # type: ignore[union-attr]


class TestCacheMiss:
    def test_missing_key(self) -> None:
        cache = CacheInvalidator()
        result = cache.lookup(_key())
        assert not result.hit
        assert result.reason == InvalidationReason.not_found

    def test_different_stage_is_miss(self) -> None:
        cache = CacheInvalidator()
        cache.store(_key(CacheStage.root_cause), "rc")
        result = cache.lookup(_key(CacheStage.patch))
        assert not result.hit

    def test_different_tree_hash_is_miss(self) -> None:
        cache = CacheInvalidator()
        cache.store(_key(tree=TREE), "v")
        result = cache.lookup(_key(tree="c" * 64))
        assert not result.hit


class TestExpiry:
    def test_expired_entry_is_evicted(self) -> None:
        cache = CacheInvalidator(max_age_seconds=0.01)
        key = _key()
        cache.store(key, "v")
        time.sleep(0.05)
        result = cache.lookup(key)
        assert result.reason == InvalidationReason.expired
        assert cache.stats()["entries"] == 0

    def test_zero_max_age_disables_ttl(self) -> None:
        cache = CacheInvalidator(max_age_seconds=0)
        key = _key()
        cache.store(key, "v")
        time.sleep(0.05)
        assert cache.lookup(key).hit


class TestTreeInvalidation:
    def test_invalidate_by_tree_removes_matching_entries(self) -> None:
        cache = CacheInvalidator()
        cache.store(_key(CacheStage.root_cause, TREE), "rc")
        cache.store(_key(CacheStage.patch, TREE), "p")
        cache.store(_key(CacheStage.root_cause, "c" * 64), "other")
        evicted = cache.invalidate_by_tree(TREE)
        assert evicted == 2
        assert cache.stats()["entries"] == 1
