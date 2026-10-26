"""Repository-aware cache invalidation.

A CacheKey covers all inputs that affect the analysis result:
  - Issue fingerprint (content hash)
  - Repository tree hash (hash of file paths + blob SHAs for relevant files)
  - Analysis stage (e.g. "root_cause", "patch", "tests")
  - Model identifier (different models may produce different results)
  - Agent version (breaking changes in prompt logic invalidate old cache entries)

Invalidation triggers:
  - Any change to the files touched by the previous analysis (tree_hash changes).
  - A new agent version that changes prompt logic (AGENT_VERSION bump).
  - Explicit TTL expiry (entries older than max_age_seconds are stale).

The cache is write-through: store() writes immediately. Reads call is_valid()
before returning; stale entries are evicted on read.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

AGENT_VERSION = "0.1.0"


class CacheStage(StrEnum):
    root_cause = "root_cause"
    patch = "patch"
    tests = "tests"
    verification = "verification"


@dataclass(frozen=True)
class CacheKey:
    fingerprint_hex: str
    tree_hash: str
    stage: CacheStage
    model_id: str
    agent_version: str = AGENT_VERSION

    def hex_key(self) -> str:
        canonical = json.dumps(
            {
                "fp": self.fingerprint_hex,
                "tree": self.tree_hash,
                "stage": self.stage.value,
                "model": self.model_id,
                "av": self.agent_version,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass
class CacheEntry:
    key: CacheKey
    value: Any
    stored_at_epoch: float
    hit_count: int = 0


class InvalidationReason(StrEnum):
    not_found = "not_found"
    expired = "expired"
    tree_changed = "tree_changed"
    agent_version_changed = "agent_version_changed"
    valid = "valid"


@dataclass(frozen=True)
class CacheLookup:
    reason: InvalidationReason
    entry: CacheEntry | None = None

    @property
    def hit(self) -> bool:
        return self.reason == InvalidationReason.valid


class CacheInvalidator:
    """In-memory LRU-like cache with explicit invalidation logic.

    max_age_seconds defaults to 24 hours. Set to 0 to disable TTL.
    """

    def __init__(self, max_age_seconds: float = 86_400) -> None:
        self._store: dict[str, CacheEntry] = {}
        self._max_age = max_age_seconds

    def store(self, key: CacheKey, value: Any) -> None:
        self._store[key.hex_key()] = CacheEntry(key=key, value=value, stored_at_epoch=time.time())

    def lookup(self, key: CacheKey) -> CacheLookup:
        hex_key = key.hex_key()
        entry = self._store.get(hex_key)
        if entry is None:
            return CacheLookup(reason=InvalidationReason.not_found)

        now = time.time()
        if self._max_age > 0 and (now - entry.stored_at_epoch) > self._max_age:
            del self._store[hex_key]
            return CacheLookup(reason=InvalidationReason.expired)

        if entry.key.agent_version != AGENT_VERSION:
            del self._store[hex_key]
            return CacheLookup(reason=InvalidationReason.agent_version_changed)

        entry.hit_count += 1
        return CacheLookup(reason=InvalidationReason.valid, entry=entry)

    def invalidate_by_tree(self, tree_hash: str) -> int:
        """Evict all entries whose tree_hash matches (repo changed)."""
        stale = [k for k, e in self._store.items() if e.key.tree_hash == tree_hash]
        for k in stale:
            del self._store[k]
        return len(stale)

    def stats(self) -> dict[str, int]:
        return {
            "entries": len(self._store),
            "total_hits": sum(e.hit_count for e in self._store.values()),
        }
