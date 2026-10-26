"""Cache-hit and verified-reuse metrics reporting.

CacheMetrics accumulates counters across an agent session and produces a
structured report. This is what gets surfaced in the hackathon demo: "We saved
X tokens and $Y by reusing N verified fixes."
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CacheMetrics:
    """Mutable counter bag — increment atomically within a single run."""

    # LLM response cache hits (analysis results reused without a new call)
    cache_hits: int = 0
    cache_misses: int = 0

    # Verified fix reuse (full patch + tests skipped)
    verified_fix_reuses: int = 0

    # Token / cost savings attributed to cache hits
    tokens_saved_by_cache: int = 0
    cost_saved_by_cache_usd: float = 0.0

    # Token / cost savings attributed to verified fix reuse
    tokens_saved_by_reuse: int = 0
    cost_saved_by_reuse_usd: float = 0.0

    # Checkpoint resumptions (stages skipped by loading a checkpoint)
    checkpoint_resumptions: int = 0
    stages_skipped_by_checkpoint: int = 0

    def record_cache_hit(self, tokens_saved: int = 0, cost_saved_usd: float = 0.0) -> None:
        self.cache_hits += 1
        self.tokens_saved_by_cache += tokens_saved
        self.cost_saved_by_cache_usd += cost_saved_usd

    def record_cache_miss(self) -> None:
        self.cache_misses += 1

    def record_fix_reuse(self, tokens_saved: int = 0, cost_saved_usd: float = 0.0) -> None:
        self.verified_fix_reuses += 1
        self.tokens_saved_by_reuse += tokens_saved
        self.cost_saved_by_reuse_usd += cost_saved_usd

    def record_checkpoint_resumption(self, stages_skipped: int) -> None:
        self.checkpoint_resumptions += 1
        self.stages_skipped_by_checkpoint += stages_skipped

    @property
    def total_tokens_saved(self) -> int:
        return self.tokens_saved_by_cache + self.tokens_saved_by_reuse

    @property
    def total_cost_saved_usd(self) -> float:
        return self.cost_saved_by_cache_usd + self.cost_saved_by_reuse_usd

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def report(self) -> dict[str, object]:
        return {
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "verified_fix_reuses": self.verified_fix_reuses,
            "tokens_saved_total": self.total_tokens_saved,
            "cost_saved_usd_total": round(self.total_cost_saved_usd, 6),
            "checkpoint_resumptions": self.checkpoint_resumptions,
            "stages_skipped_by_checkpoint": self.stages_skipped_by_checkpoint,
        }
