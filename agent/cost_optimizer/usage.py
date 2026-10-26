"""Per-call usage accounting and cumulative cost tracking.

Every LLM call must record its token counts and estimated cost via UsageTracker
before the response is returned to the workflow. The workflow reads cumulative
costs to decide whether to continue or abort (budget enforcement).

Cost calculation uses the pricing table in agent/providers/meta.py. If a call
comes from a provider whose pricing is not known, cost is recorded as 0.0 with
a warning — it is never silently omitted from totals.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CallRecord:
    run_id: str
    stage: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    cost_usd: float
    recorded_at_epoch: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def billed_prompt_tokens(self) -> int:
        """Prompt tokens actually billed (excluding cached portion)."""
        return self.prompt_tokens - self.cached_tokens


@dataclass
class RunSummary:
    run_id: str
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cached_tokens: int
    total_cost_usd: float
    call_count: int
    cache_savings_usd: float  # estimated savings from cached tokens


class UsageTracker:
    """Records every LLM call and computes per-run and global summaries."""

    def __init__(self) -> None:
        self._records: list[CallRecord] = []

    def record(self, record: CallRecord) -> None:
        self._records.append(record)

    def run_summary(self, run_id: str) -> RunSummary:
        records = [r for r in self._records if r.run_id == run_id]
        if not records:
            return RunSummary(
                run_id=run_id,
                total_prompt_tokens=0,
                total_completion_tokens=0,
                total_cached_tokens=0,
                total_cost_usd=0.0,
                call_count=0,
                cache_savings_usd=0.0,
            )
        return RunSummary(
            run_id=run_id,
            total_prompt_tokens=sum(r.prompt_tokens for r in records),
            total_completion_tokens=sum(r.completion_tokens for r in records),
            total_cached_tokens=sum(r.cached_tokens for r in records),
            total_cost_usd=sum(r.cost_usd for r in records),
            call_count=len(records),
            cache_savings_usd=_estimate_cache_savings(records),
        )

    def global_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self._records)

    def all_records(self) -> list[CallRecord]:
        return list(self._records)


def _estimate_cache_savings(records: list[CallRecord]) -> float:
    """Rough savings estimate: cached_tokens * (input_rate - cached_rate).

    Rates are averages across all calls in the list; this is an approximation
    because individual calls may use different models/rates.
    """
    total_cached = sum(r.cached_tokens for r in records)
    if total_cached == 0:
        return 0.0
    total_cost = sum(r.cost_usd for r in records)
    total_billed_prompt = sum(r.billed_prompt_tokens for r in records)
    if total_billed_prompt == 0:
        return 0.0
    avg_input_rate_per_token = total_cost / (total_billed_prompt + total_cached)
    return total_cached * avg_input_rate_per_token * 0.9  # rough 90% discount assumption
