"""Model routing: select the cheapest model that satisfies task complexity.

Routing tiers (verified pricing 2026-10-26):

  LIGHT  → muse-spark-1.3-contributor  ($0.10/$0.20 per 1M in/out)
           For: simple lookups, test generation for trivial fixes
  STANDARD → muse-spark-1.3            ($1.25/$4.25 per 1M in/out)
           For: typical root-cause analysis and patch generation
  HEAVY  → muse-wonder-1.0             ($4.00/$20.00 per 1M in/out)
           For: multi-file changes, complex architectural issues

The router does NOT automatically fall back to a different paid provider.
Fallback to Anthropic Claude (or another provider) requires:
  - FALLBACK_PROVIDER_ENABLED=true in env
  - An explicit FALLBACK_BUDGET_USD > 0

If budget is exceeded and fallback is disabled, the run is aborted with
BudgetExceededError rather than silently spending more.
"""

from __future__ import annotations

from enum import StrEnum

from agent.providers.meta import (
    MUSE_SPARK_1_3,
    MUSE_SPARK_1_3_CONTRIBUTOR,
    MUSE_WONDER_1_0,
)


class TaskComplexity(StrEnum):
    light = "light"  # ≤2 files, well-isolated change
    standard = "standard"  # 3-8 files, typical bug fix
    heavy = "heavy"  # >8 files or architectural change


# Ordered cheapest → most expensive
_ROUTING_TABLE: dict[TaskComplexity, str] = {
    TaskComplexity.light: MUSE_SPARK_1_3_CONTRIBUTOR,
    TaskComplexity.standard: MUSE_SPARK_1_3,
    TaskComplexity.heavy: MUSE_WONDER_1_0,
}


class ModelRouter:
    """Select the cheapest appropriate model for a given task complexity.

    Args:
        min_complexity: Override floor — never route below this tier even
            if the task looks simple. Useful for high-stakes stages like
            patch generation where a contributor model may produce lower
            quality output.
        force_model: Pin a specific model identifier, bypassing routing.
    """

    def __init__(
        self,
        min_complexity: TaskComplexity = TaskComplexity.light,
        force_model: str | None = None,
    ) -> None:
        self._min_complexity = min_complexity
        self._force_model = force_model

    def select(self, complexity: TaskComplexity) -> str:
        if self._force_model:
            return self._force_model
        effective = _max_complexity(complexity, self._min_complexity)
        return _ROUTING_TABLE[effective]

    def routing_table(self) -> dict[str, str]:
        return {c.value: m for c, m in _ROUTING_TABLE.items()}


def classify_complexity(affected_file_count: int, is_architectural: bool = False) -> TaskComplexity:
    """Heuristic complexity classification from static signals."""
    if is_architectural or affected_file_count > 8:
        return TaskComplexity.heavy
    if affected_file_count > 2:
        return TaskComplexity.standard
    return TaskComplexity.light


def _max_complexity(a: TaskComplexity, b: TaskComplexity) -> TaskComplexity:
    order = [TaskComplexity.light, TaskComplexity.standard, TaskComplexity.heavy]
    return order[max(order.index(a), order.index(b))]
