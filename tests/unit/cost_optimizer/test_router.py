"""Unit tests for ModelRouter and classify_complexity."""

from __future__ import annotations

import pytest

from agent.cost_optimizer.router import (
    ModelRouter,
    TaskComplexity,
    classify_complexity,
)
from agent.providers.meta import (
    MUSE_SPARK_1_3,
    MUSE_SPARK_1_3_CONTRIBUTOR,
    MUSE_WONDER_1_0,
)

pytestmark = pytest.mark.unit


class TestClassifyComplexity:
    def test_single_file_is_light(self) -> None:
        assert classify_complexity(1) == TaskComplexity.light

    def test_two_files_is_light(self) -> None:
        assert classify_complexity(2) == TaskComplexity.light

    def test_three_files_is_standard(self) -> None:
        assert classify_complexity(3) == TaskComplexity.standard

    def test_eight_files_is_standard(self) -> None:
        assert classify_complexity(8) == TaskComplexity.standard

    def test_nine_files_is_heavy(self) -> None:
        assert classify_complexity(9) == TaskComplexity.heavy

    def test_architectural_flag_forces_heavy(self) -> None:
        assert classify_complexity(1, is_architectural=True) == TaskComplexity.heavy


class TestModelRouter:
    def test_light_routes_to_contributor(self) -> None:
        router = ModelRouter()
        assert router.select(TaskComplexity.light) == MUSE_SPARK_1_3_CONTRIBUTOR

    def test_standard_routes_to_spark(self) -> None:
        router = ModelRouter()
        assert router.select(TaskComplexity.standard) == MUSE_SPARK_1_3

    def test_heavy_routes_to_wonder(self) -> None:
        router = ModelRouter()
        assert router.select(TaskComplexity.heavy) == MUSE_WONDER_1_0

    def test_min_complexity_floor(self) -> None:
        router = ModelRouter(min_complexity=TaskComplexity.standard)
        assert router.select(TaskComplexity.light) == MUSE_SPARK_1_3

    def test_force_model_overrides_routing(self) -> None:
        router = ModelRouter(force_model=MUSE_WONDER_1_0)
        assert router.select(TaskComplexity.light) == MUSE_WONDER_1_0

    def test_routing_table_has_all_tiers(self) -> None:
        table = ModelRouter().routing_table()
        assert set(table.keys()) == {"light", "standard", "heavy"}
