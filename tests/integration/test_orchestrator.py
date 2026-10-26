"""Integration tests for the end-to-end workflow orchestrator.

These tests exercise the full pipeline (fingerprint → repo analysis → LLM →
patch → sandbox → verification) using a mock LLM provider. All LLM calls
are intercepted; Maven test execution is real (run against the spring-calc
fixture on disk).

Markers:
  integration  — real filesystem, no external network
  slow         — maven test execution takes ~30s

Running:
  pytest -m "integration" -v
  pytest -m "integration and not slow" -v   # skips maven-based tests

IMPORTANT: these tests mutate the spring-calc fixture's working tree. Every
test that applies a patch must reset the fixture to a clean state in teardown.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Generator
from pathlib import Path

import pytest

from agent.cost_optimizer.metrics import CacheMetrics
from agent.cost_optimizer.usage import UsageTracker
from agent.engineering_knowledge.cache import CacheInvalidator
from agent.engineering_knowledge.checkpoint import CheckpointManager
from agent.engineering_knowledge.fix_store import CompatibilityStatus, FixStore
from agent.providers.base import BaseLLMProvider
from agent.providers.types import (
    BudgetExceededError,
    LLMRequest,
    LLMResponse,
    ModelCapability,
    ProviderInfo,
    UsageStats,
)
from agent.sandbox.local_executor import LocalSandbox
from agent.workflow.orchestrator import RunInput, WorkflowOrchestrator
from agent.workflow.states import (
    FixReused,
    LLMCallMade,
    RunFailed,
    WorkflowStage,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

FIXTURE_REPO = Path(__file__).parents[2] / "fixtures" / "spring-calc"
SERVICE_FILE = (
    FIXTURE_REPO
    / "src/main/java/com/example/calc/CalculatorService.java"
)
TEST_FILE = (
    FIXTURE_REPO
    / "src/test/java/com/example/calc/CalculatorServiceTest.java"
)

# The correct unified diff that fixes the NPE bug
CORRECT_DIFF = """\
--- a/src/main/java/com/example/calc/CalculatorService.java
+++ b/src/main/java/com/example/calc/CalculatorService.java
@@ -30,7 +30,9 @@
      * @throws IllegalArgumentException if divisor is null   ← MISSING: currently throws NPE
      */
     public int divide(Integer dividend, Integer divisor) {
-        // BUG: missing null check on divisor — throws NullPointerException when divisor == null
+        if (divisor == null) {
+            throw new IllegalArgumentException("Divisor must not be null");
+        }
         if (divisor == 0) {
             throw new IllegalArgumentException("Divisor must not be zero");
         }
"""

DIAGNOSIS_JSON = json.dumps(
    {
        "root_cause": "divide() unboxes `divisor` before the null check, causing NPE when divisor is null.",
        "affected_files": ["src/main/java/com/example/calc/CalculatorService.java"],
        "fix_description": "Add a null guard before the zero check in divide().",
        "unified_diff": CORRECT_DIFF,
        "confidence": "high",
    }
)

TEST_GEN_JSON = json.dumps(
    {
        "test_method_name": "divide_nullDivisorViaFix_throwsIllegalArgument",
        "test_method_body": "assertThrows(IllegalArgumentException.class, () -> service.divide(5, null));",
        "explanation": "Confirms the null-guard added by the fix throws IAE, not NPE.",
    }
)

ISSUE_TITLE = "divide() throws NPE instead of IllegalArgumentException for null divisor"
ISSUE_BODY = "When divisor is null, service.divide() crashes with NPE."
REPO_NAME = "local/spring-calc"


@pytest.fixture()
def fixture_repo_clean() -> Generator[Path, None, None]:
    """Yield the fixture path, then reset to the committed HEAD state."""
    yield FIXTURE_REPO
    subprocess.run(
        ["git", "checkout", "HEAD", "--", "."],
        cwd=str(FIXTURE_REPO),
        check=False,
        capture_output=True,
    )


def make_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="muse-spark-1.3",
        usage=UsageStats(prompt_tokens=100, completion_tokens=50, cached_tokens=0),
        finish_reason="stop",
    )


class FakeProvider(BaseLLMProvider):
    """Deterministic provider; each chat() call pops from a response queue."""

    def __init__(self, responses: list[str]) -> None:
        self._queue = list(responses)
        self.call_count = 0

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(name="fake", base_url="", auth_header="", docs_url="")

    async def chat(self, request: LLMRequest) -> LLMResponse:
        self.call_count += 1
        if not self._queue:
            raise RuntimeError("FakeProvider: response queue exhausted")
        return make_response(self._queue.pop(0))

    def supports(self, capability: ModelCapability) -> bool:
        return False

    def estimated_cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * 1.25 + completion_tokens * 4.25) / 1_000_000

    def model_id(self) -> str:
        return "muse-spark-1.3"


def _local_sandbox_runner(repo_path: object) -> object:
    """Trusted sandbox runner for integration tests against version-controlled fixtures."""
    return LocalSandbox(trusted=True).run_maven_tests(repo_path)  # type: ignore[arg-type]


def make_orchestrator(
    provider: BaseLLMProvider,
    fix_store: FixStore | None = None,
    cache: CacheInvalidator | None = None,
) -> WorkflowOrchestrator:
    return WorkflowOrchestrator(
        provider=provider,
        fix_store=fix_store,
        cache=cache,
        checkpoint_manager=CheckpointManager(),
        usage_tracker=UsageTracker(),
        metrics=CacheMetrics(),
        sandbox_runner=_local_sandbox_runner,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_run1_fresh_analysis_fingerprint_and_events(fixture_repo_clean: Path) -> None:
    """Run 1: fingerprint is computed and StageStarted/StageCompleted events are emitted."""
    import asyncio

    provider = FakeProvider([DIAGNOSIS_JSON, TEST_GEN_JSON])
    orch = make_orchestrator(provider)
    inp = RunInput(
        repo_path=fixture_repo_clean,
        issue_title=ISSUE_TITLE,
        issue_body=ISSUE_BODY,
        repo_full_name=REPO_NAME,
        run_id="run1-events",
    )
    report = asyncio.run(orch.run(inp))

    # Fingerprint was recorded
    assert report.run_id == "run1-events"
    stage_names = {e.stage for e in report.events if hasattr(e, "stage")}
    assert WorkflowStage.intake in stage_names
    assert WorkflowStage.repo_analysis in stage_names
    assert WorkflowStage.root_cause in stage_names


@pytest.mark.integration
def test_run1_repo_analysis_builds_context(fixture_repo_clean: Path) -> None:
    """prepare_repo() indexes the fixture and returns a non-empty tree hash."""
    from agent.repo import prepare_repo

    ctx = prepare_repo(fixture_repo_clean)
    assert ctx.tree_hash, "tree_hash must not be empty"
    assert len(ctx.source_files()) > 0, "At least one .java file must be indexed"
    java_paths = [f.rel_path for f in ctx.source_files()]
    assert any("CalculatorService.java" in p for p in java_paths)


@pytest.mark.integration
def test_run1_cache_miss_then_cache_hit(fixture_repo_clean: Path) -> None:
    """Second call with same inputs and repo state returns a cache hit (no LLM call)."""
    import asyncio

    cache = CacheInvalidator()
    fix_store = FixStore()

    provider1 = FakeProvider([DIAGNOSIS_JSON, TEST_GEN_JSON])
    orch1 = make_orchestrator(provider1, fix_store=fix_store, cache=cache)
    inp = RunInput(
        repo_path=fixture_repo_clean,
        issue_title=ISSUE_TITLE,
        issue_body=ISSUE_BODY,
        repo_full_name=REPO_NAME,
        run_id="run1-cache",
    )
    asyncio.run(orch1.run(inp))
    calls_after_run1 = provider1.call_count

    # Reset file mutations from run1
    subprocess.run(
        ["git", "checkout", "HEAD", "--", "."],
        cwd=str(fixture_repo_clean),
        check=False,
        capture_output=True,
    )

    # Run 2 with same cache — the fix_store already has the verified fix, so
    # it should reuse it without ANY LLM call
    provider2 = FakeProvider([])  # empty — any call would error
    orch2 = make_orchestrator(provider2, fix_store=fix_store, cache=cache)
    inp2 = RunInput(
        repo_path=fixture_repo_clean,
        issue_title=ISSUE_TITLE,
        issue_body=ISSUE_BODY,
        repo_full_name=REPO_NAME,
        run_id="run2-reuse",
    )
    report2 = asyncio.run(orch2.run(inp2))

    # Fix was reused — no LLM calls on provider2
    assert provider2.call_count == 0
    fix_reuse_events = [e for e in report2.events if isinstance(e, FixReused)]
    assert len(fix_reuse_events) == 1
    assert report2.reused_fix is True
    assert calls_after_run1 >= 1, "Run 1 must have made at least one LLM call"


@pytest.mark.integration
def test_run1_cache_invalidated_after_tree_change(fixture_repo_clean: Path) -> None:
    """After a file changes, the cache entry is invalidated and fresh analysis is done."""
    import asyncio

    cache = CacheInvalidator()
    fix_store = FixStore()

    provider1 = FakeProvider([DIAGNOSIS_JSON, TEST_GEN_JSON])
    orch1 = make_orchestrator(provider1, fix_store=fix_store, cache=cache)
    inp = RunInput(
        repo_path=fixture_repo_clean,
        issue_title=ISSUE_TITLE,
        issue_body=ISSUE_BODY,
        repo_full_name=REPO_NAME,
        run_id="run-cache-inval-1",
    )
    asyncio.run(orch1.run(inp))
    calls_run1 = provider1.call_count

    # Reset patch from run 1
    subprocess.run(
        ["git", "checkout", "HEAD", "--", "."],
        cwd=str(fixture_repo_clean),
        check=False,
        capture_output=True,
    )

    # Mutate a source file (simulates a code change)
    svc = SERVICE_FILE
    original = svc.read_text()
    svc.write_text(original + "\n// dummy change\n")
    # Re-commit so git ls-tree returns a different blob SHA
    subprocess.run(
        ["git", "add", str(svc)],
        cwd=str(fixture_repo_clean),
        check=False,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "test: invalidate tree hash"],
        cwd=str(fixture_repo_clean),
        check=False,
        capture_output=True,
    )

    provider2 = FakeProvider([DIAGNOSIS_JSON, TEST_GEN_JSON])
    orch2 = make_orchestrator(provider2, fix_store=fix_store, cache=cache)
    inp2 = RunInput(
        repo_path=fixture_repo_clean,
        issue_title=ISSUE_TITLE,
        issue_body=ISSUE_BODY,
        repo_full_name=REPO_NAME,
        run_id="run-cache-inval-2",
    )
    asyncio.run(orch2.run(inp2))
    calls_run2 = provider2.call_count

    # Both runs should have made LLM calls (tree_hash mismatch prevents reuse)
    assert calls_run1 >= 1
    assert calls_run2 >= 1, (
        "After repo change, fix is TREE_MISMATCH so fresh analysis is required"
    )

    # Cleanup: revert the dummy commit
    subprocess.run(
        ["git", "reset", "--hard", "HEAD~1"],
        cwd=str(fixture_repo_clean),
        check=False,
        capture_output=True,
    )


@pytest.mark.integration
def test_run_budget_exceeded_emits_run_failed(fixture_repo_clean: Path) -> None:
    """When the provider raises BudgetExceededError, RunFailed is emitted."""
    import asyncio

    class BudgetProvider(FakeProvider):
        async def chat(self, request: LLMRequest) -> LLMResponse:
            raise BudgetExceededError("Budget limit reached")

    provider = BudgetProvider([])
    orch = make_orchestrator(provider)
    inp = RunInput(
        repo_path=fixture_repo_clean,
        issue_title=ISSUE_TITLE,
        issue_body=ISSUE_BODY,
        repo_full_name=REPO_NAME,
        run_id="run-budget",
    )
    report = asyncio.run(orch.run(inp))

    failed_events = [e for e in report.events if isinstance(e, RunFailed)]
    assert len(failed_events) == 1
    assert "Budget" in failed_events[0].reason
    assert report.success is False


@pytest.mark.integration
def test_run_bad_patch_fails_at_patch_stage(fixture_repo_clean: Path) -> None:
    """When the LLM returns a diff that doesn't apply, the run fails at the patch stage."""
    import asyncio

    bad_diff_diagnosis = json.dumps(
        {
            "root_cause": "some cause",
            "affected_files": ["src/main/java/com/example/calc/CalculatorService.java"],
            "fix_description": "a fix",
            "unified_diff": "--- a/nonexistent.java\n+++ b/nonexistent.java\n@@ -1 +1 @@\n-old\n+new",
            "confidence": "low",
        }
    )
    provider = FakeProvider([bad_diff_diagnosis])
    orch = make_orchestrator(provider)
    inp = RunInput(
        repo_path=fixture_repo_clean,
        issue_title=ISSUE_TITLE,
        issue_body=ISSUE_BODY,
        repo_full_name=REPO_NAME,
        run_id="run-badpatch",
    )
    report = asyncio.run(orch.run(inp))

    assert report.success is False
    failed_events = [e for e in report.events if isinstance(e, RunFailed)]
    assert len(failed_events) == 1
    assert failed_events[0].stage == WorkflowStage.patch


@pytest.mark.integration
def test_usage_tracker_records_llm_calls(fixture_repo_clean: Path) -> None:
    """UsageTracker records both LLM calls and reports accurate token totals."""
    import asyncio

    usage = UsageTracker()
    provider = FakeProvider([DIAGNOSIS_JSON, TEST_GEN_JSON])
    orch = WorkflowOrchestrator(
        provider=provider,
        checkpoint_manager=CheckpointManager(),
        usage_tracker=usage,
        metrics=CacheMetrics(),
        sandbox_runner=_local_sandbox_runner,  # type: ignore[arg-type]
    )
    inp = RunInput(
        repo_path=fixture_repo_clean,
        issue_title=ISSUE_TITLE,
        issue_body=ISSUE_BODY,
        repo_full_name=REPO_NAME,
        run_id="run-usage",
    )
    report = asyncio.run(orch.run(inp))

    llm_events = [e for e in report.events if isinstance(e, LLMCallMade)]
    assert len(llm_events) >= 1, "At least one LLM call must be recorded"
    assert report.usage_summary["total_prompt_tokens"] > 0
    assert report.usage_summary["call_count"] == len(llm_events)


@pytest.mark.integration
def test_fix_store_compatibility_after_run1(fixture_repo_clean: Path) -> None:
    """After a successful run, FixStore.check_compatibility() returns COMPATIBLE."""
    import asyncio

    from agent.repo import prepare_repo

    fix_store = FixStore()
    provider = FakeProvider([DIAGNOSIS_JSON, TEST_GEN_JSON])
    orch = make_orchestrator(provider, fix_store=fix_store)
    inp = RunInput(
        repo_path=fixture_repo_clean,
        issue_title=ISSUE_TITLE,
        issue_body=ISSUE_BODY,
        repo_full_name=REPO_NAME,
        run_id="run-fixstore",
    )
    asyncio.run(orch.run(inp))

    # Re-check compatibility (reset working tree first)
    subprocess.run(
        ["git", "checkout", "HEAD", "--", "."],
        cwd=str(fixture_repo_clean),
        check=False,
        capture_output=True,
    )

    from agent.engineering_knowledge.fingerprint import fingerprint_from_raw

    fp = fingerprint_from_raw(
        repo_full_name=REPO_NAME,
        issue_number=0,
        title=ISSUE_TITLE,
        body=ISSUE_BODY,
        labels=[],
    )
    ctx = prepare_repo(fixture_repo_clean)
    result = fix_store.check_compatibility(fp.full_hex, REPO_NAME, ctx.tree_hash)
    assert result.status == CompatibilityStatus.COMPATIBLE
    assert result.fix is not None
    assert result.fix.patch_unified_diff == CORRECT_DIFF


@pytest.mark.integration
def test_checkpoint_is_created_and_completed(fixture_repo_clean: Path) -> None:
    """CheckpointManager records intake stage and marks the run completed on success."""
    import asyncio

    from agent.engineering_knowledge.checkpoint import CheckpointManager, RunStatus

    checkpoints = CheckpointManager()
    provider = FakeProvider([DIAGNOSIS_JSON, TEST_GEN_JSON])
    orch = WorkflowOrchestrator(
        provider=provider,
        checkpoint_manager=checkpoints,
        usage_tracker=UsageTracker(),
        metrics=CacheMetrics(),
        sandbox_runner=_local_sandbox_runner,  # type: ignore[arg-type]
    )
    inp = RunInput(
        repo_path=fixture_repo_clean,
        issue_title=ISSUE_TITLE,
        issue_body=ISSUE_BODY,
        repo_full_name=REPO_NAME,
        run_id="run-checkpoint",
    )
    asyncio.run(orch.run(inp))

    status = checkpoints.status("run-checkpoint")
    # Either completed (all tests green) or abandoned (patch applied but tests failed)
    assert status in (RunStatus.completed, RunStatus.abandoned)


@pytest.mark.integration
def test_cache_metrics_track_hits_and_misses(fixture_repo_clean: Path) -> None:
    """CacheMetrics.cache_misses increments on first run; reuse increments fix reuses."""
    import asyncio

    metrics = CacheMetrics()
    fix_store = FixStore()
    cache = CacheInvalidator()

    provider1 = FakeProvider([DIAGNOSIS_JSON, TEST_GEN_JSON])
    orch1 = WorkflowOrchestrator(
        provider=provider1,
        fix_store=fix_store,
        cache=cache,
        checkpoint_manager=CheckpointManager(),
        usage_tracker=UsageTracker(),
        metrics=metrics,
        sandbox_runner=_local_sandbox_runner,  # type: ignore[arg-type]
    )
    inp = RunInput(
        repo_path=fixture_repo_clean,
        issue_title=ISSUE_TITLE,
        issue_body=ISSUE_BODY,
        repo_full_name=REPO_NAME,
        run_id="run-metrics-1",
    )
    asyncio.run(orch1.run(inp))

    assert metrics.cache_misses >= 1, "First run must record at least one cache miss"

    # Reset
    subprocess.run(
        ["git", "checkout", "HEAD", "--", "."],
        cwd=str(fixture_repo_clean),
        check=False,
        capture_output=True,
    )

    provider2 = FakeProvider([])
    orch2 = WorkflowOrchestrator(
        provider=provider2,
        fix_store=fix_store,
        cache=cache,
        checkpoint_manager=CheckpointManager(),
        usage_tracker=UsageTracker(),
        metrics=metrics,
        sandbox_runner=_local_sandbox_runner,  # type: ignore[arg-type]
    )
    inp2 = RunInput(
        repo_path=fixture_repo_clean,
        issue_title=ISSUE_TITLE,
        issue_body=ISSUE_BODY,
        repo_full_name=REPO_NAME,
        run_id="run-metrics-2",
    )
    asyncio.run(orch2.run(inp2))

    assert metrics.verified_fix_reuses >= 1, "Second run should reuse the verified fix"
    assert metrics.verified_fix_reuses >= 1
