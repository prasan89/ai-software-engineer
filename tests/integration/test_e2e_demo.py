"""End-to-end demonstration tests for the AI Software Engineer Agent.

Purpose
-------
This module contains the three-run demonstration required by the acceptance
criteria.  Unlike tests/integration/test_orchestrator.py (which uses the
shared `fixtures/spring-calc/` tree), every test here operates on a **per-test
copy** of the fixture initialised inside pytest's `tmp_path`.  That means:

  * No teardown pollution between tests.
  * Tests can mutate, commit, or destroy the repo without affecting each other.
  * The sandbox still runs `mvn test` in a real subprocess — there is no Docker
    isolation yet (see SANDBOX SECURITY NOTE below).

Three-run demonstration
-----------------------
  Run 1 (fresh)   : fingerprint → repo analysis → mock-LLM diagnosis →
                    git-apply patch → mvn test → verify → FixStore.save()
  Run 2 (reuse)   : same issue + identical tree hash → FixStore.COMPATIBLE →
                    zero LLM calls → re-apply patch → mvn test → verify
  Run 3 (stale)   : add a commit to the fixture repo (tree hash changes) →
                    FixStore returns TREE_MISMATCH → fresh LLM call required

All three runs use FakeProvider (mock LLM) with a deterministic hard-coded
patch.  Maven runs for real; costs come from FakeProvider's pricing constants.

SANDBOX SECURITY NOTE
---------------------
`run_maven_tests()` currently executes `mvn test` in a subprocess on the
host.  There is **no** Docker isolation, no network firewall, and no resource
cap beyond the subprocess timeout.  For this fixture repository that is safe
because:

  * The fixture has no network dependencies (Spring Boot test context only).
  * No scripts from the fixture directory are executed by the agent — only
    `mvn test` is invoked with a hard-coded command.
  * The patch is produced by FakeProvider (a constant string), not by an
    untrusted LLM response.

Before accepting untrusted LLM-generated patches the sandbox MUST be upgraded
to Docker isolation (--network=none, --read-only, resource limits).  The
orchestrator already documents this in agent/sandbox/__init__.py.

Test markers
------------
  integration   real filesystem ops, no external network
  slow          each maven run takes ~30 s
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from agent.cost_optimizer.metrics import CacheMetrics
from agent.cost_optimizer.usage import UsageTracker
from agent.engineering_knowledge.cache import CacheInvalidator
from agent.engineering_knowledge.checkpoint import CheckpointManager, RunStatus
from agent.engineering_knowledge.fingerprint import fingerprint_from_raw
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
from agent.repo import prepare_repo
from agent.sandbox.local_executor import LocalSandbox
from agent.workflow.orchestrator import RunInput, RunReport, WorkflowOrchestrator
from agent.workflow.states import (
    FixReused,
    LLMCallMade,
    RunFailed,
    WorkflowStage,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIXTURE_SRC = Path(__file__).parents[2] / "fixtures" / "spring-calc"

# ---------------------------------------------------------------------------
# The known-good patch for the null-divisor bug (verified to apply with
# `git apply --check` on a clean checkout of fixtures/spring-calc).
# ---------------------------------------------------------------------------
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

# Diagnosis JSON that FakeProvider returns for the null-divisor bug.
DIAGNOSIS_JSON = json.dumps({
    "root_cause": (
        "divide() unboxes `divisor` via implicit Integer.intValue() before any "
        "null check, causing NullPointerException when divisor is null."
    ),
    "affected_files": [
        "src/main/java/com/example/calc/CalculatorService.java"
    ],
    "fix_description": "Add an explicit null guard on `divisor` before the zero check.",
    "unified_diff": CORRECT_DIFF,
    "confidence": "high",
})

# Test-generation JSON that FakeProvider returns.
TEST_GEN_JSON = json.dumps({
    "test_method_name": "divide_withNullDivisor_throwsIllegalArgumentException",
    "test_method_body": (
        "assertThrows(IllegalArgumentException.class, "
        "() -> service.divide(10, null), "
        "\"null divisor must throw IAE\");"
    ),
    "explanation": (
        "Regression guard: confirms the null guard added by the fix throws "
        "IllegalArgumentException instead of NullPointerException."
    ),
})

ISSUE_TITLE = "divide() throws NPE instead of IAE for null divisor"
ISSUE_BODY = textwrap.dedent("""\
    When `divisor` is null, CalculatorService.divide() throws
    NullPointerException instead of IllegalArgumentException.
    See ISSUE.md for reproduction steps and acceptance criteria.
""")
REPO_NAME = "local/spring-calc"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _copy_fixture(dest: Path) -> Path:
    """Copy the spring-calc fixture into dest and initialise a fresh git repo.

    We copy rather than clone so that:
      * Tests are independent of the fixture's git history.
      * Each test can make commits without polluting the shared fixture.
      * Tree hashes are stable for the file contents actually present.
    """
    repo_dir = dest / "spring-calc"
    # shutil.copytree respects symlinks but we need to exclude .git and target
    shutil.copytree(
        FIXTURE_SRC, repo_dir,
        ignore=shutil.ignore_patterns(".git", "target"),
    )
    # Fresh git repo so git apply and git ls-tree work
    subprocess.run(["git", "init", "-b", "main"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "feat: initial fixture"],
        cwd=str(repo_dir), check=True, capture_output=True,
        env={**__import__("os").environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t",
             "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t"},
    )
    return repo_dir


def _reset_repo(repo_dir: Path) -> None:
    """Hard-reset the workspace to the HEAD commit (discard all working-tree changes)."""
    subprocess.run(
        ["git", "checkout", "HEAD", "--", "."],
        cwd=str(repo_dir), check=False, capture_output=True,
    )


def _make_response(content: str, cached: int = 0) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="muse-spark-1.3",
        usage=UsageStats(prompt_tokens=400, completion_tokens=150, cached_tokens=cached),
        finish_reason="stop",
    )


class FakeProvider(BaseLLMProvider):
    """Deterministic provider; each chat() call pops the next response from a queue."""

    # muse-spark-1.3 pricing (verified 2026-10-26):
    # $1.25 / 1M input tokens, $4.25 / 1M output tokens
    _INPUT_RATE = 1.25 / 1_000_000
    _OUTPUT_RATE = 4.25 / 1_000_000

    def __init__(self, responses: list[str]) -> None:
        self._queue = list(responses)
        self.call_count = 0

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(name="fake", base_url="", auth_header="", docs_url="")

    async def chat(self, request: LLMRequest) -> LLMResponse:
        self.call_count += 1
        if not self._queue:
            raise RuntimeError(
                f"FakeProvider: response queue exhausted after {self.call_count} calls"
            )
        return _make_response(self._queue.pop(0))

    def supports(self, capability: ModelCapability) -> bool:
        return False

    def estimated_cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        return prompt_tokens * self._INPUT_RATE + completion_tokens * self._OUTPUT_RATE

    def model_id(self) -> str:
        return "muse-spark-1.3"


class BudgetProvider(FakeProvider):
    """Provider that raises BudgetExceededError on the first call."""

    async def chat(self, request: LLMRequest) -> LLMResponse:
        self.call_count += 1
        raise BudgetExceededError("Configured budget cap reached ($0.01 limit)")


class BadPatchThenGoodProvider(FakeProvider):
    """Provider that returns an unapplicable patch on attempt 1 and the correct patch on attempt 2.

    This exercises the MAX_REPAIR_ATTEMPTS retry loop.  Since each repair
    attempt calls the LLM twice (diagnosis + test-gen), and we want the first
    attempt to fail at the patch stage, we return a bad diagnosis first, then
    a good diagnosis for the second attempt.

    Actually the current orchestrator only calls LLM once per run (diagnosis
    is cached after the first call).  The retry loop re-uses the same diff.
    So to simulate a "bad patch then good patch" we need a provider that returns
    a different diff on the second call.  We do that here.
    """

    def __init__(self, bad_diff: str, good_diff: str, test_json: str) -> None:
        self._bad_diag = json.dumps({
            "root_cause": "bad attempt",
            "affected_files": [],
            "fix_description": "bad fix",
            "unified_diff": bad_diff,
            "confidence": "low",
        })
        self._good_diag = json.dumps({
            "root_cause": "correct root cause",
            "affected_files": ["src/main/java/com/example/calc/CalculatorService.java"],
            "fix_description": "add null guard",
            "unified_diff": good_diff,
            "confidence": "high",
        })
        self._test_json = test_json
        self.call_count = 0
        self._calls: list[str] = []

    async def chat(self, request: LLMRequest) -> LLMResponse:
        self.call_count += 1
        if self.call_count == 1:
            return _make_response(self._bad_diag)
        if self.call_count == 2:
            return _make_response(self._good_diag)
        return _make_response(self._test_json)


def _local_sandbox_runner(repo_path: object) -> object:
    """Trusted sandbox runner for integration tests against version-controlled fixtures."""
    return LocalSandbox(trusted=True).run_maven_tests(repo_path)  # type: ignore[arg-type]


def _make_orch(
    provider: BaseLLMProvider,
    fix_store: FixStore | None = None,
    cache: CacheInvalidator | None = None,
    checkpoints: CheckpointManager | None = None,
    usage: UsageTracker | None = None,
    metrics: CacheMetrics | None = None,
) -> WorkflowOrchestrator:
    return WorkflowOrchestrator(
        provider=provider,
        fix_store=fix_store,
        cache=cache,
        checkpoint_manager=checkpoints,
        usage_tracker=usage,
        metrics=metrics,
        sandbox_runner=_local_sandbox_runner,  # type: ignore[arg-type]
    )


def _print_report(label: str, report: RunReport) -> None:
    """Print a structured summary of a run report (visible with pytest -s)."""
    llm_events = [e for e in report.events if isinstance(e, LLMCallMade)]
    reuse_events = [e for e in report.events if isinstance(e, FixReused)]
    failed_events = [e for e in report.events if isinstance(e, RunFailed)]
    u = report.usage_summary
    v = report.verification
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  run_id        : {report.run_id}")
    print(f"  success       : {report.success}")
    print(f"  stage_reached : {report.stage_reached}")
    print(f"  reused_fix    : {report.reused_fix}")
    print(f"  llm_calls     : {len(llm_events)}")
    print(f"  fix_reuses    : {len(reuse_events)}")
    print(f"  failed_events : {len(failed_events)}")
    print(f"  prompt_tokens : {u['total_prompt_tokens']}")
    print(f"  compl_tokens  : {u['total_completion_tokens']}")
    print(f"  cached_tokens : {u['cached_tokens']}")
    print(f"  cost_usd      : ${u['total_cost_usd']:.6f}")
    print(f"  call_count    : {u['call_count']}")
    if v:
        print(f"  tests_run     : {v.tests_run}")
        print(f"  tests_passed  : {v.tests_passed}")
        print(f"  tests_failed  : {v.tests_failed}")
        print(f"  passed        : {v.passed}")
    if report.error:
        print(f"  error         : {report.error[:120]}")
    print(f"  elapsed_s     : {report.elapsed_seconds}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Shared state for the three-run demo (module-scoped)
# ---------------------------------------------------------------------------

class _ThreeRunState:
    fix_store: FixStore = FixStore()
    cache: CacheInvalidator = CacheInvalidator()
    repo_dir: Path | None = None
    run1_report: RunReport | None = None
    run2_report: RunReport | None = None
    run3_report: RunReport | None = None
    run1_tree_hash: str = ""


_state = _ThreeRunState()


# ---------------------------------------------------------------------------
# Three-run demonstration (module-scoped fixture so repos persist)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def three_run_workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a single workspace used by all three-run demo tests."""
    base = tmp_path_factory.mktemp("three_run")
    repo = _copy_fixture(base)
    _state.repo_dir = repo
    return repo


# ---------------------------------------------------------------------------
# Run 1 — fresh analysis
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
def test_run1_fresh_analysis(three_run_workspace: Path) -> None:
    """Run 1: fresh issue, no cached fix, 2 LLM calls, all tests green.

    After this test:
      * FixStore contains one entry for the null-divisor fingerprint.
      * The entry's repo_tree_hash matches the current tree.
      * verification.passed == True with 6 tests passing (5 original + 1 injected).
    """
    _reset_repo(three_run_workspace)

    provider = FakeProvider([DIAGNOSIS_JSON, TEST_GEN_JSON])
    orch = _make_orch(
        provider,
        fix_store=_state.fix_store,
        cache=_state.cache,
        usage=UsageTracker(),
        metrics=CacheMetrics(),
    )
    inp = RunInput(
        repo_path=three_run_workspace,
        issue_title=ISSUE_TITLE,
        issue_body=ISSUE_BODY,
        repo_full_name=REPO_NAME,
        run_id="demo-run1",
    )

    report = asyncio.run(orch.run(inp))
    _state.run1_report = report

    # Capture tree hash for Run 3 stale-check
    _reset_repo(three_run_workspace)
    ctx = prepare_repo(three_run_workspace)
    _state.run1_tree_hash = ctx.tree_hash

    _print_report("RUN 1 — FRESH ANALYSIS", report)

    # --- assertions ---
    assert report.success, f"Run 1 must succeed. error={report.error!r}"
    assert report.stage_reached == WorkflowStage.done
    assert report.reused_fix is False

    # Exactly 2 LLM calls: diagnosis + test generation
    llm_events = [e for e in report.events if isinstance(e, LLMCallMade)]
    assert len(llm_events) == 2, f"Expected 2 LLM calls, got {len(llm_events)}"

    # Verification passed
    assert report.verification is not None
    assert report.verification.passed
    assert report.verification.tests_run >= 5, (
        f"Expected ≥5 tests, got {report.verification.tests_run}"
    )

    # Costs are non-zero (FakeProvider records real token counts)
    assert report.usage_summary["total_prompt_tokens"] > 0
    assert report.usage_summary["total_cost_usd"] > 0.0

    # FixStore now has an entry
    assert len(_state.fix_store) == 1, "FixStore must contain the verified fix after Run 1"

    # FixStore entry has a matching tree_hash
    fp = fingerprint_from_raw(
        repo_full_name=REPO_NAME, issue_number=0,
        title=ISSUE_TITLE, body=ISSUE_BODY, labels=[],
    )
    result = _state.fix_store.check_compatibility(fp.full_hex, REPO_NAME, _state.run1_tree_hash)
    assert result.status == CompatibilityStatus.COMPATIBLE, (
        f"Expected COMPATIBLE, got {result.status}"
    )
    assert result.fix is not None
    assert result.fix.repo_tree_hash == _state.run1_tree_hash
    assert result.fix.patch_unified_diff == CORRECT_DIFF


# ---------------------------------------------------------------------------
# Run 2 — fix reuse
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
def test_run2_fix_reuse(three_run_workspace: Path) -> None:
    """Run 2: same issue, same tree → zero LLM calls, reuses verified fix.

    After this test:
      * FakeProvider was never called (queue was empty; any call would error).
      * report.reused_fix == True.
      * verification.passed == True (patch re-applied and tests re-run).
    """
    _reset_repo(three_run_workspace)

    provider = FakeProvider([])  # empty — any call would raise RuntimeError
    orch = _make_orch(
        provider,
        fix_store=_state.fix_store,
        cache=_state.cache,
        usage=UsageTracker(),
        metrics=CacheMetrics(),
    )
    inp = RunInput(
        repo_path=three_run_workspace,
        issue_title=ISSUE_TITLE,
        issue_body=ISSUE_BODY,
        repo_full_name=REPO_NAME,
        run_id="demo-run2",
    )

    report = asyncio.run(orch.run(inp))
    _state.run2_report = report
    _print_report("RUN 2 — FIX REUSE (zero LLM calls)", report)

    assert report.success, f"Run 2 must succeed. error={report.error!r}"
    assert report.reused_fix is True
    assert provider.call_count == 0, (
        f"Run 2 must make zero LLM calls; made {provider.call_count}"
    )

    reuse_events = [e for e in report.events if isinstance(e, FixReused)]
    assert len(reuse_events) == 1

    # Tests must still pass after re-applying the cached patch
    assert report.verification is not None
    assert report.verification.passed
    assert report.verification.tests_run >= 5

    # Cost is zero (no LLM calls)
    assert report.usage_summary["total_cost_usd"] == 0.0

    # Confirm the diff that was re-applied is the stored one
    assert report.unified_diff == CORRECT_DIFF


# ---------------------------------------------------------------------------
# Run 3 — stale fix rejected after tree change
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
def test_run3_stale_fix_rejected_after_source_change(three_run_workspace: Path) -> None:
    """Run 3: after a source change the FixStore entry is TREE_MISMATCH → fresh LLM call.

    Procedure:
      1. Reset workspace to clean state.
      2. Append a comment to CalculatorService.java and commit it.
         This changes the git blob SHA → new tree_hash.
      3. Run the orchestrator again with the same issue.
      4. FixStore.check_compatibility() must return TREE_MISMATCH.
      5. The orchestrator falls through to a fresh LLM analysis.
      6. FakeProvider must have been called at least once.
    """
    _reset_repo(three_run_workspace)

    # --- mutate source and commit ---
    svc = (
        three_run_workspace
        / "src/main/java/com/example/calc/CalculatorService.java"
    )
    original = svc.read_text(encoding="utf-8")
    svc.write_text(original + "\n// dependency-level change: adds a public constant\n", encoding="utf-8")
    env = {
        **__import__("os").environ,
        "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t",
    }
    subprocess.run(["git", "add", str(svc)], cwd=str(three_run_workspace), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "chore: add constant (changes tree hash)"],
        cwd=str(three_run_workspace), check=True, capture_output=True, env=env,
    )

    # Verify tree hash changed
    ctx_new = prepare_repo(three_run_workspace)
    assert ctx_new.tree_hash != _state.run1_tree_hash, (
        "Tree hash must change after source file commit"
    )

    # Verify FixStore sees TREE_MISMATCH for the new tree
    fp = fingerprint_from_raw(
        repo_full_name=REPO_NAME, issue_number=0,
        title=ISSUE_TITLE, body=ISSUE_BODY, labels=[],
    )
    compat = _state.fix_store.check_compatibility(fp.full_hex, REPO_NAME, ctx_new.tree_hash)
    assert compat.status == CompatibilityStatus.TREE_MISMATCH, (
        f"Expected TREE_MISMATCH, got {compat.status}"
    )

    # Now run: orchestrator must call LLM again
    provider = FakeProvider([DIAGNOSIS_JSON, TEST_GEN_JSON])
    orch = _make_orch(
        provider,
        fix_store=_state.fix_store,  # shares the fix_store (has Run 1 entry)
        cache=_state.cache,
        usage=UsageTracker(),
        metrics=CacheMetrics(),
    )
    inp = RunInput(
        repo_path=three_run_workspace,
        issue_title=ISSUE_TITLE,
        issue_body=ISSUE_BODY,
        repo_full_name=REPO_NAME,
        run_id="demo-run3",
    )

    report = asyncio.run(orch.run(inp))
    _state.run3_report = report
    _print_report("RUN 3 — STALE FIX REJECTED (fresh LLM call)", report)

    assert report.reused_fix is False, (
        "Run 3 must NOT reuse a fix whose tree hash does not match current repo"
    )
    assert provider.call_count >= 1, (
        f"Expected at least 1 LLM call after tree change; got {provider.call_count}"
    )
    # The run should still succeed (FakeProvider returns correct patch)
    assert report.success, f"Run 3 must succeed. error={report.error!r}"
    assert report.verification is not None
    assert report.verification.passed


# ---------------------------------------------------------------------------
# Isolated-workspace tests (each gets its own tmp_path copy)
# ---------------------------------------------------------------------------

@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    """Per-test isolated copy of spring-calc."""
    return _copy_fixture(tmp_path)


@pytest.mark.integration
def test_repo_analysis_builds_context_and_tree_hash(workspace: Path) -> None:
    """prepare_repo() indexes all Java files and produces a stable SHA-256 tree hash."""
    ctx = prepare_repo(workspace)

    assert ctx.tree_hash, "tree_hash must be non-empty"
    assert len(ctx.tree_hash) == 64, "tree_hash is SHA-256 hex (64 chars)"
    assert len(ctx.source_files()) >= 2, "At least CalculatorService + CalcApplication"

    paths = [f.rel_path for f in ctx.source_files()]
    assert any("CalculatorService.java" in p for p in paths)
    assert any("CalcApplication.java" in p for p in paths)

    # tree_hash is stable for the same file contents
    ctx2 = prepare_repo(workspace)
    assert ctx.tree_hash == ctx2.tree_hash, "tree_hash must be deterministic"


@pytest.mark.integration
def test_fix_store_compatibility_checks_tree_hash(workspace: Path) -> None:
    """FixStore.check_compatibility() distinguishes COMPATIBLE from TREE_MISMATCH.

    This confirms that matching issue fingerprints alone are not sufficient —
    the repo tree hash is also required.
    """
    from agent.engineering_knowledge.fix_store import make_verified_fix

    ctx = prepare_repo(workspace)
    fp = fingerprint_from_raw(
        repo_full_name=REPO_NAME, issue_number=0,
        title=ISSUE_TITLE, body=ISSUE_BODY, labels=[],
    )

    # Store a fix with the current tree hash
    fix = make_verified_fix(
        fingerprint_hex=fp.full_hex,
        repo_full_name=REPO_NAME,
        issue_number=0,
        patch_unified_diff=CORRECT_DIFF,
        test_file_contents={},
        repo_tree_hash=ctx.tree_hash,
    )
    store = FixStore()
    store.save(fix)

    # Same fingerprint + same tree → COMPATIBLE
    assert store.check_compatibility(fp.full_hex, REPO_NAME, ctx.tree_hash).status == CompatibilityStatus.COMPATIBLE

    # Same fingerprint + different tree → TREE_MISMATCH
    assert store.check_compatibility(fp.full_hex, REPO_NAME, "a" * 64).status == CompatibilityStatus.TREE_MISMATCH

    # Same fingerprint + correct tree + wrong repo → WRONG_REPO
    assert store.check_compatibility(fp.full_hex, "other/repo", ctx.tree_hash).status == CompatibilityStatus.WRONG_REPO

    # Fix stored without tree hash → UNVERIFIED_COMPAT
    fix_no_tree = make_verified_fix(
        fingerprint_hex=fp.full_hex,
        repo_full_name=REPO_NAME,
        issue_number=0,
        patch_unified_diff=CORRECT_DIFF,
        test_file_contents={},
        repo_tree_hash=None,
    )
    store2 = FixStore()
    store2.save(fix_no_tree)
    assert store2.check_compatibility(fp.full_hex, REPO_NAME, ctx.tree_hash).status == CompatibilityStatus.UNVERIFIED_COMPAT


@pytest.mark.integration
def test_budget_exceeded_emits_run_failed_immediately(workspace: Path) -> None:
    """BudgetExceededError on first LLM call → RunFailed event, success=False."""
    provider = BudgetProvider([])
    orch = _make_orch(provider)
    inp = RunInput(
        repo_path=workspace,
        issue_title=ISSUE_TITLE,
        issue_body=ISSUE_BODY,
        repo_full_name=REPO_NAME,
        run_id="budget-test",
    )

    report = asyncio.run(orch.run(inp))

    assert report.success is False
    failed = [e for e in report.events if isinstance(e, RunFailed)]
    assert len(failed) == 1
    assert "Budget" in failed[0].reason or "budget" in failed[0].reason.lower()
    # Stage should not have progressed past root_cause
    assert report.stage_reached in (
        WorkflowStage.intake,
        WorkflowStage.repo_analysis,
        WorkflowStage.root_cause,
    )


@pytest.mark.integration
def test_bad_patch_fails_at_patch_stage(workspace: Path) -> None:
    """A diff that doesn't apply → run fails at patch stage after MAX_REPAIR_ATTEMPTS."""
    bad_diag = json.dumps({
        "root_cause": "bogus",
        "affected_files": [],
        "fix_description": "bogus fix",
        "unified_diff": (
            "--- a/no/such/file.java\n+++ b/no/such/file.java\n"
            "@@ -1 +1 @@\n-old line\n+new line\n"
        ),
        "confidence": "low",
    })
    # Provider returns the same bad patch for all 3 repair attempts
    provider = FakeProvider([bad_diag, bad_diag, bad_diag])
    orch = _make_orch(provider)
    inp = RunInput(
        repo_path=workspace,
        issue_title=ISSUE_TITLE,
        issue_body=ISSUE_BODY,
        repo_full_name=REPO_NAME,
        run_id="badpatch-test",
    )

    report = asyncio.run(orch.run(inp))

    assert report.success is False
    failed = [e for e in report.events if isinstance(e, RunFailed)]
    assert len(failed) == 1
    assert failed[0].stage == WorkflowStage.patch


@pytest.mark.integration
def test_checkpoint_lifecycle(workspace: Path) -> None:
    """CheckpointManager progresses through stages and lands in completed or abandoned."""
    checkpoints = CheckpointManager()
    provider = FakeProvider([DIAGNOSIS_JSON, TEST_GEN_JSON])
    orch = _make_orch(provider, checkpoints=checkpoints)
    inp = RunInput(
        repo_path=workspace,
        issue_title=ISSUE_TITLE,
        issue_body=ISSUE_BODY,
        repo_full_name=REPO_NAME,
        run_id="ckpt-test",
    )

    asyncio.run(orch.run(inp))

    status = checkpoints.status("ckpt-test")
    assert status in (RunStatus.completed, RunStatus.abandoned), (
        f"Expected completed or abandoned, got {status!r}"
    )
    # At least the intake checkpoint was created
    record = checkpoints.load("ckpt-test")
    assert record is not None


@pytest.mark.integration
def test_usage_tracker_records_all_llm_calls(workspace: Path) -> None:
    """UsageTracker records every call with model, tokens, and cost."""
    usage = UsageTracker()
    provider = FakeProvider([DIAGNOSIS_JSON, TEST_GEN_JSON])
    orch = _make_orch(provider, usage=usage)
    inp = RunInput(
        repo_path=workspace,
        issue_title=ISSUE_TITLE,
        issue_body=ISSUE_BODY,
        repo_full_name=REPO_NAME,
        run_id="usage-test",
    )

    report = asyncio.run(orch.run(inp))
    summary = usage.run_summary("usage-test")

    llm_events = [e for e in report.events if isinstance(e, LLMCallMade)]
    assert len(llm_events) >= 1
    assert summary.call_count == len(llm_events)
    assert summary.total_prompt_tokens > 0
    assert summary.total_completion_tokens > 0
    assert summary.total_cost_usd > 0.0

    # Each recorded call has a model and stage
    for rec in usage.all_records():
        if rec.run_id == "usage-test":
            assert rec.model == "muse-spark-1.3"
            assert rec.stage != ""
            assert rec.cost_usd > 0.0


@pytest.mark.integration
def test_cache_hit_avoids_second_llm_call(workspace: Path) -> None:
    """CacheInvalidator: a second identical call within TTL returns a cache hit, no new LLM call."""
    cache = CacheInvalidator()

    # Run 1: fills cache
    provider1 = FakeProvider([DIAGNOSIS_JSON, TEST_GEN_JSON])
    orch1 = _make_orch(provider1, cache=cache)
    inp1 = RunInput(
        repo_path=workspace,
        issue_title=ISSUE_TITLE,
        issue_body=ISSUE_BODY,
        repo_full_name=REPO_NAME,
        run_id="cache-hit-run1",
    )
    asyncio.run(orch1.run(inp1))
    calls_run1 = provider1.call_count

    _reset_repo(workspace)

    # Run 2: same fingerprint + same tree_hash → cache hit for root_cause stage.
    # If test-gen response was also cached on Run 1, run2 may make 0 calls.
    # The test verifies that calls_run2 < calls_run1 (i.e., cache was useful).
    provider2 = FakeProvider([TEST_GEN_JSON])  # only test-gen needed if not cached
    orch2 = _make_orch(provider2, cache=cache)
    inp2 = RunInput(
        repo_path=workspace,
        issue_title=ISSUE_TITLE,
        issue_body=ISSUE_BODY,
        repo_full_name=REPO_NAME,
        run_id="cache-hit-run2",
    )
    report2 = asyncio.run(orch2.run(inp2))

    # Run 1 made 2 calls; run 2 must make fewer (0 or 1 depending on test-gen cache).
    assert calls_run1 == 2
    assert provider2.call_count < calls_run1, (
        f"Run 2 must make fewer LLM calls than run 1 ({calls_run1}) thanks to "
        f"CacheInvalidator; made {provider2.call_count}"
    )
    assert report2.success


@pytest.mark.integration
def test_sandbox_exit_codes_and_output_captured(workspace: Path) -> None:
    """sandbox.run_maven_tests() returns real exit code and captures stdout/stderr.

    SECURITY: mvn runs on the host in a subprocess.  The workspace is an
    isolated copy so mutations here don't affect the shared fixture.
    """
    from agent.sandbox import run_maven_tests

    # Before fix: expect failure (divide_nullDivisor test fails)
    result_before = run_maven_tests(workspace, timeout_seconds=120)
    assert result_before.exit_code != 0, (
        "Before fix, mvn test must fail (divide_nullDivisor_throwsIllegalArgument fails)"
    )
    assert result_before.tests_run >= 5
    assert result_before.tests_failed >= 1
    assert "Tests run" in result_before.stdout
    assert not result_before.timed_out

    # Apply the correct patch
    from agent.patch import apply_patch
    patch_result = apply_patch(workspace, CORRECT_DIFF)
    assert patch_result.success, f"Patch must apply cleanly: {patch_result.error}"

    # After fix: all tests pass
    result_after = run_maven_tests(workspace, timeout_seconds=120)
    assert result_after.exit_code == 0, "After fix, mvn test must pass"
    assert result_after.tests_failed == 0
    assert result_after.tests_errors == 0


@pytest.mark.integration
def test_cache_invalidated_after_tree_change(workspace: Path) -> None:
    """After a committed source change the fix_store returns TREE_MISMATCH and a fresh LLM call is made."""
    cache = CacheInvalidator()
    fix_store = FixStore()

    # Run 1: fills fix_store and cache
    provider1 = FakeProvider([DIAGNOSIS_JSON, TEST_GEN_JSON])
    orch1 = _make_orch(provider1, fix_store=fix_store, cache=cache)
    asyncio.run(orch1.run(RunInput(
        repo_path=workspace, issue_title=ISSUE_TITLE, issue_body=ISSUE_BODY,
        repo_full_name=REPO_NAME, run_id="inval-run1",
    )))

    _reset_repo(workspace)

    # Mutate + commit
    svc = workspace / "src/main/java/com/example/calc/CalculatorService.java"
    svc.write_text(svc.read_text() + "\n// sentinel change\n", encoding="utf-8")
    env = {
        **__import__("os").environ,
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
    }
    subprocess.run(["git", "add", str(svc)], cwd=str(workspace), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "sentinel"], cwd=str(workspace), check=True,
                   capture_output=True, env=env)

    # Run 2: new tree hash → TREE_MISMATCH → fresh analysis
    provider2 = FakeProvider([DIAGNOSIS_JSON, TEST_GEN_JSON])
    orch2 = _make_orch(provider2, fix_store=fix_store, cache=cache)
    report2 = asyncio.run(orch2.run(RunInput(
        repo_path=workspace, issue_title=ISSUE_TITLE, issue_body=ISSUE_BODY,
        repo_full_name=REPO_NAME, run_id="inval-run2",
    )))

    assert provider2.call_count >= 1, (
        "After tree change, stale fix must be rejected and LLM must be called again"
    )
    assert report2.reused_fix is False
