"""End-to-end demo: real Meta API + Docker sandbox on spring-calc fixture.

Usage:
    python scripts/e2e_demo.py [--scenario A|B|C] [--all]

Scenarios:
    A  First run: fresh analysis, patch, Docker verification
    B  Reuse:     same issue, same tree — reuse stored fix without LLM call
    C  Stale:     modify source file, confirm cache miss, fresh analysis

Requires:
    META_API_KEY env var
    Docker daemon with maven:3.9.9-eclipse-temurin-21-alpine image
    ~/.m2/repository pre-seeded (mvn dependency:go-offline)
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from agent.cost_optimizer.metrics import CacheMetrics
from agent.cost_optimizer.usage import UsageTracker
from agent.engineering_knowledge.cache import CacheInvalidator
from agent.engineering_knowledge.checkpoint import CheckpointManager
from agent.engineering_knowledge.fix_store import FixStore
from agent.providers.meta import MUSE_SPARK_1_3, MetaProvider
from agent.sandbox.types import SandboxConfig
from agent.workflow.orchestrator import RunInput, WorkflowOrchestrator
from agent.workflow.states import FixReused, LLMCallMade, RunFailed

FIXTURE_SRC = Path(__file__).parents[1] / "fixtures" / "spring-calc"
M2_REPO = Path(os.environ.get("MAVEN_LOCAL_REPO", Path.home() / ".m2" / "repository"))

ISSUE_TITLE = "divide() throws NullPointerException for null divisor"
ISSUE_BODY = (FIXTURE_SRC / "ISSUE.md").read_text() if (FIXTURE_SRC / "ISSUE.md").exists() else (
    "When divide() is called with a null divisor, it throws NPE instead of "
    "IllegalArgumentException."
)
REPO_NAME = "local/spring-calc"


def _copy_fixture(dest: Path) -> Path:
    """Copy fixture to a fresh isolated directory."""
    repo_dir = dest / "spring-calc"
    shutil.copytree(
        FIXTURE_SRC,
        repo_dir,
        ignore=shutil.ignore_patterns(".git", "target"),
    )
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "demo", "GIT_AUTHOR_EMAIL": "demo@example.com",
        "GIT_COMMITTER_NAME": "demo", "GIT_COMMITTER_EMAIL": "demo@example.com",
    }
    subprocess.run(["git", "init", "-b", "main"], cwd=str(repo_dir), check=True, capture_output=True, env=env)
    subprocess.run(["git", "add", "-A"], cwd=str(repo_dir), check=True, capture_output=True, env=env)
    subprocess.run(["git", "commit", "-m", "feat: initial"], cwd=str(repo_dir), check=True, capture_output=True, env=env)
    return repo_dir


def _make_orchestrator(
    fix_store: FixStore,
    cache: CacheInvalidator,
    metrics: CacheMetrics,
) -> WorkflowOrchestrator:
    if not os.environ.get("META_API_KEY"):
        print("ERROR: META_API_KEY not set. Source ~/.zshrc or set the variable.")
        sys.exit(1)
    if not M2_REPO.is_dir():
        print(f"ERROR: Maven local repo not found at {M2_REPO}")
        print("Run: mvn dependency:go-offline -f fixtures/spring-calc/pom.xml")
        sys.exit(1)

    provider = MetaProvider(model=MUSE_SPARK_1_3)
    config = SandboxConfig(
        timeout_seconds=300,
        maven_local_repo=M2_REPO,
    )
    return WorkflowOrchestrator(
        provider=provider,
        fix_store=fix_store,
        cache=cache,
        checkpoint_manager=CheckpointManager(),
        usage_tracker=UsageTracker(),
        metrics=metrics,
        sandbox_config=config,
    )


def _print_report(label: str, report: object, metrics: CacheMetrics) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    print(f"  run_id:        {report.run_id}")  # type: ignore[attr-defined]
    print(f"  success:       {report.success}")  # type: ignore[attr-defined]
    print(f"  reused_fix:    {report.reused_fix}")  # type: ignore[attr-defined]
    print(f"  elapsed:       {report.elapsed_seconds:.1f}s")  # type: ignore[attr-defined]

    llm_events = [e for e in report.events if isinstance(e, LLMCallMade)]  # type: ignore[attr-defined]
    reuse_events = [e for e in report.events if isinstance(e, FixReused)]  # type: ignore[attr-defined]
    failed_events = [e for e in report.events if isinstance(e, RunFailed)]  # type: ignore[attr-defined]

    print(f"  llm_calls:     {len(llm_events)}")
    if llm_events:
        for ev in llm_events:
            print(f"    model={ev.model}  prompt={ev.prompt_tokens}  "
                  f"completion={ev.completion_tokens}  cost=${ev.cost_usd:.6f}")
    if reuse_events:
        print(f"  fix_reused:    YES (tokens_saved={reuse_events[0].tokens_saved})")
    if failed_events:
        print(f"  FAILED:        {failed_events[0].reason}")

    usage = report.usage_summary  # type: ignore[attr-defined]
    print(f"  total_tokens:  {usage.get('total_prompt_tokens', 0) + usage.get('total_completion_tokens', 0)}")
    print(f"  total_cost:    ${usage.get('total_cost_usd', 0.0):.6f}")

    v = report.verification  # type: ignore[attr-defined]
    if v:
        print(f"  verification:  passed={v.passed}  tests_run={v.tests_run}  "
              f"tests_failed={v.tests_failed}  sandbox_unavailable={v.sandbox_unavailable}")
    print(f"  cache_hits:    {metrics.cache_hits}  misses={metrics.cache_misses}  "
          f"reuses={metrics.verified_fix_reuses}")

    if report.unified_diff:  # type: ignore[attr-defined]
        print("\n  PATCH:")
        for line in report.unified_diff.splitlines()[:30]:  # type: ignore[attr-defined]
            print(f"    {line}")
    print()


async def scenario_a(fix_store: FixStore, cache: CacheInvalidator, metrics: CacheMetrics, tmp: Path) -> None:
    """First run: fresh LLM analysis + Docker verification."""
    print("\n>>> SCENARIO A: First run (fresh analysis)")
    repo = _copy_fixture(tmp / "scenario_a")
    orch = _make_orchestrator(fix_store, cache, metrics)
    inp = RunInput(
        repo_path=repo,
        issue_title=ISSUE_TITLE,
        issue_body=ISSUE_BODY,
        repo_full_name=REPO_NAME,
        run_id="sprint2-A",
    )
    report = await orch.run(inp)
    _print_report("SCENARIO A: First Run", report, metrics)

    if not report.success:
        print("  SCENARIO A FAILED — check output above.")
        return

    print("  >>> SCENARIO A: PASS — fix verified inside Docker")
    # Reset fixture working tree to original buggy state
    subprocess.run(
        ["git", "checkout", "HEAD", "--", "."],
        cwd=str(repo), check=False, capture_output=True,
    )


async def scenario_b(fix_store: FixStore, cache: CacheInvalidator, metrics: CacheMetrics, tmp: Path) -> None:
    """Reuse: same issue and tree — no LLM call, fix reused from store."""
    print("\n>>> SCENARIO B: Fix reuse (same issue, same tree)")
    repo = _copy_fixture(tmp / "scenario_b")
    orch = _make_orchestrator(fix_store, cache, metrics)
    inp = RunInput(
        repo_path=repo,
        issue_title=ISSUE_TITLE,
        issue_body=ISSUE_BODY,
        repo_full_name=REPO_NAME,
        run_id="sprint2-B",
    )
    report = await orch.run(inp)
    _print_report("SCENARIO B: Fix Reuse", report, metrics)

    llm_events = [e for e in report.events if isinstance(e, LLMCallMade)]
    reuse_events = [e for e in report.events if isinstance(e, FixReused)]

    if reuse_events and len(llm_events) == 0:
        print("  >>> SCENARIO B: PASS — fix reused, 0 LLM calls")
    elif reuse_events:
        print(f"  >>> SCENARIO B: PARTIAL — fix reused but had {len(llm_events)} LLM calls")
    else:
        print("  >>> SCENARIO B: EXPECTED reuse not triggered (run scenario A first)")


async def scenario_c(fix_store: FixStore, cache: CacheInvalidator, metrics: CacheMetrics, tmp: Path) -> None:
    """Stale cache: modify source file, confirm tree_hash changes, require fresh analysis."""
    print("\n>>> SCENARIO C: Stale cache (modified source file)")
    repo = _copy_fixture(tmp / "scenario_c")

    # Modify a source file to change tree_hash
    svc = repo / "src/main/java/com/example/calc/CalculatorService.java"
    original = svc.read_text()
    svc.write_text(original + "\n    // scenario C: source modified\n")
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "demo", "GIT_AUTHOR_EMAIL": "demo@example.com",
        "GIT_COMMITTER_NAME": "demo", "GIT_COMMITTER_EMAIL": "demo@example.com",
    }
    subprocess.run(["git", "add", str(svc)], cwd=str(repo), check=True, capture_output=True, env=env)
    subprocess.run(
        ["git", "commit", "-m", "test: stale cache simulation"],
        cwd=str(repo), check=True, capture_output=True, env=env,
    )
    print("  Source file modified + committed — tree_hash will differ from run A")

    orch = _make_orchestrator(fix_store, cache, metrics)
    inp = RunInput(
        repo_path=repo,
        issue_title=ISSUE_TITLE,
        issue_body=ISSUE_BODY,
        repo_full_name=REPO_NAME,
        run_id="sprint2-C",
    )
    report = await orch.run(inp)
    _print_report("SCENARIO C: Stale Cache", report, metrics)

    llm_events = [e for e in report.events if isinstance(e, LLMCallMade)]
    reuse_events = [e for e in report.events if isinstance(e, FixReused)]

    if len(llm_events) >= 1 and not reuse_events:
        print("  >>> SCENARIO C: PASS — stale cache rejected, fresh LLM analysis ran")
    elif reuse_events:
        print("  >>> SCENARIO C: UNEXPECTED — stale fix was reused (cache invalidation broken)")
    else:
        print(f"  >>> SCENARIO C: llm_calls={len(llm_events)} reuse_events={len(reuse_events)}")


async def main() -> None:
    args = sys.argv[1:]
    run_all = "--all" in args
    scenarios = []
    for s in ["A", "B", "C"]:
        if run_all or "--scenario" in args:
            idx = args.index("--scenario") if "--scenario" in args else -1
            if idx >= 0 and idx + 1 < len(args) and args[idx + 1] == s:
                scenarios.append(s)
        elif run_all:
            scenarios.append(s)

    if not scenarios:
        # Default: run all three in sequence
        scenarios = ["A", "B", "C"]

    import tempfile
    fix_store = FixStore()
    cache = CacheInvalidator()
    metrics = CacheMetrics()

    with tempfile.TemporaryDirectory(prefix="sprint2_e2e_") as tmp_str:
        tmp = Path(tmp_str)
        if "A" in scenarios:
            await scenario_a(fix_store, cache, metrics, tmp)
        if "B" in scenarios:
            await scenario_b(fix_store, cache, metrics, tmp)
        if "C" in scenarios:
            await scenario_c(fix_store, cache, metrics, tmp)

    print("\nAll scenarios complete.")


if __name__ == "__main__":
    asyncio.run(main())
