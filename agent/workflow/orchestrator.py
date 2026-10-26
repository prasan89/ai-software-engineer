"""End-to-end workflow orchestrator.

Implements the issue → fix → verify pipeline with full cost-optimization
integration: fingerprinting, duplicate detection, fix reuse, cache,
checkpoint persistence, usage accounting, and model routing.

Security: all repository code execution goes through DockerSandbox.
DockerSandbox enforces: --network=none, --read-only root FS, --cap-drop ALL,
--security-opt no-new-privileges, --user 1000:1000, resource limits, and
guaranteed container cleanup. If Docker is unavailable, the run fails closed
with a clear error — there is no fallback to host execution.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from agent.cost_optimizer.metrics import CacheMetrics
from agent.cost_optimizer.router import ModelRouter, classify_complexity
from agent.cost_optimizer.usage import CallRecord, UsageTracker
from agent.engineering_knowledge.cache import CacheInvalidator, CacheKey, CacheStage
from agent.engineering_knowledge.checkpoint import CheckpointManager
from agent.engineering_knowledge.fingerprint import fingerprint_from_raw
from agent.engineering_knowledge.fix_store import (
    CompatibilityStatus,
    FixStore,
    make_verified_fix,
)
from agent.patch import apply_patch
from agent.providers.base import BaseLLMProvider
from agent.providers.types import BudgetExceededError, LLMRequest, Message, Role
from agent.repo import RepoContext, prepare_repo, read_file_content
from agent.sandbox import SandboxResult
from agent.sandbox.docker_executor import DockerNotAvailableError, DockerSandbox
from agent.sandbox.types import SandboxConfig
from agent.verification import interpret_sandbox_result
from agent.workflow.prompts import (
    CACHE_KEY_SYSTEM,
    SYSTEM_PROMPT,
    make_diagnosis_prompt,
    make_test_generation_prompt,
)
from agent.workflow.schemas import DiagnosisResponse, TestGenerationResponse
from agent.workflow.states import (
    CacheHit,
    FixReused,
    LLMCallMade,
    RunCompleted,
    RunEvent,
    RunFailed,
    StageCompleted,
    StageStarted,
    VerificationResult,
    WorkflowStage,
    WorkflowState,
)

logger = logging.getLogger(__name__)
_M = TypeVar("_M")

MAX_REPAIR_ATTEMPTS = 3  # max patch-then-verify cycles


@dataclass
class RunInput:
    """Input to a single agent run."""

    repo_path: str | Path
    issue_title: str
    issue_body: str
    issue_number: int = 0
    repo_full_name: str = "local/fixture"
    issue_labels: list[str] = field(default_factory=list)
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    # Token/cost budget — enforced by the provider adapter
    budget_usd: float | None = None
    # Override model routing
    force_model: str | None = None


@dataclass
class RunReport:
    """Complete result of one agent run."""

    run_id: str
    success: bool
    stage_reached: WorkflowStage
    reused_fix: bool
    root_cause: str
    fix_description: str
    unified_diff: str
    verification: VerificationResult | None
    events: list[RunEvent]
    metrics: CacheMetrics
    usage_summary: dict[str, Any]
    error: str
    sandbox_stdout: str
    sandbox_stderr: str
    elapsed_seconds: float


class WorkflowOrchestrator:
    """Runs the full issue-to-verified-fix pipeline.

    Dependencies are injected so every component can be replaced or mocked.

    Args:
        sandbox_config: Docker isolation parameters. Defaults are secure.
            Injected for testing; production code should use the default.
        sandbox_runner: Optional callable ``(repo_path) -> SandboxResult``.
            When provided, replaces the default ``DockerSandbox`` execution.
            Use ONLY for integration tests against trusted fixtures (inject
            ``LocalSandbox(trusted=True).run_maven_tests``). Never pass an
            arbitrary callable for untrusted repositories.
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        fix_store: FixStore | None = None,
        cache: CacheInvalidator | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        usage_tracker: UsageTracker | None = None,
        model_router: ModelRouter | None = None,
        metrics: CacheMetrics | None = None,
        sandbox_config: SandboxConfig | None = None,
        sandbox_runner: Callable[[str | Path], SandboxResult] | None = None,
    ) -> None:
        self._provider = provider
        self._fix_store = fix_store if fix_store is not None else FixStore()
        self._cache = cache if cache is not None else CacheInvalidator()
        self._checkpoints = checkpoint_manager if checkpoint_manager is not None else CheckpointManager()
        self._usage = usage_tracker if usage_tracker is not None else UsageTracker()
        self._router = model_router if model_router is not None else ModelRouter()
        self._metrics = metrics if metrics is not None else CacheMetrics()
        self._sandbox_config = sandbox_config
        self._sandbox_runner = sandbox_runner

    async def run(self, inp: RunInput) -> RunReport:
        """Execute the full pipeline and return a structured report."""
        import time

        start = time.monotonic()
        state = WorkflowState(run_id=inp.run_id)
        events: list[RunEvent] = []

        def emit(event: RunEvent) -> None:
            events.append(event)
            logger.info("event=%s run_id=%s", event.event, inp.run_id)

        try:
            report = await self._execute(inp, state, emit, events)
        except BudgetExceededError as exc:
            emit(RunFailed(run_id=inp.run_id, stage=state.stage, reason=str(exc)))
            report = self._make_report(inp, state, events, str(exc), "", None, None, start)

        report.elapsed_seconds = round(time.monotonic() - start, 2)
        return report

    async def _execute(
        self,
        inp: RunInput,
        state: WorkflowState,
        emit: Any,
        events: list[RunEvent],
    ) -> RunReport:
        import time

        start = time.monotonic()

        # ------------------------------------------------------------------ #
        # Stage 1: INTAKE — fingerprint and duplicate detection               #
        # ------------------------------------------------------------------ #
        emit(StageStarted(run_id=inp.run_id, stage=WorkflowStage.intake))
        state.stage = WorkflowStage.intake

        fp = fingerprint_from_raw(
            repo_full_name=inp.repo_full_name,
            issue_number=inp.issue_number,
            title=inp.issue_title,
            body=inp.issue_body,
            labels=inp.issue_labels,
        )
        state.issue_fingerprint = fp.full_hex
        state.repo_full_name = inp.repo_full_name
        state.issue_number = inp.issue_number

        self._checkpoints.create(
            inp.run_id, WorkflowStage.intake, {"fingerprint": fp.full_hex}
        )
        emit(StageCompleted(run_id=inp.run_id, stage=WorkflowStage.intake))

        # ------------------------------------------------------------------ #
        # Stage 2: REPO ANALYSIS                                               #
        # ------------------------------------------------------------------ #
        emit(StageStarted(run_id=inp.run_id, stage=WorkflowStage.repo_analysis))
        state.stage = WorkflowStage.repo_analysis

        repo = prepare_repo(inp.repo_path)
        state.repo_tree_hash = repo.tree_hash

        # Check for compatible verified fix before any LLM call
        compat = self._fix_store.check_compatibility(
            fp.full_hex, inp.repo_full_name, repo.tree_hash
        )
        if compat.status == CompatibilityStatus.COMPATIBLE and compat.fix is not None:
            fix = compat.fix
            emit(
                FixReused(
                    run_id=inp.run_id,
                    fingerprint_short=fp.short_hex,
                    tokens_saved=0,
                    cost_saved_usd=0.0,
                )
            )
            self._metrics.record_fix_reuse()
            self._checkpoints.complete(inp.run_id)

            # Re-run verification to confirm the fix still applies
            patch_result = apply_patch(inp.repo_path, fix.patch_unified_diff)
            if not patch_result.success:
                # Fix no longer applies — fall through to fresh analysis
                logger.warning("Cached fix no longer applies cleanly; running fresh analysis")
            else:
                sandbox = self._run_sandbox(inp.repo_path)
                reuse_verification = interpret_sandbox_result(inp.run_id, sandbox)
                emit(reuse_verification)
                if reuse_verification.passed:
                    emit(
                        RunCompleted(
                            run_id=inp.run_id,
                            reused_fix=True,
                            total_cost_usd=self._usage.global_cost_usd(),
                        )
                    )
                else:
                    reason = (
                        "Sandbox unavailable" if reuse_verification.sandbox_unavailable
                        else "Verification failed for reused fix"
                    )
                    emit(RunFailed(run_id=inp.run_id, stage=WorkflowStage.sandbox_execution, reason=reason))
                return self._make_report(
                    inp, state, events, "", fix.patch_unified_diff, fix, sandbox, start,
                    root_cause=f"[REUSED] {fix.metadata.get('root_cause', '')}",
                    fix_description=fix.metadata.get("fix_description", ""),
                    verification=reuse_verification,
                    reused=True,
                )

        emit(StageCompleted(run_id=inp.run_id, stage=WorkflowStage.repo_analysis))
        self._metrics.record_cache_miss()

        # ------------------------------------------------------------------ #
        # Stage 3: ROOT CAUSE ANALYSIS                                         #
        # ------------------------------------------------------------------ #
        emit(StageStarted(run_id=inp.run_id, stage=WorkflowStage.root_cause))
        state.stage = WorkflowStage.root_cause

        # Check analysis cache before calling the model
        cache_key = CacheKey(
            fingerprint_hex=fp.full_hex,
            tree_hash=repo.tree_hash,
            stage=CacheStage.root_cause,
            model_id=self._provider.model_id(),
        )
        cached = self._cache.lookup(cache_key)
        diagnosis: DiagnosisResponse
        if cached.hit and cached.entry:
            emit(CacheHit(run_id=inp.run_id, stage=WorkflowStage.root_cause, fingerprint_short=fp.short_hex))
            self._metrics.record_cache_hit()
            diagnosis = DiagnosisResponse.model_validate(cached.entry.value)
        else:
            self._metrics.record_cache_miss()
            file_contents = self._collect_context(repo)
            issue_text = f"{inp.issue_title}\n\n{inp.issue_body}"
            complexity = classify_complexity(len(repo.source_files()))
            model = inp.force_model or self._router.select(complexity)

            diagnosis = await self._call_diagnosis(
                inp.run_id, model, issue_text, file_contents, emit
            )
            self._cache.store(cache_key, diagnosis.model_dump())

        self._checkpoints.advance(
            inp.run_id, WorkflowStage.root_cause,
            {"root_cause": diagnosis.root_cause, "affected": diagnosis.affected_files}
        )
        emit(StageCompleted(run_id=inp.run_id, stage=WorkflowStage.root_cause))

        # ------------------------------------------------------------------ #
        # Stage 4 + 5: PATCH + TEST GENERATION + VERIFICATION (with retries) #
        # ------------------------------------------------------------------ #
        sandbox_result: SandboxResult | None = None
        verification: VerificationResult | None = None
        test_response: TestGenerationResponse | None = None

        for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):
            emit(StageStarted(run_id=inp.run_id, stage=WorkflowStage.patch))
            state.stage = WorkflowStage.patch

            patch_result = apply_patch(inp.repo_path, diagnosis.unified_diff)
            if not patch_result.success:
                error_msg = f"Patch attempt {attempt} failed: {patch_result.error}"
                logger.warning(error_msg)
                if attempt == MAX_REPAIR_ATTEMPTS:
                    emit(RunFailed(run_id=inp.run_id, stage=WorkflowStage.patch, reason=error_msg))
                    return self._make_report(inp, state, events, error_msg, diagnosis.unified_diff, None, None, start)
                continue

            emit(StageCompleted(run_id=inp.run_id, stage=WorkflowStage.patch))
            state.patch_diff = diagnosis.unified_diff

            # Generate regression test
            emit(StageStarted(run_id=inp.run_id, stage=WorkflowStage.test_generation))
            state.stage = WorkflowStage.test_generation

            test_cache_key = CacheKey(
                fingerprint_hex=fp.full_hex,
                tree_hash=repo.tree_hash,
                stage=CacheStage.tests,
                model_id=self._provider.model_id(),
            )
            cached_test = self._cache.lookup(test_cache_key)
            if cached_test.hit and cached_test.entry:
                emit(CacheHit(run_id=inp.run_id, stage=WorkflowStage.test_generation, fingerprint_short=fp.short_hex))
                self._metrics.record_cache_hit()
                test_response = TestGenerationResponse.model_validate(cached_test.entry.value)
            else:
                self._metrics.record_cache_miss()
                test_response = await self._call_test_generation(
                    inp.run_id, self._provider.model_id(), inp.issue_title + "\n" + inp.issue_body,
                    diagnosis, repo, emit
                )
                self._cache.store(test_cache_key, test_response.model_dump())

            emit(StageCompleted(run_id=inp.run_id, stage=WorkflowStage.test_generation))

            # Apply the generated test to the repo
            if test_response:
                self._inject_test_method(repo, test_response)

            # Run tests
            emit(StageStarted(run_id=inp.run_id, stage=WorkflowStage.sandbox_execution))
            state.stage = WorkflowStage.sandbox_execution

            sandbox_result = self._run_sandbox(inp.repo_path)
            emit(StageCompleted(run_id=inp.run_id, stage=WorkflowStage.sandbox_execution))

            # Verify
            emit(StageStarted(run_id=inp.run_id, stage=WorkflowStage.verification))
            state.stage = WorkflowStage.verification
            verification = interpret_sandbox_result(inp.run_id, sandbox_result)
            emit(verification)
            emit(StageCompleted(run_id=inp.run_id, stage=WorkflowStage.verification))

            if verification.passed:
                break

            logger.info("Verification attempt %d failed (%d tests failed); retrying", attempt, verification.tests_failed)

        if verification and verification.passed:
            state.verification_passed = True
            state.stage = WorkflowStage.done

            # Store as a verified fix for future reuse
            verified_fix = make_verified_fix(
                fingerprint_hex=fp.full_hex,
                repo_full_name=inp.repo_full_name,
                issue_number=inp.issue_number,
                patch_unified_diff=diagnosis.unified_diff,
                test_file_contents={},
                repo_tree_hash=repo.tree_hash,
                metadata={
                    "root_cause": diagnosis.root_cause,
                    "fix_description": diagnosis.fix_description,
                    "confidence": diagnosis.confidence,
                },
            )
            self._fix_store.save(verified_fix)
            self._checkpoints.complete(inp.run_id)

            emit(RunCompleted(run_id=inp.run_id, reused_fix=False, total_cost_usd=self._usage.global_cost_usd()))
        else:
            state.stage = WorkflowStage.failed
            self._checkpoints.abandon(inp.run_id)
            emit(RunFailed(run_id=inp.run_id, stage=WorkflowStage.verification, reason="Verification failed after all attempts"))

        return self._make_report(
            inp, state, events, state.error or "",
            diagnosis.unified_diff if diagnosis else "",
            None, sandbox_result, start,
            root_cause=diagnosis.root_cause if diagnosis else "",
            fix_description=diagnosis.fix_description if diagnosis else "",
            verification=verification,
        )

    # ------------------------------------------------------------------ #
    # Sandbox execution                                                     #
    # ------------------------------------------------------------------ #

    def _run_sandbox(self, repo_path: str | Path) -> SandboxResult:
        """Run Maven tests in an isolated Docker container.

        When ``sandbox_runner`` was injected at construction time (integration
        tests with trusted fixtures), that callable is used instead. In all
        other cases ``DockerSandbox`` is used and Docker unavailability fails
        closed — there is no fallback to host execution.

        Args:
            repo_path: Path to the Maven project root. Must contain pom.xml.

        Returns:
            SandboxResult. Check .sandbox_error, .timed_out, .oom_killed before
            inspecting exit_code or test counts.
        """
        if self._sandbox_runner is not None:
            return self._sandbox_runner(repo_path)

        sandbox = DockerSandbox(config=self._sandbox_config)
        try:
            return sandbox.run_maven_tests(repo_path)
        except DockerNotAvailableError as exc:
            logger.error("Docker unavailable — run cannot proceed: %s", exc)
            return SandboxResult(
                command=[],
                exit_code=-1,
                stdout="",
                stderr=str(exc),
                elapsed_seconds=0.0,
                sandbox_error=True,
                sandbox_error_message=str(exc),
            )
        except ValueError as exc:
            # pom.xml not found or invalid repo path
            logger.error("Invalid repo path for sandbox: %s", exc)
            return SandboxResult(
                command=[],
                exit_code=-1,
                stdout="",
                stderr=str(exc),
                elapsed_seconds=0.0,
                sandbox_error=True,
                sandbox_error_message=str(exc),
            )

    # ------------------------------------------------------------------ #
    # LLM call helpers                                                      #
    # ------------------------------------------------------------------ #

    async def _call_diagnosis(
        self,
        run_id: str,
        model: str,
        issue_text: str,
        file_contents: dict[str, str],
        emit: Any,
    ) -> DiagnosisResponse:
        request = LLMRequest(
            model_override=model,
            messages=[
                Message(role=Role.system, content=SYSTEM_PROMPT),
                Message(role=Role.user, content=make_diagnosis_prompt(issue_text, file_contents)),
            ],
            max_tokens=2048,
            temperature=0.1,
            prompt_cache_key=CACHE_KEY_SYSTEM,
            prompt_cache_retention="24h",
        )
        response = await self._provider.chat(request)
        self._record_usage(run_id, WorkflowStage.root_cause, response, emit)
        return self._parse_json_response(response.content, DiagnosisResponse)

    async def _call_test_generation(
        self,
        run_id: str,
        model: str,
        issue_text: str,
        diagnosis: DiagnosisResponse,
        repo: RepoContext,
        emit: Any,
    ) -> TestGenerationResponse:
        # Find the primary test file for context
        test_files = [f for f in repo.files if "Test" in f.rel_path and f.rel_path.endswith(".java")]
        existing_test_file = test_files[0].rel_path if test_files else "src/test/java/Test.java"
        existing_test_content = read_file_content(test_files[0]) if test_files else ""

        request = LLMRequest(
            model_override=model,
            messages=[
                Message(role=Role.system, content=SYSTEM_PROMPT),
                Message(
                    role=Role.user,
                    content=make_test_generation_prompt(
                        issue_text,
                        diagnosis.fix_description,
                        diagnosis.unified_diff,
                        existing_test_file,
                        existing_test_content,
                    ),
                ),
            ],
            max_tokens=1024,
            temperature=0.1,
            prompt_cache_key=CACHE_KEY_SYSTEM,
            prompt_cache_retention="24h",
        )
        response = await self._provider.chat(request)
        self._record_usage(run_id, WorkflowStage.test_generation, response, emit)
        return self._parse_json_response(response.content, TestGenerationResponse)

    # ------------------------------------------------------------------ #
    # Helpers                                                               #
    # ------------------------------------------------------------------ #

    def _collect_context(self, repo: RepoContext) -> dict[str, str]:
        """Return {rel_path: content} for relevant source files, respecting token budget."""
        contents: dict[str, str] = {}
        total_bytes = 0
        budget_bytes = 80 * 1024  # rough 20k token budget
        for entry in repo.files:
            if total_bytes >= budget_bytes:
                break
            if entry.language == "java":
                content = read_file_content(entry)
                contents[entry.rel_path] = content
                total_bytes += len(content)
        return contents

    def _record_usage(self, run_id: str, stage: WorkflowStage, response: Any, emit: Any) -> None:
        cost = self._provider.estimated_cost_usd(
            response.usage.prompt_tokens, response.usage.completion_tokens
        )
        record = CallRecord(
            run_id=run_id,
            stage=stage.value,
            model=response.model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            cached_tokens=response.usage.cached_tokens,
            cost_usd=cost,
        )
        self._usage.record(record)
        emit(
            LLMCallMade(
                run_id=run_id,
                stage=stage,
                model=response.model,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                cached_tokens=response.usage.cached_tokens,
                cost_usd=cost,
            )
        )

    @staticmethod
    def _parse_json_response(content: str, model_class: type[_M]) -> _M:
        """Strip markdown fences if present and parse JSON."""
        text = content.strip()
        # Strip ```json ... ``` or ``` ... ``` blocks
        fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
        if fence:
            text = fence.group(1)
        data = json.loads(text)
        return model_class.model_validate(data)  # type: ignore[attr-defined, no-any-return]

    def _inject_test_method(self, repo: RepoContext, test_response: TestGenerationResponse) -> None:
        """Append the generated test method to the first test file found."""
        test_files = [f for f in repo.files if "Test" in f.rel_path and f.rel_path.endswith(".java")]
        if not test_files:
            logger.warning("No test file found to inject generated test into")
            return
        test_path = Path(test_files[0].abs_path)
        content = test_path.read_text(encoding="utf-8")
        # Insert before the last closing brace of the class
        if content.rstrip().endswith("}"):
            new_method = f"\n    @Test\n    void {test_response.test_method_name}() {{\n        {test_response.test_method_body}\n    }}\n"
            updated = content.rstrip()[:-1] + new_method + "}\n"
            test_path.write_text(updated, encoding="utf-8")

    def _make_report(
        self,
        inp: RunInput,
        state: WorkflowState,
        events: list[RunEvent],
        error: str,
        diff: str,
        _fix: Any,
        sandbox: SandboxResult | None,
        start: float,
        root_cause: str = "",
        fix_description: str = "",
        verification: VerificationResult | None = None,
        reused: bool = False,
    ) -> RunReport:
        import time

        summary = self._usage.run_summary(inp.run_id)
        return RunReport(
            run_id=inp.run_id,
            success=state.stage in (WorkflowStage.done,) or (verification is not None and verification.passed),
            stage_reached=state.stage,
            reused_fix=reused,
            root_cause=root_cause,
            fix_description=fix_description,
            unified_diff=diff,
            verification=verification,
            events=events,
            metrics=self._metrics,
            usage_summary={
                "total_prompt_tokens": summary.total_prompt_tokens,
                "total_completion_tokens": summary.total_completion_tokens,
                "cached_tokens": summary.total_cached_tokens,
                "total_cost_usd": round(summary.total_cost_usd, 6),
                "call_count": summary.call_count,
                "cache_savings_usd": round(summary.cache_savings_usd, 6),
            },
            error=error,
            sandbox_stdout=sandbox.stdout if sandbox else "",
            sandbox_stderr=sandbox.stderr if sandbox else "",
            elapsed_seconds=round(time.monotonic() - start, 2),
        )
