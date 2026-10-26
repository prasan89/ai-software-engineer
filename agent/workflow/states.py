"""Typed workflow stages, run states, and structured events.

WorkflowStage defines the ordered stages of a single agent run.
WorkflowState is the complete mutable state of an in-progress run.
RunEvent is the sealed union of all structured events emitted during a run;
  consumers (API, logging, UI) receive a stream of RunEvents.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


class WorkflowStage(StrEnum):
    """Ordered stages of one agent run. Order must not change (checkpoint compat)."""

    intake = "intake"  # Fetch and validate the GitHub issue
    repo_analysis = "repo_analysis"  # Clone repo, build file index, find context
    root_cause = "root_cause"  # LLM: identify root cause and affected files
    patch = "patch"  # LLM: generate and apply unified diff
    test_generation = "test_generation"  # LLM: generate regression tests
    sandbox_execution = "sandbox_execution"  # Docker: run tests in isolation
    verification = "verification"  # Evaluate pass/fail; decide to accept or retry
    done = "done"  # Terminal success
    failed = "failed"  # Terminal failure


TERMINAL_STAGES = frozenset({WorkflowStage.done, WorkflowStage.failed})


# ---------------------------------------------------------------------------
# Run state
# ---------------------------------------------------------------------------


class WorkflowState(BaseModel):
    run_id: str
    stage: WorkflowStage = WorkflowStage.intake
    issue_fingerprint: str | None = None
    repo_full_name: str | None = None
    issue_number: int | None = None
    repo_tree_hash: str | None = None
    patch_diff: str | None = None
    test_files: dict[str, str] = Field(default_factory=dict)
    verification_passed: bool | None = None
    reused_fix: bool = False
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def is_terminal(self) -> bool:
        return self.stage in TERMINAL_STAGES


# ---------------------------------------------------------------------------
# Run events (structured, emitted to caller / SSE stream)
# ---------------------------------------------------------------------------


class StageStarted(BaseModel):
    event: Literal["stage_started"] = "stage_started"
    run_id: str
    stage: WorkflowStage


class StageCompleted(BaseModel):
    event: Literal["stage_completed"] = "stage_completed"
    run_id: str
    stage: WorkflowStage


class CacheHit(BaseModel):
    event: Literal["cache_hit"] = "cache_hit"
    run_id: str
    stage: WorkflowStage
    fingerprint_short: str


class FixReused(BaseModel):
    event: Literal["fix_reused"] = "fix_reused"
    run_id: str
    fingerprint_short: str
    tokens_saved: int = 0
    cost_saved_usd: float = 0.0


class LLMCallMade(BaseModel):
    event: Literal["llm_call"] = "llm_call"
    run_id: str
    stage: WorkflowStage
    model: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    cost_usd: float


class VerificationResult(BaseModel):
    event: Literal["verification_result"] = "verification_result"
    run_id: str
    passed: bool
    tests_run: int
    tests_passed: int
    tests_failed: int


class RunFailed(BaseModel):
    event: Literal["run_failed"] = "run_failed"
    run_id: str
    stage: WorkflowStage
    reason: str


class RunCompleted(BaseModel):
    event: Literal["run_completed"] = "run_completed"
    run_id: str
    reused_fix: bool
    total_cost_usd: float


RunEvent = (
    StageStarted
    | StageCompleted
    | CacheHit
    | FixReused
    | LLMCallMade
    | VerificationResult
    | RunFailed
    | RunCompleted
)
