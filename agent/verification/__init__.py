"""Verification: interpret sandbox execution results and emit a VerificationResult event.

Outcome mapping
---------------
  sandbox_error=True        → sandbox_unavailable=True, passed=False
  timed_out=True            → timed_out=True, passed=False
  oom_killed=True           → oom_killed=True, passed=False
  exit_code != 0 or fails   → passed=False
  exit_code == 0, no fails  → passed=True

A sandbox failure NEVER maps to passed=True.
"""

from __future__ import annotations

from agent.sandbox import SandboxResult
from agent.workflow.states import VerificationResult


def interpret_sandbox_result(run_id: str, result: SandboxResult) -> VerificationResult:
    """Convert a SandboxResult into a structured VerificationResult event."""
    if result.sandbox_error:
        return VerificationResult(
            event="verification_result",
            run_id=run_id,
            passed=False,
            tests_run=0,
            tests_passed=0,
            tests_failed=0,
            sandbox_unavailable=True,
        )
    if result.oom_killed:
        return VerificationResult(
            event="verification_result",
            run_id=run_id,
            passed=False,
            tests_run=result.tests_run,
            tests_passed=0,
            tests_failed=result.tests_failed,
            oom_killed=True,
        )
    if result.timed_out:
        return VerificationResult(
            event="verification_result",
            run_id=run_id,
            passed=False,
            tests_run=result.tests_run,
            tests_passed=0,
            tests_failed=result.tests_failed,
            timed_out=True,
        )
    passed_count = result.tests_run - result.tests_failed - result.tests_errors
    all_pass = result.exit_code == 0 and result.tests_failed == 0 and result.tests_errors == 0
    return VerificationResult(
        event="verification_result",
        run_id=run_id,
        passed=all_pass,
        tests_run=result.tests_run,
        tests_passed=max(0, passed_count),
        tests_failed=result.tests_failed + result.tests_errors,
    )
