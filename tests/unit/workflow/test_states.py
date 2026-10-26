"""Unit tests for workflow states and events."""

from __future__ import annotations

import pytest

from agent.workflow.states import (
    TERMINAL_STAGES,
    WorkflowStage,
    WorkflowState,
)

pytestmark = pytest.mark.unit


class TestWorkflowState:
    def test_initial_stage_is_intake(self) -> None:
        state = WorkflowState(run_id="run-1")
        assert state.stage == WorkflowStage.intake

    def test_done_is_terminal(self) -> None:
        state = WorkflowState(run_id="run-1", stage=WorkflowStage.done)
        assert state.is_terminal()

    def test_failed_is_terminal(self) -> None:
        state = WorkflowState(run_id="run-1", stage=WorkflowStage.failed)
        assert state.is_terminal()

    def test_in_progress_stages_not_terminal(self) -> None:
        non_terminal = [s for s in WorkflowStage if s not in TERMINAL_STAGES]
        for stage in non_terminal:
            state = WorkflowState(run_id="x", stage=stage)
            assert not state.is_terminal(), f"{stage} should not be terminal"

    def test_reused_fix_defaults_false(self) -> None:
        state = WorkflowState(run_id="x")
        assert state.reused_fix is False
