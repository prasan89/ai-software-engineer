"""Checkpoint management for resumable runs.

A checkpoint captures the serializable state of an in-progress agent run
after each completed stage. If the process restarts, the orchestrator loads
the latest checkpoint and resumes from the next stage rather than starting
from scratch — avoiding redundant LLM calls.

Checkpoint lifecycle:
  create(run_id, stage, state)  →  stored
  load(run_id)                  →  latest CheckpointRecord or None
  advance(run_id, stage, state) →  appends a new record
  complete(run_id)              →  marks run finished; checkpoints may be GC'd
  abandon(run_id)               →  marks run failed

Storage is in-memory for the scaffold. The interface is designed to be
replaced with a filesystem or database backend without changing callers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from agent.workflow.states import WorkflowStage


class RunStatus(StrEnum):
    in_progress = "in_progress"
    completed = "completed"
    abandoned = "abandoned"


@dataclass
class CheckpointRecord:
    run_id: str
    stage: WorkflowStage
    state: dict[str, Any]
    saved_at_epoch: float = field(default_factory=time.time)


@dataclass
class RunRecord:
    run_id: str
    status: RunStatus
    checkpoints: list[CheckpointRecord] = field(default_factory=list)

    @property
    def latest(self) -> CheckpointRecord | None:
        return self.checkpoints[-1] if self.checkpoints else None


class CheckpointManager:
    """In-memory checkpoint store.

    All state must be JSON-serializable so that a future filesystem backend
    can persist and reload it without code changes.
    """

    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}

    def create(self, run_id: str, stage: WorkflowStage, state: dict[str, Any]) -> None:
        if run_id in self._runs:
            raise ValueError(f"Run {run_id!r} already exists; use advance() to add stages.")
        record = RunRecord(run_id=run_id, status=RunStatus.in_progress)
        record.checkpoints.append(CheckpointRecord(run_id=run_id, stage=stage, state=state))
        self._runs[run_id] = record

    def advance(self, run_id: str, stage: WorkflowStage, state: dict[str, Any]) -> None:
        run = self._require(run_id)
        if run.status != RunStatus.in_progress:
            raise ValueError(f"Run {run_id!r} is {run.status.value}; cannot advance.")
        run.checkpoints.append(CheckpointRecord(run_id=run_id, stage=stage, state=state))

    def load(self, run_id: str) -> CheckpointRecord | None:
        run = self._runs.get(run_id)
        return run.latest if run else None

    def complete(self, run_id: str) -> None:
        self._require(run_id).status = RunStatus.completed

    def abandon(self, run_id: str) -> None:
        self._require(run_id).status = RunStatus.abandoned

    def status(self, run_id: str) -> RunStatus | None:
        run = self._runs.get(run_id)
        return run.status if run else None

    def _require(self, run_id: str) -> RunRecord:
        run = self._runs.get(run_id)
        if run is None:
            raise KeyError(f"Run {run_id!r} not found.")
        return run

    def __len__(self) -> int:
        return len(self._runs)
