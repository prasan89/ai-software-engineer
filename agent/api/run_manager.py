"""In-process run manager: stores active runs, provides SSE event queues.

Security:
- run_id is a UUID4 generated server-side; never accepted from user input.
- Each run gets an isolated asyncio.Queue; one run cannot peek into another.
- Completed run reports are evicted after MAX_AGE to bound memory.
- No secrets (API keys, tokens) are placed in RunRecord or events.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from agent.workflow.orchestrator import RunReport
from agent.workflow.states import RunEvent

logger = logging.getLogger(__name__)

MAX_AGE_SECONDS = 3600  # keep completed runs for 1 hour
MAX_ACTIVE_RUNS = 20    # hard cap to bound resource use


@dataclass
class RunRecord:
    run_id: str
    queue: asyncio.Queue[RunEvent | None]  # None is the sentinel (run complete)
    report: RunReport | None = None
    error: str = ""
    created_at: float = field(default_factory=time.monotonic)


class RunManager:
    """Process-global registry of active and recently completed runs."""

    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}

    def create(self, run_id: str) -> RunRecord:
        self._evict_stale()
        if len(self._runs) >= MAX_ACTIVE_RUNS:
            raise RuntimeError(
                f"Too many concurrent runs ({MAX_ACTIVE_RUNS} max). Try again shortly."
            )
        record = RunRecord(run_id=run_id, queue=asyncio.Queue(maxsize=500))
        self._runs[run_id] = record
        return record

    def get(self, run_id: str) -> RunRecord | None:
        return self._runs.get(run_id)

    def emit(self, run_id: str, event: RunEvent) -> None:
        record = self._runs.get(run_id)
        if record is None:
            return
        try:
            record.queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("Event queue full for run %s; dropping event", run_id)

    def complete(self, run_id: str, report: RunReport) -> None:
        record = self._runs.get(run_id)
        if record is None:
            return
        record.report = report
        with contextlib.suppress(asyncio.QueueFull):
            record.queue.put_nowait(None)  # sentinel

    def fail(self, run_id: str, error: str) -> None:
        record = self._runs.get(run_id)
        if record is None:
            return
        record.error = error
        with contextlib.suppress(asyncio.QueueFull):
            record.queue.put_nowait(None)

    def _evict_stale(self) -> None:
        now = time.monotonic()
        stale = [
            rid for rid, rec in self._runs.items()
            if rec.report is not None and (now - rec.created_at) > MAX_AGE_SECONDS
        ]
        for rid in stale:
            del self._runs[rid]


# Process-global singleton — shared across request handlers
_manager = RunManager()


def get_run_manager() -> RunManager:
    return _manager


def resolve_fixture_path(fixture_key: str) -> Path:
    """Map a fixture key to its absolute path on disk.

    Only pre-registered keys are accepted to prevent path traversal.
    """
    _FIXTURES: dict[str, Path] = {
        "spring-calc": Path(__file__).parents[2] / "fixtures" / "spring-calc",
    }
    path = _FIXTURES.get(fixture_key)
    if path is None:
        raise ValueError(f"Unknown fixture key: {fixture_key!r}. Valid: {list(_FIXTURES)}")
    if not path.is_dir():
        raise ValueError(f"Fixture directory not found: {path}")
    return path
