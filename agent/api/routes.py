"""API route definitions — POST /runs, SSE /runs/{id}/events, GET /runs/{id}."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from agent.api.run_manager import get_run_manager, resolve_fixture_path
from agent.api.settings import get_settings
from agent.engineering_knowledge.cache import CacheInvalidator
from agent.engineering_knowledge.fix_store import FixStore
from agent.providers.meta import MUSE_SPARK_1_3, MetaProvider
from agent.sandbox.types import SandboxConfig
from agent.workflow.orchestrator import RunInput, WorkflowOrchestrator
from agent.workflow.states import RunEvent

logger = logging.getLogger(__name__)

router = APIRouter()

# Shared instances (process-lifetime; no secrets stored here)
_fix_store = FixStore()
_cache = CacheInvalidator()

# -----------------------------------------------------------------------
# Static assets path
# -----------------------------------------------------------------------
_STATIC_DIR = Path(__file__).parent / "static"


# -----------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    version: str


@router.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version="0.1.0")


# -----------------------------------------------------------------------
# Fixtures list
# -----------------------------------------------------------------------


class FixtureInfo(BaseModel):
    key: str
    label: str
    issue_title: str
    issue_body: str


@router.get("/fixtures", response_model=list[FixtureInfo], tags=["fixtures"])
async def list_fixtures() -> list[FixtureInfo]:
    fixture_dir = Path(__file__).parents[2] / "fixtures" / "spring-calc"
    issue_file = fixture_dir / "ISSUE.md"
    issue_text = issue_file.read_text() if issue_file.exists() else ""
    lines = issue_text.strip().splitlines()
    # Strip markdown heading markers and trim
    raw_title = lines[0].lstrip("#").strip() if lines else "divide() throws NullPointerException"
    # Remove "Issue #N — " prefix if present
    if " — " in raw_title:
        raw_title = raw_title.split(" — ", 1)[1].strip()
    body = "\n".join(lines[2:]).strip() if len(lines) > 2 else issue_text
    return [
        FixtureInfo(
            key="spring-calc",
            label="Spring Boot Calculator — NPE in divide()",
            issue_title=raw_title,
            issue_body=body,
        )
    ]


# -----------------------------------------------------------------------
# POST /runs — create a new run
# -----------------------------------------------------------------------


class CreateRunRequest(BaseModel):
    fixture_key: str = Field(..., min_length=1, max_length=64)
    issue_title: str = Field(..., min_length=1, max_length=500)
    issue_body: str = Field(default="", max_length=8000)

    @field_validator("fixture_key")
    @classmethod
    def _safe_key(cls, v: str) -> str:
        if not re.match(r"^[a-z0-9_-]+$", v):
            raise ValueError("fixture_key may only contain [a-z0-9_-]")
        return v


class CreateRunResponse(BaseModel):
    run_id: str
    status: str = "intake"


@router.post("/runs", response_model=CreateRunResponse, tags=["runs"])
async def create_run(body: CreateRunRequest) -> CreateRunResponse:
    settings = get_settings()
    if not settings.meta_api_key:
        raise HTTPException(status_code=503, detail="META_API_KEY not configured on server")

    try:
        fixture_src = resolve_fixture_path(body.fixture_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    run_id = str(uuid.uuid4())[:8]
    manager = get_run_manager()
    try:
        manager.create(run_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc

    # Copy fixture to an isolated temp directory — each run gets its own workspace
    tmp_root = Path(tempfile.mkdtemp(prefix=f"run_{run_id}_"))
    repo_path = tmp_root / "repo"
    shutil.copytree(fixture_src, repo_path, ignore=shutil.ignore_patterns(".git", "target"))
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "agent",
        "GIT_AUTHOR_EMAIL": "agent@local",
        "GIT_COMMITTER_NAME": "agent",
        "GIT_COMMITTER_EMAIL": "agent@local",
    }
    subprocess.run(["git", "init", "-b", "main"], cwd=str(repo_path), check=True,
                   capture_output=True, env=env)
    subprocess.run(["git", "add", "-A"], cwd=str(repo_path), check=True,
                   capture_output=True, env=env)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo_path), check=True,
                   capture_output=True, env=env)

    m2_repo = Path(os.environ.get("MAVEN_LOCAL_REPO", Path.home() / ".m2" / "repository"))

    async def _run() -> None:
        try:
            provider = MetaProvider(model=MUSE_SPARK_1_3)
            config = SandboxConfig(
                timeout_seconds=300,
                maven_local_repo=m2_repo if m2_repo.is_dir() else None,
            )

            def _emit(event: RunEvent) -> None:
                manager.emit(run_id, event)

            orch = WorkflowOrchestrator(
                provider=provider,
                fix_store=_fix_store,
                cache=_cache,
                sandbox_config=config,
                event_callback=_emit,
            )

            inp = RunInput(
                repo_path=repo_path,
                issue_title=body.issue_title,
                issue_body=body.issue_body,
                repo_full_name=f"demo/{body.fixture_key}",
                run_id=run_id,
            )
            report = await orch.run(inp)
            manager.complete(run_id, report)
        except Exception as exc:
            logger.exception("Run %s failed: %s", run_id, exc)
            manager.fail(run_id, str(exc))
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    _task = asyncio.create_task(_run())  # noqa: RUF006 — fire-and-forget; run_manager tracks lifecycle
    return CreateRunResponse(run_id=run_id)


# -----------------------------------------------------------------------
# GET /runs/{run_id}/events — SSE stream
# -----------------------------------------------------------------------


@router.get("/runs/{run_id}/events", tags=["runs"])
async def stream_events(run_id: str, request: Request) -> StreamingResponse:
    if not re.match(r"^[a-f0-9]{8}$", run_id):
        raise HTTPException(status_code=400, detail="Invalid run_id format")
    manager = get_run_manager()
    record = manager.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")

    async def _generate() -> AsyncGenerator[str, None]:
        # If run already finished before SSE connected, drain the queue
        while True:
            if await request.is_disconnected():
                break
            try:
                event: RunEvent | None = await asyncio.wait_for(
                    record.queue.get(), timeout=30.0
                )
            except TimeoutError:
                # Send keepalive comment
                yield ": keepalive\n\n"
                continue

            if event is None:
                # Sentinel — run is done; send the final report
                if record.report is not None:
                    payload = {
                        "event": "run_report",
                        "run_id": run_id,
                        "success": record.report.success,
                        "reused_fix": record.report.reused_fix,
                        "root_cause": record.report.root_cause,
                        "fix_description": record.report.fix_description,
                        "unified_diff": record.report.unified_diff,
                        "elapsed_seconds": record.report.elapsed_seconds,
                        "usage_summary": record.report.usage_summary,
                        "error": record.report.error,
                        "sandbox_stdout": record.report.sandbox_stdout[-2000:],
                        "verification": (
                            {
                                "passed": record.report.verification.passed,
                                "tests_run": record.report.verification.tests_run,
                                "tests_passed": record.report.verification.tests_passed,
                                "tests_failed": record.report.verification.tests_failed,
                                "sandbox_unavailable": record.report.verification.sandbox_unavailable,
                                "timed_out": record.report.verification.timed_out,
                                "oom_killed": record.report.verification.oom_killed,
                            }
                            if record.report.verification else None
                        ),
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                elif record.error:
                    yield f"data: {json.dumps({'event': 'run_error', 'run_id': run_id, 'error': record.error})}\n\n"
                break

            # Regular event
            try:
                payload = event.model_dump()
                yield f"data: {json.dumps(payload)}\n\n"
            except Exception:
                pass

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# -----------------------------------------------------------------------
# GET /runs/{run_id} — current status / result
# -----------------------------------------------------------------------


class RunStatusResponse(BaseModel):
    run_id: str
    done: bool
    success: bool | None = None
    error: str = ""


@router.get("/runs/{run_id}", response_model=RunStatusResponse, tags=["runs"])
async def get_run(run_id: str) -> RunStatusResponse:
    if not re.match(r"^[a-f0-9]{8}$", run_id):
        raise HTTPException(status_code=400, detail="Invalid run_id format")
    manager = get_run_manager()
    record = manager.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    done = record.report is not None or bool(record.error)
    return RunStatusResponse(
        run_id=run_id,
        done=done,
        success=record.report.success if record.report else None,
        error=record.error,
    )


# -----------------------------------------------------------------------
# GET / — serve the single-page UI
# -----------------------------------------------------------------------


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> HTMLResponse:
    html_path = _STATIC_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>UI not built</h1>", status_code=503)
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
