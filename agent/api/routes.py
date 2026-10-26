"""API route definitions — scaffold stubs."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    version: str


@router.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version="0.1.0")


# ---------------------------------------------------------------------------
# Placeholder: POST /runs  (to be implemented)
# ---------------------------------------------------------------------------
# Accepts: { "repo_full_name": "owner/repo", "issue_number": 42 }
# Returns: { "run_id": "...", "status": "intake" }
# Streams:  GET /runs/{run_id}/events  (SSE, RunEvent stream)
