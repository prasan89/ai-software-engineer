"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI

from agent.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Software Engineer Agent",
        description="Accepts GitHub issues, analyzes Java/Spring Boot repos, implements and verifies fixes.",
        version="0.1.0",
    )
    app.include_router(router)
    return app


app = create_app()
