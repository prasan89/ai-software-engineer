"""Unit tests for the FastAPI routes and run manager.

Tests cover:
- Health endpoint
- Fixtures list endpoint
- Run creation validation
- Run manager: create, emit, complete, fail, eviction
- SSE route: 404 for unknown run_id, 400 for bad run_id format
- resolve_fixture_path: valid and invalid keys
- Regression test preservation in orchestrator
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# App fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    from agent.api.app import create_app
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_health_version_field(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert "version" in resp.json()


# ---------------------------------------------------------------------------
# Fixtures list
# ---------------------------------------------------------------------------


class TestFixturesList:
    def test_fixtures_returns_list(self, client: TestClient) -> None:
        resp = client.get("/fixtures")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_fixture_has_required_fields(self, client: TestClient) -> None:
        data = client.get("/fixtures").json()
        f = data[0]
        assert "key" in f
        assert "label" in f
        assert "issue_title" in f
        assert "issue_body" in f

    def test_spring_calc_fixture_present(self, client: TestClient) -> None:
        keys = [f["key"] for f in client.get("/fixtures").json()]
        assert "spring-calc" in keys


# ---------------------------------------------------------------------------
# Run creation validation
# ---------------------------------------------------------------------------


class TestCreateRunValidation:
    def test_missing_meta_key_returns_503(self, client: TestClient) -> None:
        with patch("agent.api.routes.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(meta_api_key="")
            resp = client.post("/runs", json={
                "fixture_key": "spring-calc",
                "issue_title": "Test issue",
            })
        assert resp.status_code == 503

    def test_invalid_fixture_key_returns_400(self, client: TestClient) -> None:
        with patch("agent.api.routes.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(meta_api_key="fake-key")
            resp = client.post("/runs", json={
                "fixture_key": "nonexistent-fixture",
                "issue_title": "Test",
            })
        assert resp.status_code == 400

    def test_empty_issue_title_returns_422(self, client: TestClient) -> None:
        resp = client.post("/runs", json={
            "fixture_key": "spring-calc",
            "issue_title": "",
        })
        assert resp.status_code == 422

    def test_fixture_key_with_path_traversal_returns_422(self, client: TestClient) -> None:
        resp = client.post("/runs", json={
            "fixture_key": "../etc/passwd",
            "issue_title": "test",
        })
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Run ID validation
# ---------------------------------------------------------------------------


class TestRunIdValidation:
    def test_bad_run_id_format_returns_400_events(self, client: TestClient) -> None:
        resp = client.get("/runs/../../etc/events")
        assert resp.status_code in (400, 404)

    def test_bad_run_id_format_returns_400_status(self, client: TestClient) -> None:
        resp = client.get("/runs/tooooooolong")
        assert resp.status_code in (400, 404)

    def test_unknown_run_id_returns_404_events(self, client: TestClient) -> None:
        resp = client.get("/runs/deadbeef/events")
        assert resp.status_code == 404

    def test_unknown_run_id_returns_404_status(self, client: TestClient) -> None:
        resp = client.get("/runs/deadbeef")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Run manager
# ---------------------------------------------------------------------------


class TestRunManager:
    def test_create_returns_record(self) -> None:
        from agent.api.run_manager import RunManager
        mgr = RunManager()
        rec = mgr.create("aabbccdd")
        assert rec.run_id == "aabbccdd"
        assert rec.report is None

    def test_get_unknown_returns_none(self) -> None:
        from agent.api.run_manager import RunManager
        mgr = RunManager()
        assert mgr.get("unknown") is None

    def test_emit_puts_event_in_queue(self) -> None:
        from agent.api.run_manager import RunManager
        from agent.workflow.states import RunCompleted
        mgr = RunManager()
        mgr.create("aabb1122")
        ev = RunCompleted(run_id="aabb1122", reused_fix=False, total_cost_usd=0.0)
        mgr.emit("aabb1122", ev)
        rec = mgr.get("aabb1122")
        assert rec is not None
        item = rec.queue.get_nowait()
        assert item is ev

    def test_complete_puts_sentinel(self) -> None:
        from agent.api.run_manager import RunManager
        mgr = RunManager()
        mgr.create("ccddee11")
        fake_report = MagicMock()
        mgr.complete("ccddee11", fake_report)
        rec = mgr.get("ccddee11")
        assert rec is not None
        sentinel = rec.queue.get_nowait()
        assert sentinel is None
        assert rec.report is fake_report

    def test_fail_sets_error(self) -> None:
        from agent.api.run_manager import RunManager
        mgr = RunManager()
        mgr.create("ffee1122")
        mgr.fail("ffee1122", "something broke")
        rec = mgr.get("ffee1122")
        assert rec is not None
        assert rec.error == "something broke"

    def test_max_active_runs_raises(self) -> None:
        from agent.api.run_manager import MAX_ACTIVE_RUNS, RunManager
        mgr = RunManager()
        for i in range(MAX_ACTIVE_RUNS):
            run_id = f"{i:08x}"
            mgr.create(run_id)
            mgr.fail(run_id, "stub")  # mark as done but don't evict (no report)
        with pytest.raises(RuntimeError, match="Too many"):
            # Need one more that has no report (can't evict)
            mgr.create("ffffffff")

    def test_emit_unknown_run_noop(self) -> None:
        from agent.api.run_manager import RunManager
        from agent.workflow.states import RunCompleted
        mgr = RunManager()
        ev = RunCompleted(run_id="nope", reused_fix=False, total_cost_usd=0.0)
        mgr.emit("nope", ev)  # should not raise


# ---------------------------------------------------------------------------
# resolve_fixture_path
# ---------------------------------------------------------------------------


class TestResolveFixturePath:
    def test_known_key_returns_path(self) -> None:
        from agent.api.run_manager import resolve_fixture_path
        p = resolve_fixture_path("spring-calc")
        assert isinstance(p, Path)

    def test_unknown_key_raises_value_error(self) -> None:
        from agent.api.run_manager import resolve_fixture_path
        with pytest.raises(ValueError, match="Unknown fixture key"):
            resolve_fixture_path("not-a-fixture")


# ---------------------------------------------------------------------------
# Orchestrator: event_callback wiring
# ---------------------------------------------------------------------------


class TestOrchestratorEventCallback:
    @pytest.mark.asyncio
    async def test_event_callback_receives_events(self, tmp_path: Path) -> None:
        """event_callback is called for every emitted event."""
        from agent.providers.base import BaseLLMProvider
        from agent.providers.types import LLMRequest, LLMResponse, ModelCapability, ProviderInfo
        from agent.workflow.orchestrator import RunInput, WorkflowOrchestrator

        class _FakeProvider(BaseLLMProvider):
            @property
            def info(self) -> ProviderInfo:
                return ProviderInfo(name="fake", base_url="", auth_header="", docs_url="")
            async def chat(self, request: LLMRequest) -> LLMResponse:
                raise RuntimeError("not called")
            def supports(self, c: ModelCapability) -> bool:
                return False
            def estimated_cost_usd(self, p: int, c: int) -> float:
                return 0.0
            def model_id(self) -> str:
                return "fake"

        received: list[object] = []

        def _cb(event: object) -> None:
            received.append(event)

        orch = WorkflowOrchestrator(
            provider=_FakeProvider(),
            event_callback=_cb,
        )

        # Run with a nonexistent path so it fails fast, but still emits intake events
        inp = RunInput(
            repo_path=tmp_path / "nonexistent",
            issue_title="t",
            issue_body="b",
            run_id="test-cb",
        )
        with contextlib.suppress(Exception):
            await orch.run(inp)
        # Should have received at least the intake stage_started event
        assert len(received) >= 1


# ---------------------------------------------------------------------------
# Orchestrator: regression test preservation
# ---------------------------------------------------------------------------


class TestRegressionTestPreservation:
    def test_test_files_captured_after_injection(self, tmp_path: Path) -> None:
        """After _inject_test_method, the injected test method is present in the file."""
        from agent.providers.base import BaseLLMProvider
        from agent.providers.types import LLMRequest, LLMResponse, ModelCapability, ProviderInfo
        from agent.repo import FileEntry, RepoContext
        from agent.workflow.orchestrator import WorkflowOrchestrator
        from agent.workflow.schemas import TestGenerationResponse

        class _FakeProvider(BaseLLMProvider):
            @property
            def info(self) -> ProviderInfo:
                return ProviderInfo(name="fake", base_url="", auth_header="", docs_url="")
            async def chat(self, request: LLMRequest) -> LLMResponse:
                raise RuntimeError("not called")
            def supports(self, c: ModelCapability) -> bool:
                return False
            def estimated_cost_usd(self, p: int, c: int) -> float:
                return 0.0
            def model_id(self) -> str:
                return "fake"

        # Create a minimal repo structure
        src = tmp_path / "src/test/java/com/example"
        src.mkdir(parents=True)
        test_file = src / "MyTest.java"
        test_file.write_text(
            "class MyTest {\n    @Test void existing() {}\n}\n", encoding="utf-8"
        )

        entry = FileEntry(
            rel_path="src/test/java/com/example/MyTest.java",
            abs_path=str(test_file),
            language="java",
            size_bytes=test_file.stat().st_size,
            git_blob_sha="",
        )
        repo = RepoContext(
            root=str(tmp_path),
            tree_hash="abc123",
            files=[entry],
        )

        orch = WorkflowOrchestrator(provider=_FakeProvider())
        tr = TestGenerationResponse(
            test_method_name="divide_null",
            test_method_body="assertThrows(Exception.class, () -> {});",
            explanation="test",
        )
        orch._inject_test_method(repo, tr)

        updated = test_file.read_text(encoding="utf-8")
        assert "divide_null" in updated
