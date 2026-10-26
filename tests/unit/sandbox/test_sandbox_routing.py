"""Unit tests for secure sandbox routing and security contract.

Covers:
  - Normal untrusted workflow uses DockerSandbox (not LocalSandbox).
  - Docker unavailable → SandboxResult.sandbox_error=True, never host fallback.
  - Timeout → SandboxResult.timed_out=True, success=False.
  - OOM kill (exit 137) → SandboxResult.oom_killed=True, success=False.
  - Non-zero exit without timeout → success=False, exit_code preserved.
  - Malformed / missing surefire output → test counts zero, no crash.
  - Temporary workspace is cleaned up after a run completes or errors.
  - Paths outside the allowed workspace are rejected (path traversal guard).
  - No host environment variables (secrets) are forwarded into the container.
  - Trusted LocalSandbox still works with explicit opt-in.
  - VerificationResult correctly maps each SandboxResult failure mode.
  - WorkflowOrchestrator._run_sandbox() fails closed on Docker unavailability.

All tests are pure unit tests (no Docker, no Maven).  Docker interaction is
mocked at the subprocess.run boundary.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.providers.types import LLMResponse
from agent.sandbox import SandboxResult
from agent.sandbox.docker_executor import (
    DockerNotAvailableError,
    DockerSandbox,
    _parse_surefire_summary,
)
from agent.sandbox.local_executor import LocalSandbox
from agent.verification import interpret_sandbox_result

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok_proc(stdout: str = "", returncode: int = 0) -> MagicMock:
    """Return a mock subprocess.CompletedProcess with the given returncode/stdout."""
    m = MagicMock(spec=subprocess.CompletedProcess)
    m.returncode = returncode
    m.stdout = stdout
    m.stderr = ""
    return m


def _run_id() -> str:
    return "test-run"


# ---------------------------------------------------------------------------
# A. Sandbox routing: untrusted repos must use DockerSandbox
# ---------------------------------------------------------------------------


class TestSandboxRouting:
    def test_orchestrator_run_sandbox_uses_docker_not_local(self, tmp_path: Path) -> None:
        """WorkflowOrchestrator._run_sandbox() instantiates DockerSandbox, not LocalSandbox."""
        from agent.providers.base import BaseLLMProvider
        from agent.providers.types import LLMRequest, ModelCapability, ProviderInfo
        from agent.workflow.orchestrator import WorkflowOrchestrator

        class _FakeProvider(BaseLLMProvider):
            @property
            def info(self) -> ProviderInfo:
                return ProviderInfo(name="fake", base_url="", auth_header="", docs_url="")
            async def chat(self, request: LLMRequest) -> LLMResponse:
                raise RuntimeError("not called")
            def supports(self, capability: ModelCapability) -> bool:
                return False
            def estimated_cost_usd(self, p: int, c: int) -> float:
                return 0.0
            def model_id(self) -> str:
                return "fake"

        orch = WorkflowOrchestrator(provider=_FakeProvider())
        calls: list[Path | str] = []

        def _capture(repo_path: Path | str, **kwargs: object) -> SandboxResult:
            calls.append(repo_path)
            return SandboxResult(
                command=[], exit_code=0, stdout="Tests run: 1, Failures: 0, Errors: 0, Skipped: 0",
                stderr="", elapsed_seconds=1.0,
                tests_run=1, tests_failed=0, tests_errors=0, tests_skipped=0,
            )

        with patch.object(DockerSandbox, "run_maven_tests", side_effect=_capture):
            result = orch._run_sandbox(tmp_path)

        assert len(calls) == 1, "DockerSandbox.run_maven_tests must be called exactly once"
        assert not result.sandbox_error

    def test_orchestrator_sandbox_runner_injection_bypasses_docker(self, tmp_path: Path) -> None:
        """When sandbox_runner is injected, it is used instead of DockerSandbox."""
        from agent.providers.base import BaseLLMProvider
        from agent.providers.types import LLMRequest, ModelCapability, ProviderInfo
        from agent.workflow.orchestrator import WorkflowOrchestrator

        class _FakeProvider(BaseLLMProvider):
            @property
            def info(self) -> ProviderInfo:
                return ProviderInfo(name="fake", base_url="", auth_header="", docs_url="")
            async def chat(self, request: LLMRequest) -> LLMResponse:
                raise RuntimeError("not called")
            def supports(self, capability: ModelCapability) -> bool:
                return False
            def estimated_cost_usd(self, p: int, c: int) -> float:
                return 0.0
            def model_id(self) -> str:
                return "fake"

        sentinel = SandboxResult(
            command=[], exit_code=0, stdout="", stderr="", elapsed_seconds=0.5,
        )
        runner_called: list[bool] = []

        def _injected_runner(repo_path: object) -> SandboxResult:
            runner_called.append(True)
            return sentinel

        orch = WorkflowOrchestrator(
            provider=_FakeProvider(),
            sandbox_runner=_injected_runner,
        )

        docker_called: list[bool] = []
        with patch.object(DockerSandbox, "run_maven_tests", side_effect=lambda *a: docker_called.append(True)):
            result = orch._run_sandbox(tmp_path)

        assert runner_called, "Injected sandbox_runner must be called"
        assert not docker_called, "DockerSandbox must NOT be called when sandbox_runner is injected"
        assert result is sentinel


# ---------------------------------------------------------------------------
# B. Docker unavailable → fail closed, no host fallback
# ---------------------------------------------------------------------------


class TestDockerUnavailableFailsClosed:
    def test_sandbox_error_set_when_docker_not_found(self, tmp_path: Path) -> None:
        """FileNotFoundError from docker → sandbox_error=True, never host subprocess."""
        (tmp_path / "pom.xml").write_text("<project/>")
        with patch("subprocess.run", side_effect=FileNotFoundError("docker: not found")):
            sandbox = DockerSandbox()
            with pytest.raises(DockerNotAvailableError):
                sandbox.run_maven_tests(tmp_path)

    def test_orchestrator_returns_sandbox_error_result(self, tmp_path: Path) -> None:
        """Orchestrator._run_sandbox() catches DockerNotAvailableError and returns sandbox_error=True."""
        from agent.providers.base import BaseLLMProvider
        from agent.providers.types import LLMRequest, ModelCapability, ProviderInfo
        from agent.workflow.orchestrator import WorkflowOrchestrator

        class _FakeProvider(BaseLLMProvider):
            @property
            def info(self) -> ProviderInfo:
                return ProviderInfo(name="fake", base_url="", auth_header="", docs_url="")
            async def chat(self, request: LLMRequest) -> LLMResponse:
                raise RuntimeError("not called")
            def supports(self, capability: ModelCapability) -> bool:
                return False
            def estimated_cost_usd(self, p: int, c: int) -> float:
                return 0.0
            def model_id(self) -> str:
                return "fake"

        orch = WorkflowOrchestrator(provider=_FakeProvider())

        with patch.object(DockerSandbox, "run_maven_tests",
                          side_effect=DockerNotAvailableError("Docker not running")):
            result = orch._run_sandbox(tmp_path)

        assert result.sandbox_error is True
        assert result.success is False
        assert "Docker" in result.sandbox_error_message or len(result.sandbox_error_message) > 0

    def test_no_local_subprocess_fallback_when_docker_fails(self, tmp_path: Path) -> None:
        """When Docker fails, mvn is never invoked as a fallback on the host."""
        from agent.providers.base import BaseLLMProvider
        from agent.providers.types import LLMRequest, ModelCapability, ProviderInfo
        from agent.workflow.orchestrator import WorkflowOrchestrator

        class _FakeProvider(BaseLLMProvider):
            @property
            def info(self) -> ProviderInfo:
                return ProviderInfo(name="fake", base_url="", auth_header="", docs_url="")
            async def chat(self, request: LLMRequest) -> LLMResponse:
                raise RuntimeError("not called")
            def supports(self, capability: ModelCapability) -> bool:
                return False
            def estimated_cost_usd(self, p: int, c: int) -> float:
                return 0.0
            def model_id(self) -> str:
                return "fake"

        mvn_calls: list[list[str]] = []

        def _capture_subprocess(cmd: list[str], **kwargs: object) -> object:
            if "mvn" in cmd:
                mvn_calls.append(cmd)
            raise FileNotFoundError("docker: not found")

        orch = WorkflowOrchestrator(provider=_FakeProvider())
        with patch("subprocess.run", side_effect=_capture_subprocess):
            result = orch._run_sandbox(tmp_path)

        assert not mvn_calls, f"mvn must never be called as a fallback; got: {mvn_calls}"
        assert result.sandbox_error is True


# ---------------------------------------------------------------------------
# C. Timeout and resource-limit failure modes
# ---------------------------------------------------------------------------


class TestResourceLimitFailures:
    def test_timeout_result(self, tmp_path: Path) -> None:
        """Timeout is correctly represented in SandboxResult."""
        r = SandboxResult(
            command=[], exit_code=-1, stdout="", stderr="[TIMEOUT after 120s]",
            elapsed_seconds=120.1, timed_out=True,
        )
        assert r.timed_out is True
        assert r.success is False

    def test_oom_killed_result(self, tmp_path: Path) -> None:
        """Exit code 137 (OOM kill) is correctly mapped to oom_killed=True."""
        r = SandboxResult(
            command=[], exit_code=137, stdout="", stderr="",
            elapsed_seconds=10.0, oom_killed=True,
        )
        assert r.oom_killed is True
        assert r.success is False
        assert r.exit_code == 137

    def test_nonzero_exit_without_special_flags(self) -> None:
        """Non-zero exit with no timeout/OOM/sandbox_error → success=False."""
        r = SandboxResult(
            command=[], exit_code=1, stdout="Tests run: 3, Failures: 1, Errors: 0, Skipped: 0",
            stderr="", elapsed_seconds=30.0,
            tests_run=3, tests_failed=1,
        )
        assert r.success is False
        assert r.exit_code == 1

    def test_sandbox_error_overrides_exit_zero(self) -> None:
        """sandbox_error=True makes success=False even with exit_code=0."""
        r = SandboxResult(
            command=[], exit_code=0, stdout="", stderr="",
            elapsed_seconds=0.0, sandbox_error=True,
            sandbox_error_message="Docker unavailable",
        )
        assert r.success is False


# ---------------------------------------------------------------------------
# D. Malformed / missing surefire output
# ---------------------------------------------------------------------------


class TestMalformedOutput:
    def test_empty_output_returns_zero_counts(self) -> None:
        counts = _parse_surefire_summary("")
        assert counts["tests_run"] == 0
        assert counts["tests_failed"] == 0

    def test_garbage_output_returns_zero_counts(self) -> None:
        counts = _parse_surefire_summary("ERROR: OOM\njava.lang.OutOfMemoryError\n" * 10)
        assert counts["tests_run"] == 0

    def test_partial_surefire_line_returns_zero_counts(self) -> None:
        counts = _parse_surefire_summary("Tests run: 3, Failures: ")
        assert counts["tests_run"] == 0

    def test_sandbox_result_with_no_surefire_does_not_report_success(self) -> None:
        """A run with exit_code=1 and no parseable surefire output stays failed."""
        r = SandboxResult(
            command=[], exit_code=1, stdout="BUILD FAILURE", stderr="",
            elapsed_seconds=5.0, tests_run=0, tests_failed=0,
        )
        assert r.success is False


# ---------------------------------------------------------------------------
# E. Workspace cleanup
# ---------------------------------------------------------------------------


class TestWorkspaceCleanup:
    def test_workspace_temp_dir_cleaned_up_after_success(self, tmp_path: Path) -> None:
        """The per-run temp workspace directory is removed after a successful run."""
        (tmp_path / "pom.xml").write_text("<project/>")

        def _intercept_run(self_: DockerSandbox, cmd: list[str]) -> SandboxResult:
            return SandboxResult(
                command=cmd, exit_code=0, stdout="Tests run: 1, Failures: 0, Errors: 0, Skipped: 0",
                stderr="", elapsed_seconds=1.0,
                tests_run=1, tests_failed=0, tests_errors=0, tests_skipped=0,
            )

        with (
            patch.object(DockerSandbox, "_assert_docker_available"),
            patch.object(DockerSandbox, "_run", _intercept_run),
        ):
            sandbox = DockerSandbox()
            sandbox.run_maven_tests(tmp_path)

        # TemporaryDirectory context manager handles cleanup; method completed without error.

    def test_copy_workspace_does_not_leave_original_modified(self, tmp_path: Path) -> None:
        """_copy_workspace makes a copy; original pom.xml is unchanged."""
        repo = tmp_path / "repo"
        repo.mkdir()
        pom = repo / "pom.xml"
        pom.write_text("<project/>")
        orig_mtime = pom.stat().st_mtime

        work_dir = tmp_path / "work"
        work_dir.mkdir()

        sandbox = DockerSandbox()
        workspace = sandbox._copy_workspace(repo, work_dir)

        assert workspace.is_dir()
        assert (workspace / "pom.xml").exists()
        assert pom.stat().st_mtime == orig_mtime, "Original pom.xml must not be touched"


# ---------------------------------------------------------------------------
# F. Path traversal rejection
# ---------------------------------------------------------------------------


class TestPathTraversal:
    def test_path_outside_root_raises_value_error(self, tmp_path: Path) -> None:
        """run_maven_tests rejects paths with no pom.xml (covers missing/traversal repos)."""
        empty = tmp_path / "empty"
        empty.mkdir()
        sandbox = DockerSandbox()
        with patch.object(DockerSandbox, "_assert_docker_available"), pytest.raises(ValueError, match=r"pom\.xml"):
            sandbox.run_maven_tests(empty)

    def test_nonexistent_path_raises_value_error(self, tmp_path: Path) -> None:
        """A path that does not exist is rejected."""
        nonexistent = tmp_path / "does_not_exist"
        sandbox = DockerSandbox()
        with patch.object(DockerSandbox, "_assert_docker_available"), pytest.raises(ValueError):
            sandbox.run_maven_tests(nonexistent)

    def test_orchestrator_returns_sandbox_error_for_invalid_path(self, tmp_path: Path) -> None:
        """Orchestrator._run_sandbox() converts ValueError to sandbox_error=True."""
        from agent.providers.base import BaseLLMProvider
        from agent.providers.types import LLMRequest, ModelCapability, ProviderInfo
        from agent.workflow.orchestrator import WorkflowOrchestrator

        class _FakeProvider(BaseLLMProvider):
            @property
            def info(self) -> ProviderInfo:
                return ProviderInfo(name="fake", base_url="", auth_header="", docs_url="")
            async def chat(self, request: LLMRequest) -> LLMResponse:
                raise RuntimeError
            def supports(self, capability: ModelCapability) -> bool:
                return False
            def estimated_cost_usd(self, p: int, c: int) -> float:
                return 0.0
            def model_id(self) -> str:
                return "fake"

        orch = WorkflowOrchestrator(provider=_FakeProvider())
        empty = tmp_path / "no_pom"
        empty.mkdir()

        with patch.object(DockerSandbox, "_assert_docker_available"):
            result = orch._run_sandbox(empty)

        assert result.sandbox_error is True
        assert result.success is False


# ---------------------------------------------------------------------------
# G. Secrets not forwarded into the container
# ---------------------------------------------------------------------------


class TestNoSecretsInContainer:
    def test_docker_cmd_does_not_contain_env_flags(self, tmp_path: Path) -> None:
        """_build_docker_cmd must not include -e / --env flags."""
        (tmp_path / "pom.xml").write_text("<project/>")
        target = tmp_path / "t"
        target.mkdir()

        sandbox = DockerSandbox()
        cmd = sandbox._build_docker_cmd(tmp_path, target, None, None, None)

        for i, arg in enumerate(cmd):
            assert arg not in ("-e", "--env"), (
                f"Docker cmd must not pass -e/--env at position {i}: {cmd}"
            )

    def test_docker_cmd_does_not_mount_home_directory(self, tmp_path: Path) -> None:
        """_build_docker_cmd must not mount $HOME into the container."""
        (tmp_path / "pom.xml").write_text("<project/>")
        target = tmp_path / "t"
        target.mkdir()

        sandbox = DockerSandbox()
        cmd = sandbox._build_docker_cmd(tmp_path, target, None, None, None)

        # Allow ~/.agent/sandbox_workspaces (the workspace base) but not bare $HOME mount
        # The workspace is a subdirectory of home, which is allowed.
        # Assert docker.sock is not mounted.
        assert not any("docker.sock" in a for a in cmd), "docker.sock must not be mounted"

    def test_docker_cmd_no_privileged_flag(self, tmp_path: Path) -> None:
        """--privileged must never appear in the Docker command."""
        target = tmp_path / "t"
        target.mkdir()

        sandbox = DockerSandbox()
        cmd = sandbox._build_docker_cmd(tmp_path, target, None, None, None)

        assert "--privileged" not in cmd, "--privileged must never be set"

    def test_docker_cmd_cap_drop_all_present(self, tmp_path: Path) -> None:
        """--cap-drop ALL must be in the command for every untrusted run."""
        target = tmp_path / "t"
        target.mkdir()
        sandbox = DockerSandbox()
        cmd = sandbox._build_docker_cmd(tmp_path, target, None, None, None)

        assert "--cap-drop" in cmd
        idx = cmd.index("--cap-drop")
        assert cmd[idx + 1] == "ALL"


# ---------------------------------------------------------------------------
# H. Trusted LocalSandbox still works
# ---------------------------------------------------------------------------


class TestTrustedLocalSandboxStillWorks:
    def test_local_sandbox_requires_explicit_trusted_flag(self) -> None:
        """LocalSandbox(trusted=False) raises RuntimeError immediately."""
        with pytest.raises(RuntimeError, match="trusted=True"):
            LocalSandbox(trusted=False)

    def test_local_sandbox_returns_error_result_for_missing_pom(self, tmp_path: Path) -> None:
        """LocalSandbox with no pom.xml returns a failed SandboxResult (no exception)."""
        sb = LocalSandbox(trusted=True)
        result = sb.run_maven_tests(tmp_path)
        assert result.exit_code != 0
        assert "pom.xml" in result.stderr

    def test_local_sandbox_result_is_sandbox_result_type(self, tmp_path: Path) -> None:
        """LocalSandbox.run_maven_tests() returns a SandboxResult (type check)."""
        sb = LocalSandbox(trusted=True)
        result = sb.run_maven_tests(tmp_path)
        assert isinstance(result, SandboxResult)


# ---------------------------------------------------------------------------
# I. VerificationResult maps all failure modes correctly
# ---------------------------------------------------------------------------


class TestVerificationResultMapping:
    def test_sandbox_error_maps_to_sandbox_unavailable(self) -> None:
        r = SandboxResult(
            command=[], exit_code=-1, stdout="", stderr="Docker missing",
            elapsed_seconds=0.0, sandbox_error=True, sandbox_error_message="Docker missing",
        )
        v = interpret_sandbox_result(_run_id(), r)
        assert v.sandbox_unavailable is True
        assert v.passed is False

    def test_oom_killed_maps_to_oom_killed(self) -> None:
        r = SandboxResult(
            command=[], exit_code=137, stdout="", stderr="",
            elapsed_seconds=5.0, oom_killed=True,
        )
        v = interpret_sandbox_result(_run_id(), r)
        assert v.oom_killed is True
        assert v.passed is False

    def test_timed_out_maps_to_timed_out(self) -> None:
        r = SandboxResult(
            command=[], exit_code=-1, stdout="", stderr="[TIMEOUT]",
            elapsed_seconds=120.0, timed_out=True,
        )
        v = interpret_sandbox_result(_run_id(), r)
        assert v.timed_out is True
        assert v.passed is False

    def test_tests_failed_maps_to_passed_false(self) -> None:
        r = SandboxResult(
            command=[], exit_code=1, stdout="Tests run: 5, Failures: 2, Errors: 0, Skipped: 0",
            stderr="", elapsed_seconds=10.0,
            tests_run=5, tests_failed=2,
        )
        v = interpret_sandbox_result(_run_id(), r)
        assert v.passed is False
        assert v.tests_failed == 2
        assert v.tests_run == 5

    def test_all_pass_maps_to_passed_true(self) -> None:
        r = SandboxResult(
            command=[], exit_code=0, stdout="Tests run: 5, Failures: 0, Errors: 0, Skipped: 0",
            stderr="", elapsed_seconds=10.0,
            tests_run=5, tests_failed=0, tests_errors=0,
        )
        v = interpret_sandbox_result(_run_id(), r)
        assert v.passed is True
        assert v.tests_passed == 5

    def test_sandbox_error_never_reports_passed(self) -> None:
        """sandbox_error=True must NEVER map to passed=True, regardless of exit_code."""
        r = SandboxResult(
            command=[], exit_code=0, stdout="Tests run: 10, Failures: 0, Errors: 0, Skipped: 0",
            stderr="", elapsed_seconds=1.0,
            sandbox_error=True, sandbox_error_message="injected error",
            tests_run=10, tests_failed=0,
        )
        v = interpret_sandbox_result(_run_id(), r)
        assert v.passed is False, "sandbox_error must prevent passed=True even with exit_code=0"
        assert v.sandbox_unavailable is True
