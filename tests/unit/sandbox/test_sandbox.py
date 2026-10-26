"""Unit tests for sandbox abstractions (no Docker required).

These tests cover:
  - SandboxConfig defaults and field validation
  - _parse_surefire_summary parser
  - _validate_maven_args allowlist
  - _build_docker_cmd flag construction
  - LocalSandbox opt-in guard
  - DockerNotAvailableError when Docker is absent

Tests that actually spawn containers are in tests/sandbox/ (marker: sandbox).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.sandbox.docker_executor import (
    DockerNotAvailableError,
    DockerSandbox,
    _parse_surefire_summary,
    _validate_maven_args,
)
from agent.sandbox.local_executor import LocalSandbox
from agent.sandbox.types import SandboxConfig, SandboxMode, SandboxResult

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# SandboxConfig defaults
# ---------------------------------------------------------------------------


class TestSandboxConfig:
    def test_defaults_are_secure(self) -> None:
        cfg = SandboxConfig()
        assert cfg.network_disabled is True
        assert cfg.read_only is True
        assert cfg.cap_drop_all is True
        assert cfg.no_new_privileges is True

    def test_run_as_user_is_not_root(self) -> None:
        cfg = SandboxConfig()
        uid, gid = cfg.run_as_user.split(":")
        assert uid != "0"
        assert gid != "0"

    def test_tmpfs_mounts_include_tmp(self) -> None:
        cfg = SandboxConfig()
        assert "/tmp" in cfg.tmpfs_mounts

    def test_pids_limit_is_bounded(self) -> None:
        # 128 prevents fork bombs inside the container
        assert SandboxConfig().pids_limit <= 512

    def test_memory_limit_present(self) -> None:
        assert SandboxConfig().memory_limit  # non-empty


class TestSandboxMode:
    def test_enum_values(self) -> None:
        assert SandboxMode.docker == "docker"
        assert SandboxMode.local == "local"


# ---------------------------------------------------------------------------
# _parse_surefire_summary
# ---------------------------------------------------------------------------


class TestParseSurefireSummary:
    def test_parses_standard_line(self) -> None:
        output = "Tests run: 5, Failures: 1, Errors: 0, Skipped: 2"
        counts = _parse_surefire_summary(output)
        assert counts == {
            "tests_run": 5,
            "tests_failed": 1,
            "tests_errors": 0,
            "tests_skipped": 2,
        }

    def test_uses_last_summary_line(self) -> None:
        output = (
            "Tests run: 2, Failures: 0, Errors: 0, Skipped: 0\n"
            "Tests run: 8, Failures: 1, Errors: 0, Skipped: 1"  # aggregate
        )
        counts = _parse_surefire_summary(output)
        assert counts["tests_run"] == 8

    def test_empty_output_returns_zeros(self) -> None:
        counts = _parse_surefire_summary("")
        assert counts == {
            "tests_run": 0,
            "tests_failed": 0,
            "tests_errors": 0,
            "tests_skipped": 0,
        }

    def test_no_surefire_block_returns_zeros(self) -> None:
        counts = _parse_surefire_summary("BUILD FAILURE\n[ERROR] Something went wrong")
        assert counts["tests_run"] == 0


# ---------------------------------------------------------------------------
# _validate_maven_args
# ---------------------------------------------------------------------------


class TestValidateMavenArgs:
    def test_allowed_test_selection(self) -> None:
        _validate_maven_args(["-Dtest=MyTest", "-Dsurefire.failIfNoSpecifiedTests=false"])

    def test_allowed_module_args(self) -> None:
        _validate_maven_args(["-pl module1", "-am"])

    def test_rejects_arbitrary_plugin(self) -> None:
        with pytest.raises(ValueError, match="not in allowlist"):
            _validate_maven_args(["-Dplugin.exec=malicious"])

    def test_rejects_shell_meta(self) -> None:
        with pytest.raises(ValueError, match="not in allowlist"):
            _validate_maven_args(["--offline"])


# ---------------------------------------------------------------------------
# DockerSandbox._build_docker_cmd flag composition
# ---------------------------------------------------------------------------


class TestDockerSandboxCmdBuilder:
    def _sandbox(self, **kwargs: object) -> DockerSandbox:
        cfg = SandboxConfig(**kwargs)  # type: ignore[arg-type]
        return DockerSandbox(config=cfg)

    def _build(self, sandbox: DockerSandbox, workspace: Path | None = None) -> list[str]:
        if workspace is None:
            workspace = Path("/tmp/ws")
        return sandbox._build_docker_cmd(workspace, None, None, None, None)

    def test_rm_flag_present(self) -> None:
        cmd = self._build(self._sandbox())
        assert "--rm" in cmd

    def test_network_none_when_disabled(self) -> None:
        cmd = self._build(self._sandbox(network_disabled=True))
        assert "--network" in cmd
        idx = cmd.index("--network")
        assert cmd[idx + 1] == "none"

    def test_network_omitted_when_enabled(self) -> None:
        cmd = self._build(self._sandbox(network_disabled=False))
        assert "--network" not in cmd

    def test_read_only_flag(self) -> None:
        cmd = self._build(self._sandbox(read_only=True))
        assert "--read-only" in cmd

    def test_cap_drop_all(self) -> None:
        cmd = self._build(self._sandbox(cap_drop_all=True))
        assert "--cap-drop" in cmd
        idx = cmd.index("--cap-drop")
        assert cmd[idx + 1] == "ALL"

    def test_no_new_privileges(self) -> None:
        cmd = self._build(self._sandbox(no_new_privileges=True))
        assert "--security-opt" in cmd
        idx = cmd.index("--security-opt")
        assert cmd[idx + 1] == "no-new-privileges"

    def test_user_set_to_non_root(self) -> None:
        cmd = self._build(self._sandbox(run_as_user="1000:1000"))
        assert "--user" in cmd
        idx = cmd.index("--user")
        assert cmd[idx + 1] == "1000:1000"

    def test_memory_and_swap_equal(self) -> None:
        cmd = self._build(self._sandbox(memory_limit="512m"))
        assert "--memory" in cmd
        mem_idx = cmd.index("--memory")
        swap_idx = cmd.index("--memory-swap")
        assert cmd[mem_idx + 1] == "512m"
        assert cmd[swap_idx + 1] == "512m"

    def test_cpus_flag(self) -> None:
        cmd = self._build(self._sandbox(cpu_limit=0.5))
        assert "--cpus" in cmd
        idx = cmd.index("--cpus")
        assert cmd[idx + 1] == "0.5"

    def test_pids_limit_flag(self) -> None:
        cmd = self._build(self._sandbox(pids_limit=64))
        assert "--pids-limit" in cmd
        idx = cmd.index("--pids-limit")
        assert cmd[idx + 1] == "64"

    def test_tmpfs_mounts_present(self) -> None:
        cmd = self._build(self._sandbox(tmpfs_mounts=["/tmp"]))
        tmpfs_indices = [i for i, v in enumerate(cmd) if v == "--tmpfs"]
        mounts = {cmd[i + 1] for i in tmpfs_indices}
        # tmpfs mounts include :exec flag e.g. "/tmp:exec"
        assert any("/tmp" in m for m in mounts)

    def test_workspace_mounted_read_write(self) -> None:
        ws = Path("/some/workspace")
        cmd = self._build(self._sandbox(), workspace=ws)
        # -v /some/workspace:/workspace:rw (rw so Docker can overlay /workspace/target)
        vol_idx = cmd.index("-v")
        assert cmd[vol_idx + 1].endswith(":rw")
        assert "/workspace:rw" in cmd[vol_idx + 1]

    def test_no_docker_socket_mount(self) -> None:
        cmd = self._build(self._sandbox())
        # docker.sock must never appear in the command
        assert not any("docker.sock" in arg for arg in cmd)

    def test_test_class_appended(self) -> None:
        sandbox = self._sandbox()
        cmd = sandbox._build_docker_cmd(Path("/ws"), None, None, "MyTest", None)
        assert "-Dtest=MyTest" in cmd

    def test_image_in_cmd(self) -> None:
        cmd = self._build(self._sandbox(image="maven:test"))
        assert "maven:test" in cmd


# ---------------------------------------------------------------------------
# LocalSandbox opt-in guard
# ---------------------------------------------------------------------------


class TestLocalSandboxGuard:
    def test_requires_trusted_true(self) -> None:
        with pytest.raises(RuntimeError, match="trusted=True"):
            LocalSandbox(trusted=False)

    def test_accepts_trusted_true(self) -> None:
        sb = LocalSandbox(trusted=True)
        assert sb is not None

    def test_rejects_missing_pom(self, tmp_path: Path) -> None:
        sb = LocalSandbox(trusted=True)
        result = sb.run_maven_tests(tmp_path)
        assert result.exit_code != 0
        assert "pom.xml" in result.stderr


# ---------------------------------------------------------------------------
# DockerSandbox: Docker unavailable scenario
# ---------------------------------------------------------------------------


class TestDockerNotAvailable:
    def test_raises_when_docker_absent(self, tmp_path: Path) -> None:
        # Simulate docker command not found
        with patch("subprocess.run", side_effect=FileNotFoundError("docker not found")):
            sandbox = DockerSandbox()
            with pytest.raises(DockerNotAvailableError):
                sandbox.run_maven_tests(str(tmp_path))

    def test_raises_when_docker_returns_error(self, tmp_path: Path) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 1
        with patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "docker"),
        ):
            sandbox = DockerSandbox()
            with pytest.raises(DockerNotAvailableError):
                sandbox.run_maven_tests(str(tmp_path))


# ---------------------------------------------------------------------------
# SandboxResult properties
# ---------------------------------------------------------------------------


class TestSandboxResult:
    def test_success_when_exit_zero(self) -> None:
        r = SandboxResult(command=[], exit_code=0, stdout="", stderr="", elapsed_seconds=1.0)
        assert r.success is True

    def test_failure_on_nonzero_exit(self) -> None:
        r = SandboxResult(command=[], exit_code=1, stdout="", stderr="", elapsed_seconds=1.0)
        assert r.success is False

    def test_failure_on_timeout(self) -> None:
        r = SandboxResult(command=[], exit_code=0, stdout="", stderr="", elapsed_seconds=1.0, timed_out=True)
        assert r.success is False

    def test_failure_on_oom_killed(self) -> None:
        r = SandboxResult(command=[], exit_code=0, stdout="", stderr="", elapsed_seconds=1.0, oom_killed=True)
        assert r.success is False
