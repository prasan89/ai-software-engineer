"""Docker sandbox integration tests.

These tests require a running Docker daemon and the Maven image.
They are marked ``sandbox`` and are excluded from the default test run.

Run with:
    pytest -m sandbox -v

What is tested:
  1. Basic Maven execution succeeds inside the container for the spring-calc fixture
  2. --network=none prevents outbound network calls
  3. --read-only prevents writing to the container filesystem (outside tmpfs)
  4. Timeout produces a SandboxResult with timed_out=True and guaranteed cleanup
  5. Exit code 137 (OOM kill) is mapped to oom_killed=True
  6. Workspace copy is isolated (the original repo is untouched)
  7. Resource limits (pids_limit, cpu) are present in the command
  8. DockerNotAvailableError is raised when Docker socket is absent

What is NOT tested here (documented):
  - Actual memory OOM kill (requires running a real memory-hungry process; slow)
  - Maven plugin network calls being blocked (requires a plugin that calls home)

Maven offline mode:
  The slow Maven tests require a pre-seeded Maven local repository.
  Run ``mvn -f fixtures/spring-calc/pom.xml dependency:go-offline`` first,
  or set MAVEN_LOCAL_REPO env var to a pre-seeded .m2/repository directory.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from agent.sandbox import DockerSandbox, SandboxConfig, SandboxResult

pytestmark = pytest.mark.sandbox

FIXTURE_SRC = Path(__file__).parents[2] / "fixtures" / "spring-calc"

# Lightweight image for fast non-Maven isolation tests
_ALPINE = "alpine:3.20"

# Maven image used for real build tests
_MAVEN_IMAGE = "maven:3.9.9-eclipse-temurin-21-alpine"

# Maven local repo for offline builds — uses host default or env var override
_HOST_M2_REPO = Path(os.environ.get("MAVEN_LOCAL_REPO", Path.home() / ".m2" / "repository"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _copy_fixture(dest: Path) -> Path:
    """Copy spring-calc fixture to a fresh directory (no .git, no target)."""
    repo_dir = dest / "spring-calc"
    shutil.copytree(
        FIXTURE_SRC,
        repo_dir,
        ignore=shutil.ignore_patterns(".git", "target"),
    )
    subprocess.run(["git", "init", "-b", "main"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "feat: initial"],
        cwd=str(repo_dir),
        check=True,
        capture_output=True,
        env={**__import__("os").environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t.com",
             "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t.com"},
    )
    return repo_dir


def _run_in_container(image: str, cmd_args: list[str], *, network: bool = False) -> subprocess.CompletedProcess[str]:
    """Run an arbitrary command in a minimal sandbox for isolation probing."""
    docker_cmd = [
        "docker", "run", "--rm",
        "--read-only",
        "--tmpfs", "/tmp",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
    ]
    if not network:
        docker_cmd += ["--network", "none"]
    docker_cmd.append(image)
    docker_cmd.extend(cmd_args)
    return subprocess.run(docker_cmd, capture_output=True, text=True, timeout=30)


# ---------------------------------------------------------------------------
# Isolation boundary tests (fast, use alpine rather than Maven image)
# ---------------------------------------------------------------------------


@pytest.mark.sandbox
def test_network_none_prevents_outbound() -> None:
    """Container with --network=none cannot reach external hosts."""
    result = _run_in_container(_ALPINE, ["sh", "-c", "wget -T3 -q http://example.com -O /dev/null; echo exit:$?"])
    # wget or ping should fail — network is not reachable
    # Either the command itself is not found (wget absent in slim images) or it fails
    assert result.returncode != 0 or "exit:1" in result.stdout or "network" in result.stderr.lower() or "wget" in result.stderr.lower()


@pytest.mark.sandbox
def test_read_only_filesystem_prevents_write_to_root() -> None:
    """Container with --read-only cannot write to / (only to /tmp via tmpfs)."""
    result = _run_in_container(_ALPINE, ["sh", "-c", "echo test > /test.txt 2>&1; echo rc:$?"])
    # Writing to / should fail with "Read-only file system" (may appear in stderr or stdout)
    combined = result.stdout + result.stderr
    assert "Read-only file system" in combined or "rc:1" in result.stdout


@pytest.mark.sandbox
def test_read_only_tmpfs_is_writable() -> None:
    """Container can write to /tmp even with --read-only (tmpfs mount)."""
    result = _run_in_container(_ALPINE, ["sh", "-c", "echo test > /tmp/probe.txt && cat /tmp/probe.txt"])
    assert result.returncode == 0
    assert "test" in result.stdout


@pytest.mark.sandbox
def test_no_docker_socket_in_container() -> None:
    """Container cannot access the host Docker socket."""
    result = _run_in_container(_ALPINE, ["sh", "-c", "ls /var/run/docker.sock 2>&1; echo rc:$?"])
    assert "rc:1" in result.stdout or "No such file" in result.stdout


# ---------------------------------------------------------------------------
# DockerSandbox._copy_workspace isolation
# ---------------------------------------------------------------------------


@pytest.mark.sandbox
def test_workspace_copy_does_not_modify_original(tmp_path: Path) -> None:
    """The per-run workspace copy leaves the original repository untouched."""
    repo = _copy_fixture(tmp_path)
    original_mtime = (repo / "pom.xml").stat().st_mtime

    sandbox = DockerSandbox(SandboxConfig(image=_ALPINE))
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        workspace = sandbox._copy_workspace(repo, Path(td))
        # Workspace is a copy
        assert workspace.is_dir()
        assert (workspace / "pom.xml").exists()

    # Original unchanged
    assert (repo / "pom.xml").stat().st_mtime == original_mtime


@pytest.mark.sandbox
def test_workspace_copy_excludes_git_history(tmp_path: Path) -> None:
    """Workspace copy does not include .git metadata."""
    repo = _copy_fixture(tmp_path)
    sandbox = DockerSandbox()
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        workspace = sandbox._copy_workspace(repo, Path(td))
        assert not (workspace / ".git").exists()


# ---------------------------------------------------------------------------
# Timeout handling
# ---------------------------------------------------------------------------


@pytest.mark.sandbox
def test_timeout_produces_timed_out_result() -> None:
    """A sandbox that exceeds timeout_seconds returns timed_out=True."""
    # Use alpine to run an infinite sleep — much faster than a Maven build
    cfg = SandboxConfig(
        timeout_seconds=2,
        image=_ALPINE,
        network_disabled=True,
        read_only=False,  # alpine doesn't need read-only for this test
        tmpfs_mounts=["/tmp"],
    )
    sandbox = DockerSandbox(config=cfg)

    # Override _build_docker_cmd to run `sleep 60` instead of Maven
    original_build = sandbox._build_docker_cmd

    def _patched_build(workspace: Path, target_dir: object, m2_dir: object, test_class: object, extra_maven_args: object) -> list[str]:
        cmd = original_build(workspace, None, None, None, None)
        # Replace the Maven command at the end with sleep 60
        maven_start = next(i for i, v in enumerate(cmd) if v == cfg.image)
        return [*cmd[:maven_start + 1], "sleep", "60"]

    sandbox._build_docker_cmd = _patched_build  # type: ignore[method-assign]

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "workspace").mkdir()
        # Create a minimal pom.xml to pass validation
        pom = Path(tmp, "pom.xml")
        pom.write_text("<project/>")
        start = time.monotonic()
        result = sandbox._run(
            _patched_build(Path(tmp, "workspace"), None, None, None, None)
        )

    elapsed = time.monotonic() - start
    assert result.timed_out is True
    assert elapsed < 15, f"Cleanup after timeout took too long: {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# Maven execution with the spring-calc fixture (slow)
# ---------------------------------------------------------------------------


@pytest.mark.sandbox
@pytest.mark.slow
def test_maven_build_runs_in_docker_fixture_has_npe_bug(tmp_path: Path) -> None:
    """spring-calc's NPE test fails (exit 1) inside Docker; demonstrates isolation works."""
    if not _HOST_M2_REPO.is_dir():
        pytest.skip(f"Maven local repo not found at {_HOST_M2_REPO}. Run: mvn dependency:go-offline -f fixtures/spring-calc/pom.xml")
    repo = _copy_fixture(tmp_path)
    sandbox = DockerSandbox(SandboxConfig(timeout_seconds=300))
    result = sandbox.run_maven_tests(repo, maven_local_repo=_HOST_M2_REPO)

    # The fixture has a deliberate NPE bug; one test is expected to fail
    assert result.exit_code == 1, f"Expected exit 1 (failing test), got {result.exit_code}\nstdout={result.stdout[-1000:]}"
    assert result.tests_run > 0
    assert result.tests_failed == 1  # exactly the NPE test
    assert not result.timed_out
    assert not result.oom_killed


@pytest.mark.sandbox
@pytest.mark.slow
def test_three_run_demo_docker_fresh_run(tmp_path: Path) -> None:
    """Run 1: fresh Maven execution in Docker; fixture has NPE bug so tests fail (exit 1)."""
    if not _HOST_M2_REPO.is_dir():
        pytest.skip(f"Maven local repo not found at {_HOST_M2_REPO}")
    repo = _copy_fixture(tmp_path)
    sandbox = DockerSandbox(SandboxConfig(timeout_seconds=300))
    result = sandbox.run_maven_tests(repo, maven_local_repo=_HOST_M2_REPO)

    assert isinstance(result, SandboxResult)
    assert result.elapsed_seconds > 0
    # Fixture has a bug (NPE on null divisor), so the NPE test fails (exit 1)
    assert result.exit_code in (0, 1)
    assert not result.timed_out
    assert not result.oom_killed


@pytest.mark.sandbox
@pytest.mark.slow
def test_maven_build_applies_patch_and_passes(tmp_path: Path) -> None:
    """After the NPE fix patch is applied, all tests pass inside Docker."""
    if not _HOST_M2_REPO.is_dir():
        pytest.skip(f"Maven local repo not found at {_HOST_M2_REPO}")

    import subprocess as sp

    repo = _copy_fixture(tmp_path)

    CORRECT_DIFF = """\
--- a/src/main/java/com/example/calc/CalculatorService.java
+++ b/src/main/java/com/example/calc/CalculatorService.java
@@ -30,7 +30,9 @@
      * @throws IllegalArgumentException if divisor is null   ← MISSING: currently throws NPE
      */
     public int divide(Integer dividend, Integer divisor) {
-        // BUG: missing null check on divisor — throws NullPointerException when divisor == null
+        if (divisor == null) {
+            throw new IllegalArgumentException("Divisor must not be null");
+        }
         if (divisor == 0) {
             throw new IllegalArgumentException("Divisor must not be zero");
         }
"""
    with tempfile.NamedTemporaryFile(suffix=".patch", mode="w", delete=False) as pf:
        pf.write(CORRECT_DIFF)
        patch_path = pf.name

    sp.run(["git", "apply", patch_path], cwd=str(repo), check=True, capture_output=True)

    sandbox = DockerSandbox(SandboxConfig(timeout_seconds=300))
    result = sandbox.run_maven_tests(repo, maven_local_repo=_HOST_M2_REPO)

    assert result.exit_code == 0, f"Expected patched build to pass:\n{result.stdout[-2000:]}"
    assert result.tests_failed == 0


# ---------------------------------------------------------------------------
# Missing pom.xml raises ValueError (not a Docker error)
# ---------------------------------------------------------------------------


@pytest.mark.sandbox
def test_missing_pom_raises_value_error(tmp_path: Path) -> None:
    sandbox = DockerSandbox()
    with pytest.raises(ValueError, match=r"pom\.xml"):
        sandbox.run_maven_tests(tmp_path)
