"""Local (host) executor for trusted fixtures ONLY.

SECURITY WARNING
----------------
This executor runs Maven directly on the host filesystem with no Docker isolation.
It MUST only be used for:

  1. Local development against fixtures that are fully trusted and version-controlled
     in this repository (e.g. tests/fixtures/spring-calc).
  2. CI pipelines where the repository has been reviewed and is known safe.

NEVER use this executor automatically for arbitrary GitHub repositories or any
user-supplied code. Doing so would execute untrusted code on the host with the
agent process's full privileges.

The DockerSandbox in docker_executor.py is the correct executor for untrusted repos.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from agent.sandbox.docker_executor import _parse_surefire_summary, _validate_maven_args
from agent.sandbox.types import SandboxResult


class LocalSandbox:
    """Runs Maven tests directly on the host.

    Opt-in only — the caller must set ``trusted=True`` explicitly to make
    it clear that this bypass of Docker isolation is intentional.

    Args:
        trusted: Must be ``True``; forces the call site to be explicit.
        timeout_seconds: Hard subprocess timeout. Defaults to 300 s.

    Raises:
        RuntimeError: if ``trusted`` is not ``True``.
    """

    def __init__(self, trusted: bool = False, timeout_seconds: int = 300) -> None:
        if not trusted:
            raise RuntimeError(
                "LocalSandbox requires trusted=True. "
                "For untrusted code use DockerSandbox instead."
            )
        self._timeout = timeout_seconds

    def run_maven_tests(
        self,
        repo_root: str | Path,
        test_class: str | None = None,
        extra_maven_args: list[str] | None = None,
    ) -> SandboxResult:
        """Run `mvn test` on the host in repo_root.

        Args:
            repo_root: Path to the Maven project root (must contain pom.xml).
            test_class: If set, pass -Dtest=<class> to limit execution.
            extra_maven_args: Additional Maven arguments (validated against allowlist).

        Returns:
            SandboxResult with exit code, stdout, stderr, timing, and parsed counts.
        """
        root = Path(repo_root).resolve()
        if not root.is_dir():
            return SandboxResult(
                command=[],
                exit_code=1,
                stdout="",
                stderr=f"Repo root does not exist: {root}",
                elapsed_seconds=0.0,
            )
        if not (root / "pom.xml").exists():
            return SandboxResult(
                command=[],
                exit_code=1,
                stdout="",
                stderr=f"No pom.xml found at {root}",
                elapsed_seconds=0.0,
            )

        cmd = ["mvn", "test", "-B", "--no-transfer-progress"]
        if test_class:
            cmd.append(f"-Dtest={test_class}")
        if extra_maven_args:
            _validate_maven_args(extra_maven_args)
            cmd.extend(extra_maven_args)

        start = time.monotonic()
        timed_out = False
        try:
            result = subprocess.run(
                cmd,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            exit_code = result.returncode
            stdout = result.stdout
            stderr = result.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = -1
            raw = exc.stdout
            stdout = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else (raw or "")
            stderr = f"[TIMEOUT after {self._timeout}s]"

        elapsed = round(time.monotonic() - start, 2)
        counts = _parse_surefire_summary(stdout)

        return SandboxResult(
            command=cmd,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            elapsed_seconds=elapsed,
            timed_out=timed_out,
            tests_run=counts["tests_run"],
            tests_failed=counts["tests_failed"],
            tests_errors=counts["tests_errors"],
            tests_skipped=counts["tests_skipped"],
        )
