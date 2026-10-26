"""Sandbox execution for running Maven tests.

SECURITY STATUS (vertical slice):
  Currently executes Maven in a subprocess on the host with a timeout and
  no network operations. Full Docker isolation (no-network, resource limits,
  read-only filesystem) is the next milestone and is documented below.

  DO NOT claim production-grade isolation from this module.
  DO NOT execute arbitrary shell commands from LLM output directly —
  only the predefined Maven command is run.

Docker isolation target (not yet implemented):
  docker run --rm --network=none --read-only
    --tmpfs /tmp --tmpfs /root/.m2
    --memory=1g --cpus=1
    -v <repo>:/workspace:ro -v <work_volume>:/workspace/target
    maven:3.9-eclipse-temurin-17
    mvn test -f /workspace/pom.xml
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SandboxResult:
    command: list[str]
    exit_code: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    timed_out: bool = False
    tests_run: int = 0
    tests_failed: int = 0
    tests_errors: int = 0
    tests_skipped: int = 0

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


def run_maven_tests(
    repo_root: str | Path,
    timeout_seconds: int = 120,
    test_class: str | None = None,
    extra_args: list[str] | None = None,
) -> SandboxResult:
    """Run `mvn test` in the repository with a hard timeout.

    Args:
        repo_root: Path to the Maven project root (must contain pom.xml).
        timeout_seconds: Hard kill timeout. Defaults to 120 s.
        test_class: If set, run only this test class (passed as -Dtest=<class>).
        extra_args: Additional Maven arguments (validated against allowlist).

    Returns:
        SandboxResult with exit code, stdout, stderr, timing, and parsed counts.

    Security:
        Only Maven is invoked — no shell interpolation, no LLM-supplied commands.
        extra_args are validated against an allowlist.
    """
    root = Path(repo_root).resolve()
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
    if extra_args:
        _validate_maven_args(extra_args)
        cmd.extend(extra_args)

    start = time.monotonic()
    timed_out = False
    try:
        result = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        exit_code = result.returncode
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.TimeoutExpired as e:
        timed_out = True
        exit_code = -1
        stdout = (e.stdout or b"").decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = f"[TIMEOUT after {timeout_seconds}s]"

    elapsed = time.monotonic() - start
    counts = _parse_surefire_summary(stdout)

    return SandboxResult(
        command=cmd,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        elapsed_seconds=round(elapsed, 2),
        timed_out=timed_out,
        **counts,
    )


_ALLOWED_MAVEN_ARG_PREFIXES = (
    "-Dtest=",
    "-pl ",
    "-am",
    "-Dsurefire",
    "-Dfailsafe",
)


def _validate_maven_args(args: list[str]) -> None:
    for arg in args:
        if not any(arg.startswith(p) for p in _ALLOWED_MAVEN_ARG_PREFIXES):
            raise ValueError(
                f"Maven arg not in allowlist: {arg!r}. "
                "Only test-selection and module args are permitted."
            )


def _parse_surefire_summary(output: str) -> dict[str, int]:
    """Parse Maven Surefire summary line: 'Tests run: N, Failures: N, Errors: N, Skipped: N'"""
    import re

    counts = {"tests_run": 0, "tests_failed": 0, "tests_errors": 0, "tests_skipped": 0}
    pattern = re.compile(
        r"Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+),\s*Skipped:\s*(\d+)"
    )
    # Use the last summary line (the aggregate, not per-class lines)
    last_match = None
    for m in pattern.finditer(output):
        last_match = m
    if last_match:
        counts["tests_run"] = int(last_match.group(1))
        counts["tests_failed"] = int(last_match.group(2))
        counts["tests_errors"] = int(last_match.group(3))
        counts["tests_skipped"] = int(last_match.group(4))
    return counts
