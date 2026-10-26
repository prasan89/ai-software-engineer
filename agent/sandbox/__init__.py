"""Sandbox execution layer for running Maven tests.

Public API
----------
    DockerSandbox     — isolated Docker executor (required for untrusted repos)
    LocalSandbox      — host executor (opt-in, trusted fixtures only)
    SandboxResult     — result dataclass
    SandboxConfig     — Docker resource / isolation configuration
    SandboxMode       — enum (docker | local)
    run_maven_tests() — convenience wrapper; uses LocalSandbox(trusted=True)

Security status
---------------
  DockerSandbox provides:
    - Non-root execution (UID 1000)
    - --network=none (no outbound or lateral network)
    - --read-only root FS with explicit tmpfs scratch mounts
    - --cap-drop ALL (no Linux capabilities)
    - --security-opt no-new-privileges
    - CPU, memory, process-count, and wall-clock limits
    - Guaranteed container removal (--rm + best-effort kill on timeout)
    - No host Docker socket or credential mounts
    - Per-run workspace copy (host tree never touched by container)

  LocalSandbox is available for trusted fixtures with an explicit opt-in flag.
  It must never be used automatically for arbitrary repositories.

  See agent/sandbox/docker_executor.py for documented residual risks.
"""

from __future__ import annotations

from pathlib import Path

from agent.sandbox.docker_executor import DockerNotAvailableError, DockerSandbox
from agent.sandbox.local_executor import LocalSandbox
from agent.sandbox.types import SandboxConfig, SandboxMode, SandboxResult


def run_maven_tests(
    repo_root: str | Path,
    timeout_seconds: int = 120,
    test_class: str | None = None,
    extra_args: list[str] | None = None,
) -> SandboxResult:
    """Run Maven tests via LocalSandbox on trusted fixtures.

    This is a compatibility shim for the orchestrator and integration tests
    that run against version-controlled fixtures. It must not be called with
    arbitrary user-supplied repositories.

    For untrusted code use DockerSandbox directly.
    """
    sandbox = LocalSandbox(trusted=True, timeout_seconds=timeout_seconds)
    return sandbox.run_maven_tests(
        repo_root=repo_root,
        test_class=test_class,
        extra_maven_args=extra_args,
    )


__all__ = [
    "DockerNotAvailableError",
    "DockerSandbox",
    "LocalSandbox",
    "SandboxConfig",
    "SandboxMode",
    "SandboxResult",
    "run_maven_tests",
]
