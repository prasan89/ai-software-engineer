"""Shared types for the sandbox execution layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class SandboxMode(StrEnum):
    """Execution mode selector.

    DOCKER: fully isolated Docker container (required for untrusted repos).
    LOCAL:  host subprocess — opt-in only; never used for arbitrary public repos.
    """

    docker = "docker"
    local = "local"


@dataclass
class SandboxResult:
    """Outcome of one sandbox execution."""

    command: list[str]
    exit_code: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    timed_out: bool = False
    oom_killed: bool = False
    # Set when the sandbox itself could not run (Docker unavailable, path error,
    # etc.). When True, the exit_code and test counts are meaningless.
    sandbox_error: bool = False
    sandbox_error_message: str = ""
    tests_run: int = 0
    tests_failed: int = 0
    tests_errors: int = 0
    tests_skipped: int = 0

    @property
    def success(self) -> bool:
        return (
            self.exit_code == 0
            and not self.timed_out
            and not self.oom_killed
            and not self.sandbox_error
        )


@dataclass
class SandboxConfig:
    """Resource and isolation parameters for a single sandbox run.

    Defaults are conservative and suitable for untrusted Maven builds.
    """

    # Hard wall-clock timeout (seconds). Container is killed after this.
    timeout_seconds: int = 300

    # Memory limit passed to Docker --memory. String because Docker accepts "1g", "512m".
    memory_limit: str = "1g"

    # CPU quota (fractional cores). 1.0 == one full core.
    cpu_limit: float = 1.0

    # Maximum number of processes/threads inside the container.
    pids_limit: int = 128

    # Network: True means --network=none. Set False only for tests that need
    # Maven Central; production always uses True.
    network_disabled: bool = True

    # Read-only root FS. tmpfs mounts supply writable temp space.
    read_only: bool = True

    # Image to use. Pinned by default; callers may override for testing.
    image: str = "maven:3.9.9-eclipse-temurin-21-alpine"

    # Writable tmpfs paths inside the container.
    # /tmp  — scratch space for Maven's own temp files and Maven local repo (/tmp/.m2)
    # /workspace/target — build output directory (bind-mounted from host)
    # Note: /root/.m2 is NOT listed here because we use /tmp/.m2 via -Dmaven.repo.local=/tmp/.m2
    # which is owned by the container user (uid 1000 cannot write to /root).
    tmpfs_mounts: list[str] = field(
        default_factory=lambda: ["/tmp"]
    )

    # Drop ALL Linux capabilities; grant none back.
    cap_drop_all: bool = True

    # Prevent privilege escalation inside the container.
    no_new_privileges: bool = True

    # Run as this UID:GID inside the container (not root).
    # Alpine-based Maven images have uid 1000 (maven user) available.
    run_as_user: str = "1000:1000"
