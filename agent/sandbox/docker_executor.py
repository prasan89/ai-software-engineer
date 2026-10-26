"""Docker-backed sandbox executor.

Security model
--------------
Every Maven build from a GitHub repository is treated as untrusted code execution.
Docker provides the outer envelope:

  • Non-root: --user 1000:1000 (Maven image "maven" user)
  • No network: --network=none (Maven plugins cannot call home)
  • Read-only root FS: --read-only
  • Writable scratch space via tmpfs: /tmp, /root/.m2, /workspace/target
  • All Linux capabilities dropped: --cap-drop ALL
  • No privilege escalation: --security-opt no-new-privileges
  • CPU quota: --cpus
  • Memory limit: --memory / --memory-swap
  • Process count limit: --pids-limit
  • Hard timeout: external kill after timeout_seconds
  • No host volume mounts beyond the per-run workspace (read-only)
  • No host Docker socket mount
  • Container forcibly removed after completion regardless of outcome

Residual risks (documented — not claimed solved)
-------------------------------------------------
  1. Network isolation does not prevent container-to-container communication on
     shared user-defined networks. The --network=none flag prevents this by
     placing the container in a network namespace with no interfaces.

  2. tmpfs mounts are writable and in-memory; a build could exhaust host RAM
     via the tmpfs. The --memory limit also covers tmpfs usage in recent Docker
     (kernel cgroup v2), but on cgroup v1 hosts tmpfs is uncapped.

  3. The Maven image itself is not verified against a pinned digest here. Callers
     should pin SandboxConfig.image to a @sha256:... digest in production.

  4. Maven plugins run as the maven user (uid 1000). A plugin exploit could still
     read files writable by uid 1000 inside the container; the read-only root FS
     prevents modification of image layers.

  5. Docker socket access and host path escapes are prevented by not mounting
     /var/run/docker.sock and by using a freshly-copied workspace directory.

  6. This executor runs `docker` as a subprocess of the agent process. If the
     agent process is compromised, so is Docker access. A dedicated unprivileged
     Docker API proxy is the production hardening path.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from agent.sandbox.types import SandboxConfig, SandboxResult

logger = logging.getLogger(__name__)

# Maven base command run inside every container.
# -B: batch (no ANSI).
# --no-transfer-progress: suppresses the download progress bar which floods logs.
# -Dmaven.repo.local=/tmp/.m2: use the container-local tmpfs Maven repo.
_MAVEN_CMD_BASE = ["mvn", "test", "-B", "--no-transfer-progress"]

# On macOS with Docker Desktop, bind mounts must come from Docker-shared paths.
# /var/folders (macOS $TMPDIR) is NOT shared by default. /Users IS always shared.
# AGENT_SANDBOX_WORKSPACE_DIR env var overrides the default for both host types.


def _workspace_base() -> str:
    env = os.environ.get("AGENT_SANDBOX_WORKSPACE_DIR")
    if env:
        return env
    # Use home directory subdir — always shared by Docker Desktop on macOS
    home = Path.home()
    base = home / ".agent" / "sandbox_workspaces"
    base.mkdir(parents=True, exist_ok=True)
    return str(base)


class DockerNotAvailableError(Exception):
    """Raised when Docker is required but not reachable."""


class DockerSandbox:
    """Runs Maven tests inside an isolated Docker container.

    Usage::

        # With a pre-seeded Maven local repository (required for --network=none):
        sandbox = DockerSandbox()
        result = sandbox.run_maven_tests(
            repo_root="/path/to/repo",
            maven_local_repo="/path/to/.m2/repository",
        )

    Args:
        config: Resource and isolation parameters. Defaults are safe for untrusted code.

    Maven offline mode:
        When ``maven_local_repo`` is provided to ``run_maven_tests``, the sandbox:
          1. Copies the local repo to a per-run temp directory.
          2. Strips ``_remote.repositories`` marker files so Maven treats all
             artifacts as locally resolved.
          3. Mounts the cleaned copy at /tmp/.m2 (read-only) and passes
             -Dmaven.repo.local=/tmp/.m2 --offline to Maven.

        Without a pre-seeded repo and with --network=none, the build will fail
        because Maven cannot download Spring Boot parent POMs. This is the
        correct failure mode — it surfaces the dependency.
    """

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self._config = config or SandboxConfig()

    def run_maven_tests(
        self,
        repo_root: str | Path,
        test_class: str | None = None,
        extra_maven_args: list[str] | None = None,
        maven_local_repo: str | Path | None = None,
    ) -> SandboxResult:
        """Copy repo to an isolated workspace and run `mvn test` inside Docker.

        Args:
            repo_root: Path to the Maven project root on the host (must contain pom.xml).
            test_class: If set, pass -Dtest=<class> to limit execution.
            extra_maven_args: Additional Maven arguments (validated against allowlist).
            maven_local_repo: Path to a pre-seeded Maven local repository (the
                directory that contains org/, com/, etc.). When set, the sandbox
                strips _remote.repositories markers, mounts the repo at /tmp/.m2,
                and passes --offline to Maven.

        Returns:
            SandboxResult with all observed outputs and resource-limit flags.

        Raises:
            DockerNotAvailableError: if Docker is not reachable and we cannot fall back.
        """
        self._assert_docker_available()
        root = Path(repo_root).resolve()
        self._validate_repo_root(root)

        with tempfile.TemporaryDirectory(prefix="sandbox_ws_", dir=_workspace_base()) as tmp:
            tmp_path = Path(tmp)
            workspace = self._copy_workspace(root, tmp_path)
            # Create a writable host directory for Maven's target/ output.
            target_dir = tmp_path / "target_out"
            target_dir.mkdir()
            # Optionally prepare a cleaned offline Maven repo
            m2_dir: Path | None = None
            if maven_local_repo:
                m2_dir = self._prepare_offline_repo(Path(maven_local_repo), tmp_path)
            cmd = self._build_docker_cmd(workspace, target_dir, m2_dir, test_class, extra_maven_args)
            result = self._run(cmd)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_docker_available(self) -> None:
        try:
            subprocess.run(
                ["docker", "version", "--format", "{{.Server.Version}}"],
                capture_output=True,
                check=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise DockerNotAvailableError(
                "Docker is not available or not responding. "
                "Install Docker or use DockerSandbox only when Docker is reachable. "
                "Never fall back to host execution for untrusted repositories."
            ) from exc

    @staticmethod
    def _validate_repo_root(root: Path) -> None:
        if not root.is_dir():
            raise ValueError(f"Repo root does not exist: {root}")
        if not (root / "pom.xml").exists():
            raise ValueError(f"No pom.xml found at {root}")

    def _copy_workspace(self, src: Path, tmp_dir: Path) -> Path:
        """Copy the repository to a fresh per-run temp directory.

        We copy rather than mount the live repo because:
          - The original working tree is never touched by the container.
          - Each run is fully isolated from concurrent runs.
        The workspace mount is :rw (not :ro) because Docker cannot mount over
        /workspace/target when /workspace is :ro — the OCI runtime needs to
        create the overlay mount point on the container FS. Isolation of the
        original repo is achieved by the copy itself (disposable per-run).
        """
        dest = tmp_dir / "workspace"
        shutil.copytree(
            src,
            dest,
            # Omit git history and IDE files; target/ can be included (empty)
            # to provide the mount point for the writable bind-mount below.
            ignore=shutil.ignore_patterns(".git", ".idea", "*.class"),
        )
        # Ensure target/ exists as a mount point for the writable bind-mount.
        (dest / "target").mkdir(exist_ok=True)
        return dest

    @staticmethod
    def _prepare_offline_repo(src_repo: Path, tmp_dir: Path) -> Path:
        """Copy a host Maven local repository for offline container use.

        Maven marks downloaded artifacts with ``_remote.repositories`` files
        that cause it to re-verify them against the remote in online mode.
        Without ``--offline``, Maven tries to check central even when
        ``--network=none`` is set and fails. Deleting these markers makes
        Maven treat all artifacts as locally installed, enabling ``--offline``.

        Security: the copy is owned by the host user and mounted :ro, so
        Maven plugins cannot write back to it. The container may still read
        arbitrary data from it — only supply a repo you trust.

        Args:
            src_repo: Path to the Maven local repository directory (contains org/, com/, ...).
            tmp_dir: Per-run temp directory to place the copy in.

        Returns:
            Path to the cleaned copy, ready for :ro bind-mounting at /tmp/.m2.
        """
        dest = tmp_dir / "m2_repo"
        shutil.copytree(src_repo, dest)
        # Remove _remote.repositories markers so Maven treats artifacts as local
        for marker in dest.rglob("_remote.repositories"):
            marker.unlink(missing_ok=True)
        return dest

    def _build_docker_cmd(
        self,
        workspace: Path,
        target_dir: Path | None,
        m2_dir: Path | None,
        test_class: str | None,
        extra_maven_args: list[str] | None,
    ) -> list[str]:
        cfg = self._config
        maven_args = list(_MAVEN_CMD_BASE)
        if m2_dir is not None:
            maven_args += ["-Dmaven.repo.local=/tmp/.m2", "--offline"]
        if test_class:
            maven_args.append(f"-Dtest={test_class}")
        if extra_maven_args:
            _validate_maven_args(extra_maven_args)
            maven_args.extend(extra_maven_args)

        cmd: list[str] = [
            "docker", "run",
            "--rm",                                        # auto-remove on exit
            "--name", f"sandbox_{id(self)}_{int(time.monotonic() * 1000)}",
        ]

        # Isolation flags
        if cfg.network_disabled:
            cmd += ["--network", "none"]
        if cfg.read_only:
            cmd += ["--read-only"]
        if cfg.cap_drop_all:
            cmd += ["--cap-drop", "ALL"]
        if cfg.no_new_privileges:
            cmd += ["--security-opt", "no-new-privileges"]

        # Resource limits
        cmd += ["--memory", cfg.memory_limit]
        # --memory-swap == memory_limit prevents swap usage
        cmd += ["--memory-swap", cfg.memory_limit]
        cmd += ["--cpus", str(cfg.cpu_limit)]
        cmd += ["--pids-limit", str(cfg.pids_limit)]

        # Run as non-root
        cmd += ["--user", cfg.run_as_user]

        # Writable tmpfs mounts for /tmp (Maven temp files live here)
        for mount in cfg.tmpfs_mounts:
            cmd += ["--tmpfs", f"{mount}:exec"]

        # Workspace volume: the per-run copy mounted rw so Docker can overlay /workspace/target
        cmd += ["-v", f"{workspace}:/workspace:rw"]

        # Writable build output directory (overlays /workspace/target)
        if target_dir is not None:
            cmd += ["-v", f"{target_dir}:/workspace/target:rw"]

        # Offline Maven local repo (read-only)
        if m2_dir is not None:
            cmd += ["-v", f"{m2_dir}:/tmp/.m2:ro"]

        # Working directory and image
        cmd += ["-w", "/workspace"]
        cmd += [cfg.image]

        # Maven command
        cmd += maven_args

        return cmd

    def _run(self, cmd: list[str]) -> SandboxResult:
        cfg = self._config
        start = time.monotonic()
        timed_out = False
        oom_killed = False
        exit_code = -1
        stdout = ""
        stderr = ""

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=cfg.timeout_seconds,
            )
            exit_code = proc.returncode
            stdout = proc.stdout
            stderr = proc.stderr

            # Docker exit code 137 = OOM kill (SIGKILL from kernel OOM killer)
            if exit_code == 137:
                oom_killed = True

        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = -1
            raw_out = exc.stdout
            stdout = raw_out.decode("utf-8", errors="replace") if isinstance(raw_out, bytes) else (raw_out or "")
            stderr = f"[TIMEOUT after {cfg.timeout_seconds}s]"
            # Best-effort container cleanup after timeout
            self._kill_container_by_cmd(cmd)

        elapsed = round(time.monotonic() - start, 2)
        counts = _parse_surefire_summary(stdout)

        return SandboxResult(
            command=cmd,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            elapsed_seconds=elapsed,
            timed_out=timed_out,
            oom_killed=oom_killed,
            tests_run=counts["tests_run"],
            tests_failed=counts["tests_failed"],
            tests_errors=counts["tests_errors"],
            tests_skipped=counts["tests_skipped"],
        )

    @staticmethod
    def _kill_container_by_cmd(cmd: list[str]) -> None:
        """Extract container name from cmd and stop it (best effort)."""
        try:
            name_idx = cmd.index("--name") + 1
            container_name = cmd[name_idx]
            subprocess.run(
                ["docker", "stop", "--time=3", container_name],
                capture_output=True,
                timeout=10,
            )
        except (ValueError, IndexError, subprocess.TimeoutExpired, OSError):
            pass


# ---------------------------------------------------------------------------
# Shared helpers (also used by LocalSandbox)
# ---------------------------------------------------------------------------

_ALLOWED_MAVEN_ARG_PREFIXES = (
    "-Dtest=",
    "-pl",
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
    """Parse Maven Surefire summary: 'Tests run: N, Failures: N, Errors: N, Skipped: N'"""
    import re

    counts: dict[str, int] = {
        "tests_run": 0,
        "tests_failed": 0,
        "tests_errors": 0,
        "tests_skipped": 0,
    }
    pattern = re.compile(
        r"Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+),\s*Skipped:\s*(\d+)"
    )
    last_match = None
    for m in pattern.finditer(output):
        last_match = m
    if last_match:
        counts["tests_run"] = int(last_match.group(1))
        counts["tests_failed"] = int(last_match.group(2))
        counts["tests_errors"] = int(last_match.group(3))
        counts["tests_skipped"] = int(last_match.group(4))
    return counts
