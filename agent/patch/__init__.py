"""Patch generation and application.

Applies a unified diff to a repository worktree using `git apply`.
The apply is always preceded by --check to avoid partial mutations.

Security note: the diff string comes from the LLM and is untrusted.
`git apply --check` validates it before any file is modified.
Path traversal is checked before applying.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PatchResult:
    success: bool
    modified_files: list[str]
    stdout: str
    stderr: str
    error: str = ""


def apply_patch(repo_root: str | Path, unified_diff: str) -> PatchResult:
    """Validate and apply a unified diff to the repository.

    Steps:
      1. Write the diff to a temporary file.
      2. Run `git apply --check` — aborts if the patch does not apply cleanly.
      3. Run `git apply` — applies the patch.
      4. Return the list of modified files.

    Args:
        repo_root: Absolute path to the repository root.
        unified_diff: Unified diff string (as would come from the LLM).

    Returns:
        PatchResult with success flag and modified file list.
    """
    root = Path(repo_root).resolve()
    if not root.is_dir():
        return PatchResult(
            success=False,
            modified_files=[],
            stdout="",
            stderr="",
            error=f"Repo root does not exist: {root}",
        )

    # Reject patches that reference paths outside the repository
    if "../" in unified_diff or "\\..\\" in unified_diff:
        return PatchResult(
            success=False,
            modified_files=[],
            stdout="",
            stderr="",
            error="Patch rejected: contains path traversal sequence",
        )

    with tempfile.NamedTemporaryFile(suffix=".patch", mode="w", delete=False) as f:
        f.write(unified_diff)
        patch_file = f.name

    try:
        # Step 1: dry-run check
        check = subprocess.run(
            ["git", "apply", "--check", patch_file],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if check.returncode != 0:
            return PatchResult(
                success=False,
                modified_files=[],
                stdout=check.stdout,
                stderr=check.stderr,
                error="git apply --check failed; patch does not apply cleanly",
            )

        # Step 2: get list of files the patch will touch
        stat = subprocess.run(
            ["git", "apply", "--stat", patch_file],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=30,
        )
        modified = _parse_stat_files(stat.stdout)

        # Step 3: apply
        apply = subprocess.run(
            ["git", "apply", patch_file],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if apply.returncode != 0:
            return PatchResult(
                success=False,
                modified_files=[],
                stdout=apply.stdout,
                stderr=apply.stderr,
                error="git apply failed after --check passed",
            )

        return PatchResult(
            success=True,
            modified_files=modified,
            stdout=apply.stdout,
            stderr=apply.stderr,
        )
    finally:
        Path(patch_file).unlink(missing_ok=True)


def _parse_stat_files(stat_output: str) -> list[str]:
    files = []
    for line in stat_output.splitlines():
        line = line.strip()
        if "|" in line:
            path = line.split("|")[0].strip()
            if path:
                files.append(path)
    return files
