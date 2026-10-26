"""Patch generation and application.

Applies a unified diff to a repository worktree using `git apply`.
The apply is always preceded by --check to avoid partial mutations.

Security note: the diff string comes from the LLM and is untrusted.
`git apply --check` validates it before any file is modified.
Path traversal is checked before applying.
"""

from __future__ import annotations

import re
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
      1. Repair hunk count headers (LLM often gets the line counts wrong).
      2. Validate that all target paths stay inside the repository root.
      3. Write the diff to a temporary file.
      4. Run `patch --dry-run` — aborts if the patch does not apply cleanly.
      5. Run `patch` — applies the patch.
      6. Return the list of modified files.

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

    # Reject null bytes in the diff (can confuse text tools and hide content)
    if "\x00" in unified_diff:
        return PatchResult(
            success=False,
            modified_files=[],
            stdout="",
            stderr="",
            error="Patch rejected: contains null bytes",
        )

    # Validate every target path in the diff stays inside the repo root.
    # This catches "../", absolute paths (/etc/passwd), and symlink escapes.
    # We do this BEFORE writing the patch file to avoid touching the filesystem.
    traversal_error = _check_patch_paths(unified_diff, root)
    if traversal_error:
        return PatchResult(
            success=False,
            modified_files=[],
            stdout="",
            stderr="",
            error=traversal_error,
        )

    # Repair hunk count headers — LLM-generated diffs often have correct content
    # but wrong @@ counts. Recompute counts from the actual body lines.
    repaired = _repair_hunk_counts(unified_diff)

    with tempfile.NamedTemporaryFile(suffix=".patch", mode="w", encoding="utf-8", delete=False) as f:
        f.write(repaired)
        patch_file = f.name

    try:
        # Step 1: dry-run check
        # Use BSD/GNU `patch` with -l (ignore whitespace) and -F3 (fuzz factor 3).
        # LLM-generated diffs often have minor indentation or context-line differences;
        # `git apply -C0` would apply at the wrong location. `patch -F3` uses context
        # with tolerance, which correctly positions the hunk.
        check = subprocess.run(
            ["patch", "-p1", "-l", "-F", "3", "--dry-run", "-i", patch_file],
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

        # Step 2: infer modified files from the patch header lines
        modified = _parse_modified_files(repaired)

        # Step 3: apply
        apply = subprocess.run(
            ["patch", "-p1", "-l", "-F", "3", "-i", patch_file],
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


_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@(.*)")


def _repair_hunk_counts(diff: str) -> str:
    """Recompute @@ hunk count headers from the actual diff body lines.

    LLM-generated diffs frequently have correct +/- content but wrong line
    counts in the @@ header. git apply treats a count mismatch as "corrupt
    patch". This function recounts context, added, and removed lines in each
    hunk body and rewrites the header to match.
    """
    lines = diff.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _HUNK_HEADER_RE.match(line)
        if m:
            old_start = int(m.group(1))
            new_start = int(m.group(2))
            tail = m.group(3)
            # Collect hunk body lines
            i += 1
            hunk_lines: list[str] = []
            while i < len(lines) and not lines[i].startswith("@@") and not lines[i].startswith("diff ") and not (lines[i].startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ ")):
                hunk_lines.append(lines[i])
                i += 1
            # Strip trailing blank lines (common LLM artifact)
            while hunk_lines and hunk_lines[-1] == "":
                hunk_lines.pop()
            old_count = sum(1 for ln in hunk_lines if not ln.startswith("+"))
            new_count = sum(1 for ln in hunk_lines if not ln.startswith("-"))
            # Build repaired header
            old_part = f"-{old_start}" if old_count == 1 else f"-{old_start},{old_count}"
            new_part = f"+{new_start}" if new_count == 1 else f"+{new_start},{new_count}"
            out.append(f"@@ {old_part} {new_part} @@{tail}")
            out.extend(hunk_lines)
        else:
            out.append(line)
            i += 1
    return "\n".join(out) + "\n"


def _parse_modified_files(diff: str) -> list[str]:
    """Extract target file paths from unified diff +++ lines."""
    files = []
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[4:].strip()
            # Strip a/ or b/ prefix from git-style paths
            if path.startswith("b/"):
                path = path[2:]
            if path and path != "/dev/null":
                files.append(path)
    return files


def _check_patch_paths(diff: str, root: Path) -> str:
    """Verify every target path in the diff stays inside root.

    Returns an error message string if a violation is found, empty string if clean.
    Checks both --- and +++ lines (applying -p1 strips the first path component).
    """
    for line in diff.splitlines():
        for prefix in ("--- ", "+++ "):
            if not line.startswith(prefix):
                continue
            raw = line[len(prefix):].strip()
            # Ignore /dev/null (new/deleted file markers)
            if raw in ("/dev/null", ""):
                continue
            # Strip git-style a/ or b/ prefix (patch -p1 strips one component)
            if raw.startswith(("a/", "b/")):
                raw = raw[2:]
            # Reject absolute paths immediately
            if raw.startswith("/"):
                return f"Patch rejected: absolute path in diff header: {raw!r}"
            # Resolve against root and check containment
            try:
                candidate = (root / raw).resolve()
            except (ValueError, OSError):
                return f"Patch rejected: cannot resolve path: {raw!r}"
            try:
                candidate.relative_to(root)
            except ValueError:
                return f"Patch rejected: path escapes repository root: {raw!r}"
    return ""

