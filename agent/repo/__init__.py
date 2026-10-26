"""Repository preparation and analysis.

For the vertical slice we support local path repositories only.
Remote clone (git clone) will be added once the local workflow is verified.

Security note: all paths are validated and canonicalized before any
filesystem operation. Repository code is never executed on the host
directly — execution happens only through the sandbox module.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# File extensions we include in context for Java/Spring Boot analysis
JAVA_SOURCE_EXTENSIONS = frozenset({".java", ".xml", ".properties", ".yaml", ".yml", ".gradle"})
# Hard limits to prevent feeding too much context to the model
MAX_FILES_IN_INDEX = 500
MAX_FILE_BYTES = 64 * 1024  # 64 KB per file for context


@dataclass
class FileEntry:
    rel_path: str       # relative to repo root, always forward-slash
    abs_path: str
    size_bytes: int
    git_blob_sha: str   # empty string if file is not tracked by git
    language: str       # "java", "xml", "properties", "yaml", "other"


@dataclass
class RepoContext:
    root: Path
    files: list[FileEntry] = field(default_factory=list)
    tree_hash: str = ""
    head_sha: str = ""

    def file_map(self) -> dict[str, FileEntry]:
        return {f.rel_path: f for f in self.files}

    def source_files(self) -> list[FileEntry]:
        return [f for f in self.files if f.rel_path.endswith(".java")]


def prepare_repo(repo_path: str | Path) -> RepoContext:
    """Prepare and index a local repository for analysis.

    Args:
        repo_path: Absolute path to the repository root. Must exist.

    Returns:
        RepoContext with indexed source files and tree hash.

    Raises:
        ValueError: if the path does not exist or is not a directory.
    """
    root = Path(repo_path).resolve()
    if not root.is_dir():
        raise ValueError(f"Repository path does not exist or is not a directory: {root}")

    head_sha = _get_head_sha(root)
    git_blobs = _get_git_blobs(root)

    files: list[FileEntry] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if _should_skip(path, root):
            continue
        if len(files) >= MAX_FILES_IN_INDEX:
            break
        rel = path.relative_to(root).as_posix()
        ext = path.suffix.lower()
        if ext not in JAVA_SOURCE_EXTENSIONS:
            continue
        size = path.stat().st_size
        files.append(
            FileEntry(
                rel_path=rel,
                abs_path=str(path),
                size_bytes=size,
                git_blob_sha=git_blobs.get(rel, ""),
                language=_classify_language(ext),
            )
        )

    tree_hash = compute_tree_hash({f.rel_path: f.git_blob_sha for f in files if f.git_blob_sha})
    return RepoContext(root=root, files=files, tree_hash=tree_hash, head_sha=head_sha)


def read_file_content(entry: FileEntry, max_bytes: int = MAX_FILE_BYTES) -> str:
    """Read file content up to max_bytes, returning a UTF-8 string."""
    path = Path(entry.abs_path)
    raw = path.read_bytes()[:max_bytes]
    return raw.decode("utf-8", errors="replace")


def compute_tree_hash(file_paths_and_shas: dict[str, str]) -> str:
    """Stable SHA-256 over {path: git_blob_sha} — used as CacheKey.tree_hash."""
    canonical = json.dumps(
        dict(sorted(file_paths_and_shas.items())),
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _get_head_sha(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def _get_git_blobs(root: Path) -> dict[str, str]:
    """Return {rel_path: blob_sha} for all tracked files via git ls-tree."""
    try:
        result = subprocess.run(
            ["git", "ls-tree", "-r", "HEAD", "--format=%(objectname) %(path)"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        blobs: dict[str, str] = {}
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    blobs[parts[1]] = parts[0]
        return blobs
    except Exception:
        return {}


def _should_skip(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    parts = rel.parts
    return any(
        p in {".git", "target", "build", ".gradle", "node_modules", "__pycache__", ".venv"}
        for p in parts
    )


def _classify_language(ext: str) -> str:
    return {".java": "java", ".xml": "xml", ".properties": "properties"}.get(ext, "yaml" if ext in {".yaml", ".yml"} else "other")
