"""Verified fix storage and compatibility checks.

A VerifiedFix is the artifact produced when a previous run successfully:
  1. Implemented a patch.
  2. Generated regression tests.
  3. Passed verification (all tests green, no regressions).

Fixes are keyed by IssueFingerprint. Before reusing a fix, FixStore checks
compatibility with the current repository state via a RepoContext hash.

Compatibility rules:
  - A fix is compatible if its repo_tree_hash matches the current tree hash
    of the files the patch touches.
  - If repo_tree_hash is None the fix was stored without tree verification;
    it is flagged as UNVERIFIED_COMPAT and must not be applied silently.
  - Fixes are never auto-applied to a different repo_full_name.

Storage is intentionally simple (in-memory for the scaffold). A persistent
backend (SQLite or a JSON file) will be added when the workflow is wired.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class CompatibilityStatus(StrEnum):
    COMPATIBLE = "compatible"
    TREE_MISMATCH = "tree_mismatch"
    WRONG_REPO = "wrong_repo"
    UNVERIFIED_COMPAT = "unverified_compat"
    NOT_FOUND = "not_found"


@dataclass(frozen=True)
class VerifiedFix:
    fingerprint_hex: str  # full 64-char hex from IssueFingerprint
    repo_full_name: str
    issue_number: int
    patch_unified_diff: str  # unified diff, ready to apply
    test_file_contents: dict[str, str]  # path -> content
    repo_tree_hash: str | None  # SHA of the touched-files tree at fix time
    verified_at_epoch: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompatibilityResult:
    status: CompatibilityStatus
    fix: VerifiedFix | None = None
    reason: str = ""


class FixStore:
    """In-memory store for verified fixes. Replace backend for persistence.

    All methods are synchronous at the scaffold stage; async wrappers will
    be added when the storage backend is swapped to async I/O.
    """

    def __init__(self) -> None:
        self._store: dict[str, VerifiedFix] = {}

    def save(self, fix: VerifiedFix) -> None:
        self._store[fix.fingerprint_hex] = fix

    def get(self, fingerprint_hex: str) -> VerifiedFix | None:
        return self._store.get(fingerprint_hex)

    def check_compatibility(
        self,
        fingerprint_hex: str,
        repo_full_name: str,
        current_tree_hash: str | None,
    ) -> CompatibilityResult:
        fix = self.get(fingerprint_hex)
        if fix is None:
            return CompatibilityResult(status=CompatibilityStatus.NOT_FOUND)
        if fix.repo_full_name != repo_full_name:
            return CompatibilityResult(
                status=CompatibilityStatus.WRONG_REPO,
                fix=fix,
                reason=(f"Fix was for {fix.repo_full_name}, current repo is {repo_full_name}"),
            )
        if fix.repo_tree_hash is None:
            return CompatibilityResult(
                status=CompatibilityStatus.UNVERIFIED_COMPAT,
                fix=fix,
                reason="Fix was stored without a repo tree hash; manual review required.",
            )
        if current_tree_hash is not None and fix.repo_tree_hash != current_tree_hash:
            return CompatibilityResult(
                status=CompatibilityStatus.TREE_MISMATCH,
                fix=fix,
                reason=(
                    f"Repo tree changed since fix was verified. "
                    f"Stored: {fix.repo_tree_hash[:12]}, "
                    f"Current: {current_tree_hash[:12]}"
                ),
            )
        return CompatibilityResult(status=CompatibilityStatus.COMPATIBLE, fix=fix)

    def __len__(self) -> int:
        return len(self._store)


def make_verified_fix(
    *,
    fingerprint_hex: str,
    repo_full_name: str,
    issue_number: int,
    patch_unified_diff: str,
    test_file_contents: dict[str, str],
    repo_tree_hash: str | None,
    metadata: dict[str, Any] | None = None,
) -> VerifiedFix:
    return VerifiedFix(
        fingerprint_hex=fingerprint_hex,
        repo_full_name=repo_full_name,
        issue_number=issue_number,
        patch_unified_diff=patch_unified_diff,
        test_file_contents=test_file_contents,
        repo_tree_hash=repo_tree_hash,
        verified_at_epoch=time.time(),
        metadata=metadata or {},
    )
