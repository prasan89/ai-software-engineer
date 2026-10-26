"""Unit tests for FixStore."""

from __future__ import annotations

import pytest

from agent.engineering_knowledge.fix_store import (
    CompatibilityStatus,
    FixStore,
    make_verified_fix,
)

pytestmark = pytest.mark.unit

TREE_A = "a" * 64
TREE_B = "b" * 64
FP = "f" * 64


def _fix(repo: str = "owner/repo", tree_hash: str | None = TREE_A) -> object:
    return make_verified_fix(
        fingerprint_hex=FP,
        repo_full_name=repo,
        issue_number=1,
        patch_unified_diff="--- a/Foo.java\n+++ b/Foo.java\n",
        test_file_contents={"src/test/FooTest.java": "// test"},
        repo_tree_hash=tree_hash,
    )


class TestFixStoreSaveAndGet:
    def test_save_and_retrieve(self) -> None:
        store = FixStore()
        fix = _fix()
        store.save(fix)  # type: ignore[arg-type]
        assert store.get(FP) is fix

    def test_get_missing_returns_none(self) -> None:
        store = FixStore()
        assert store.get("0" * 64) is None

    def test_len_increments(self) -> None:
        store = FixStore()
        assert len(store) == 0
        store.save(_fix())  # type: ignore[arg-type]
        assert len(store) == 1


class TestCompatibilityChecks:
    def test_compatible_same_tree(self) -> None:
        store = FixStore()
        store.save(_fix())  # type: ignore[arg-type]
        result = store.check_compatibility(FP, "owner/repo", TREE_A)
        assert result.status == CompatibilityStatus.COMPATIBLE
        assert result.fix is not None

    def test_tree_mismatch(self) -> None:
        store = FixStore()
        store.save(_fix())  # type: ignore[arg-type]
        result = store.check_compatibility(FP, "owner/repo", TREE_B)
        assert result.status == CompatibilityStatus.TREE_MISMATCH
        assert "Stored" in result.reason

    def test_wrong_repo(self) -> None:
        store = FixStore()
        store.save(_fix())  # type: ignore[arg-type]
        result = store.check_compatibility(FP, "other/repo", TREE_A)
        assert result.status == CompatibilityStatus.WRONG_REPO

    def test_not_found(self) -> None:
        store = FixStore()
        result = store.check_compatibility(FP, "owner/repo", TREE_A)
        assert result.status == CompatibilityStatus.NOT_FOUND

    def test_unverified_compat_when_no_tree_hash(self) -> None:
        store = FixStore()
        store.save(_fix(tree_hash=None))  # type: ignore[arg-type]
        result = store.check_compatibility(FP, "owner/repo", TREE_A)
        assert result.status == CompatibilityStatus.UNVERIFIED_COMPAT

    def test_compatible_when_current_tree_hash_is_none(self) -> None:
        store = FixStore()
        store.save(_fix())  # type: ignore[arg-type]
        result = store.check_compatibility(FP, "owner/repo", current_tree_hash=None)
        assert result.status == CompatibilityStatus.COMPATIBLE
