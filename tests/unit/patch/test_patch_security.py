"""Security regression tests for agent/patch/__init__.py.

Covers path traversal, null byte injection, absolute path rejection,
and the _check_patch_paths helper directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_diff(target: str = "src/Foo.java") -> str:
    """Minimal valid-looking unified diff for the given target path."""
    return (
        f"--- a/{target}\n"
        f"+++ b/{target}\n"
        "@@ -1,3 +1,4 @@\n"
        " class Foo {\n"
        "+    // comment\n"
        "     void bar() {}\n"
        " }\n"
    )


# ---------------------------------------------------------------------------
# _check_patch_paths — unit tests for the path-validation helper
# ---------------------------------------------------------------------------


class TestCheckPatchPaths:
    def test_valid_relative_path_passes(self, tmp_path: Path) -> None:
        from agent.patch import _check_patch_paths
        diff = _make_valid_diff("src/Foo.java")
        # Create the target so resolve() can canonicalize it
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "Foo.java").write_text("x")
        assert _check_patch_paths(diff, tmp_path) == ""

    def test_path_traversal_dot_dot_rejected(self, tmp_path: Path) -> None:
        from agent.patch import _check_patch_paths
        diff = "--- a/../etc/passwd\n+++ b/../etc/passwd\n@@ -1 +1 @@\nroot\n"
        error = _check_patch_paths(diff, tmp_path)
        assert error != ""
        assert "escapes" in error or "absolute" in error or "traversal" in error.lower() or "rejected" in error

    def test_absolute_path_rejected(self, tmp_path: Path) -> None:
        from agent.patch import _check_patch_paths
        diff = "--- /etc/passwd\n+++ /etc/passwd\n@@ -1 +1 @@\nroot\n"
        error = _check_patch_paths(diff, tmp_path)
        assert error != ""
        assert "absolute" in error

    def test_dev_null_is_allowed(self, tmp_path: Path) -> None:
        from agent.patch import _check_patch_paths
        diff = "--- /dev/null\n+++ b/src/New.java\n@@ -0,0 +1 @@\n+x\n"
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "New.java").write_text("x")
        assert _check_patch_paths(diff, tmp_path) == ""

    def test_path_without_prefix_that_escapes(self, tmp_path: Path) -> None:
        from agent.patch import _check_patch_paths
        # After stripping a/ prefix this becomes ../../etc/passwd relative to root
        diff = "--- a/../../etc/passwd\n+++ b/../../etc/passwd\n@@ -1 +1 @@\nroot\n"
        error = _check_patch_paths(diff, tmp_path)
        assert error != ""

    def test_nested_valid_path_passes(self, tmp_path: Path) -> None:
        from agent.patch import _check_patch_paths
        (tmp_path / "src" / "main" / "java").mkdir(parents=True)
        (tmp_path / "src" / "main" / "java" / "Calc.java").write_text("x")
        diff = _make_valid_diff("src/main/java/Calc.java")
        assert _check_patch_paths(diff, tmp_path) == ""


# ---------------------------------------------------------------------------
# apply_patch — integration against a real filesystem
# ---------------------------------------------------------------------------


class TestApplyPatchSecurity:
    def test_null_byte_in_diff_rejected(self, tmp_path: Path) -> None:
        from agent.patch import apply_patch
        diff = "--- a/Foo.java\n+++ b/Foo.java\n@@ -1 +1 @@\n+\x00evil\n"
        result = apply_patch(tmp_path, diff)
        assert result.success is False
        assert "null" in result.error.lower()

    def test_absolute_path_in_diff_rejected(self, tmp_path: Path) -> None:
        # Make tmp_path a git repo so apply_patch gets past the root check
        import subprocess

        from agent.patch import apply_patch
        subprocess.run(["git", "init", "-b", "main"], cwd=tmp_path, capture_output=True)
        diff = "--- /etc/passwd\n+++ /etc/passwd\n@@ -1 +1 @@\nevil\n"
        result = apply_patch(tmp_path, diff)
        assert result.success is False
        assert "absolute" in result.error

    def test_traversal_via_dot_dot_rejected(self, tmp_path: Path) -> None:
        import subprocess

        from agent.patch import apply_patch
        subprocess.run(["git", "init", "-b", "main"], cwd=tmp_path, capture_output=True)
        diff = "--- a/../secret\n+++ b/../secret\n@@ -1 +1 @@\nevil\n"
        result = apply_patch(tmp_path, diff)
        assert result.success is False

    def test_valid_patch_accepted(self, tmp_path: Path) -> None:
        import subprocess

        from agent.patch import apply_patch
        env = {
            "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
            **__import__("os").environ,
        }
        src = tmp_path / "Foo.java"
        src.write_text("class Foo {\n    void bar() {}\n}\n")
        subprocess.run(["git", "init", "-b", "main"], cwd=tmp_path, capture_output=True, env=env)
        subprocess.run(["git", "add", "Foo.java"], cwd=tmp_path, capture_output=True, env=env)
        subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True, env=env)
        diff = (
            "--- a/Foo.java\n"
            "+++ b/Foo.java\n"
            "@@ -1,3 +1,4 @@\n"
            " class Foo {\n"
            "+    // added\n"
            "     void bar() {}\n"
            " }\n"
        )
        result = apply_patch(tmp_path, diff)
        assert result.success is True
        assert "// added" in src.read_text()


# ---------------------------------------------------------------------------
# Docker sandbox — test_class validation
# ---------------------------------------------------------------------------


class TestValidateTestClass:
    def test_simple_class_name_accepted(self) -> None:
        from agent.sandbox.docker_executor import _validate_test_class
        _validate_test_class("CalculatorServiceTest")  # should not raise

    def test_class_with_method_selector_accepted(self) -> None:
        from agent.sandbox.docker_executor import _validate_test_class
        _validate_test_class("CalculatorServiceTest#divide_null")

    def test_wildcard_accepted(self) -> None:
        from agent.sandbox.docker_executor import _validate_test_class
        _validate_test_class("com.example.*Test")

    def test_multiple_classes_accepted(self) -> None:
        from agent.sandbox.docker_executor import _validate_test_class
        _validate_test_class("FooTest,BarTest")

    def test_space_injection_rejected(self) -> None:
        from agent.sandbox.docker_executor import _validate_test_class
        with pytest.raises(ValueError, match="Invalid test_class"):
            _validate_test_class("FooTest -Devil=true")

    def test_semicolon_injection_rejected(self) -> None:
        from agent.sandbox.docker_executor import _validate_test_class
        with pytest.raises(ValueError, match="Invalid test_class"):
            _validate_test_class("FooTest;-Devil=true")

    def test_shell_metachar_rejected(self) -> None:
        from agent.sandbox.docker_executor import _validate_test_class
        with pytest.raises(ValueError, match="Invalid test_class"):
            _validate_test_class("FooTest$(whoami)")

    def test_empty_string_rejected(self) -> None:
        from agent.sandbox.docker_executor import _validate_test_class
        with pytest.raises(ValueError):
            _validate_test_class("")


# ---------------------------------------------------------------------------
# RunManager — stale eviction includes failed runs
# ---------------------------------------------------------------------------


class TestRunManagerEviction:
    def test_failed_run_evicts_after_max_age(self) -> None:
        """Failed runs (error set, no report) must evict to prevent memory leak."""
        import time

        from agent.api.run_manager import MAX_AGE_SECONDS, RunManager

        mgr = RunManager()
        mgr.create("aa00bb11")
        mgr.fail("aa00bb11", "something went wrong")

        rec = mgr.get("aa00bb11")
        assert rec is not None
        # Backdate the created_at so it looks stale
        rec.created_at = time.monotonic() - (MAX_AGE_SECONDS + 10)

        # _evict_stale is called by create(); trigger it
        mgr.create("cc00dd11")
        assert mgr.get("aa00bb11") is None, "Stale failed run was not evicted"

    def test_active_run_not_evicted(self) -> None:
        """In-progress runs (no report, no error) must not be evicted."""
        import time

        from agent.api.run_manager import MAX_AGE_SECONDS, RunManager

        mgr = RunManager()
        mgr.create("ee00ff11")

        rec = mgr.get("ee00ff11")
        assert rec is not None
        rec.created_at = time.monotonic() - (MAX_AGE_SECONDS + 10)

        mgr.create("gg00hh11")
        # In-progress run should still be there (it has no report and no error)
        assert mgr.get("ee00ff11") is not None


# ---------------------------------------------------------------------------
# Orchestrator — _inject_test_method rejects symlink escape
# ---------------------------------------------------------------------------


class TestInjectTestMethodBoundsCheck:
    def test_symlink_escape_prevented(self, tmp_path: Path) -> None:
        """If abs_path resolves outside the repo root, write is refused."""
        from agent.providers.base import BaseLLMProvider
        from agent.providers.types import LLMRequest, LLMResponse, ModelCapability, ProviderInfo
        from agent.repo import FileEntry, RepoContext
        from agent.workflow.orchestrator import WorkflowOrchestrator
        from agent.workflow.schemas import TestGenerationResponse

        class _FakeProvider(BaseLLMProvider):
            @property
            def info(self) -> ProviderInfo:
                return ProviderInfo(name="fake", base_url="", auth_header="", docs_url="")
            async def chat(self, request: LLMRequest) -> LLMResponse:
                raise RuntimeError("not called")
            def supports(self, c: ModelCapability) -> bool:
                return False
            def estimated_cost_usd(self, p: int, c: int) -> float:
                return 0.0
            def model_id(self) -> str:
                return "fake"

        # Create a file OUTSIDE the repo root
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        outside_file = outside_dir / "OutsideTest.java"
        outside_file.write_text("class OutsideTest {\n    void existing() {}\n}\n")

        # The repo root is a different directory
        repo_root = tmp_path / "repo"
        repo_root.mkdir()

        entry = FileEntry(
            rel_path="src/test/OutsideTest.java",
            abs_path=str(outside_file),  # points outside repo root
            language="java",
            size_bytes=outside_file.stat().st_size,
            git_blob_sha="",
        )
        repo = RepoContext(root=repo_root, files=[entry])

        orch = WorkflowOrchestrator(provider=_FakeProvider())
        tr = TestGenerationResponse(
            test_method_name="injected",
            test_method_body="assertTrue(true);",
            explanation="test",
        )

        original_content = outside_file.read_text()
        orch._inject_test_method(repo, tr)

        # File outside repo root must be unchanged
        assert outside_file.read_text() == original_content
