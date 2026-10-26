"""Unit tests for issue fingerprinting."""

from __future__ import annotations

import pytest

from agent.engineering_knowledge.fingerprint import (
    IssueInput,
    fingerprint_from_raw,
    fingerprint_issue,
)

pytestmark = pytest.mark.unit


def _make(title: str = "NullPointerException in UserService", body: str = "body") -> IssueInput:
    return IssueInput(
        repo_full_name="owner/repo",
        issue_number=1,
        title=title,
        body=body,
        labels=("bug",),
    )


class TestFingerprintDeterminism:
    def test_same_content_produces_same_fingerprint(self) -> None:
        fp1 = fingerprint_issue(_make())
        fp2 = fingerprint_issue(_make())
        assert fp1.full_hex == fp2.full_hex

    def test_different_issue_numbers_same_content_are_equal(self) -> None:
        issue_a = IssueInput("owner/repo", 1, "title", "body", ("bug",))
        issue_b = IssueInput("owner/repo", 99, "title", "body", ("bug",))
        assert fingerprint_issue(issue_a).full_hex == fingerprint_issue(issue_b).full_hex

    def test_different_title_produces_different_fingerprint(self) -> None:
        fp1 = fingerprint_issue(_make(title="Error in LoginService"))
        fp2 = fingerprint_issue(_make(title="Error in LogoutService"))
        assert fp1.full_hex != fp2.full_hex

    def test_different_body_produces_different_fingerprint(self) -> None:
        fp1 = fingerprint_issue(_make(body="NPE at line 42"))
        fp2 = fingerprint_issue(_make(body="NPE at line 43"))
        assert fp1.full_hex != fp2.full_hex

    def test_different_repo_produces_different_fingerprint(self) -> None:
        a = IssueInput("owner/repo-a", 1, "title", "body", ("bug",))
        b = IssueInput("owner/repo-b", 1, "title", "body", ("bug",))
        assert fingerprint_issue(a).full_hex != fingerprint_issue(b).full_hex


class TestNormalization:
    def test_whitespace_collapsed_before_hashing(self) -> None:
        a = IssueInput("o/r", 1, "  NPE   in  Service  ", "body", ())
        b = IssueInput("o/r", 1, "NPE in Service", "body", ())
        assert fingerprint_issue(a).full_hex == fingerprint_issue(b).full_hex

    def test_case_insensitive(self) -> None:
        a = IssueInput("o/r", 1, "NPE IN SERVICE", "body", ())
        b = IssueInput("o/r", 1, "npe in service", "body", ())
        assert fingerprint_issue(a).full_hex == fingerprint_issue(b).full_hex

    def test_label_order_does_not_matter(self) -> None:
        a = IssueInput("o/r", 1, "t", "b", ("bug", "help wanted"))
        b = IssueInput("o/r", 1, "t", "b", ("help wanted", "bug"))
        assert fingerprint_issue(a).full_hex == fingerprint_issue(b).full_hex


class TestShortHex:
    def test_short_hex_is_prefix_of_full_hex(self) -> None:
        fp = fingerprint_issue(_make())
        assert fp.full_hex.startswith(fp.short_hex)
        assert len(fp.short_hex) == 16
        assert len(fp.full_hex) == 64

    def test_str_returns_short_hex(self) -> None:
        fp = fingerprint_issue(_make())
        assert str(fp) == fp.short_hex


class TestFingerprintFromRaw:
    def test_convenience_constructor_matches_dataclass(self) -> None:
        raw = fingerprint_from_raw(
            repo_full_name="owner/repo",
            issue_number=5,
            title="Bug",
            body="description",
            labels=["bug", "high-priority"],
        )
        direct = fingerprint_issue(
            IssueInput("owner/repo", 5, "bug", "description", ("bug", "high-priority"))
        )
        assert raw.full_hex == direct.full_hex
