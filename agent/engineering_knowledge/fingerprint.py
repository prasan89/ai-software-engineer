"""Issue fingerprinting and exact deduplication.

An issue fingerprint is a stable, deterministic hash over the normalized
issue content. Two issues with the same fingerprint are treated as exact
duplicates and can safely reuse the same verified fix.

Normalization rules (to reduce spurious mismatches):
- Lowercase the title and body.
- Strip leading/trailing whitespace and collapse runs of whitespace.
- Sort label names alphabetically.
- Ignore metadata that doesn't affect root cause: timestamps, author, URL.

The hash is SHA-256 over the canonical JSON representation of the normalized
fields. A 16-char hex prefix is used as the short fingerprint for display;
the full 64-char hex is used for storage keys.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class IssueInput:
    """Normalized representation of an inbound GitHub issue."""

    repo_full_name: str  # e.g. "owner/repo"
    issue_number: int
    title: str
    body: str
    labels: tuple[str, ...]  # sorted


@dataclass(frozen=True)
class IssueFingerprint:
    full_hex: str  # 64-char SHA-256 hex
    short_hex: str  # first 16 chars, for display only

    def __str__(self) -> str:
        return self.short_hex


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def fingerprint_issue(issue: IssueInput) -> IssueFingerprint:
    """Compute a stable fingerprint for an issue.

    The fingerprint is independent of issue number, author, timestamps, and
    URL — only content and labels determine the hash.
    """
    canonical = json.dumps(
        {
            "repo": issue.repo_full_name,
            "title": _normalize_text(issue.title),
            "body": _normalize_text(issue.body),
            "labels": sorted(issue.labels),
        },
        separators=(",", ":"),
        ensure_ascii=False,
        sort_keys=True,
    )
    full = hashlib.sha256(canonical.encode()).hexdigest()
    return IssueFingerprint(full_hex=full, short_hex=full[:16])


def fingerprint_from_raw(
    *,
    repo_full_name: str,
    issue_number: int,
    title: str,
    body: str,
    labels: list[str],
) -> IssueFingerprint:
    """Convenience constructor that accepts raw strings."""
    return fingerprint_issue(
        IssueInput(
            repo_full_name=repo_full_name,
            issue_number=issue_number,
            title=title,
            body=body,
            labels=tuple(sorted(labels)),
        )
    )
