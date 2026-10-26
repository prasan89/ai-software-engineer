"""Structured output types for LLM responses."""

from __future__ import annotations

from pydantic import BaseModel, field_validator


class DiagnosisResponse(BaseModel):
    root_cause: str
    affected_files: list[str]
    fix_description: str
    unified_diff: str
    confidence: str = "medium"

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: str) -> str:
        if v not in ("high", "medium", "low"):
            return "medium"
        return v

    @field_validator("unified_diff")
    @classmethod
    def validate_diff_format(cls, v: str) -> str:
        stripped = v.strip()
        # Accept standard unified diff (--- a/...) or git extended format
        # (diff --git a/... b/...) — normalize to plain unified diff by
        # stripping the leading "diff --git" header lines.
        if stripped.startswith("diff --git"):
            lines = stripped.splitlines()
            # Find the first "--- a/" line and use everything from there
            for i, line in enumerate(lines):
                if line.startswith("--- ") or line.startswith("---\t"):
                    return "\n".join(lines[i:]) + "\n"
            raise ValueError("unified_diff has diff --git header but no --- line found")
        if not stripped.startswith("---"):
            raise ValueError("unified_diff must start with '--- ' or 'diff --git'")
        return v


class TestGenerationResponse(BaseModel):
    test_method_name: str
    test_method_body: str
    explanation: str
