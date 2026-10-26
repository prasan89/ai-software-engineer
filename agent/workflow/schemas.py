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
        if not v.strip().startswith("---"):
            raise ValueError("unified_diff must start with '--- '")
        return v


class TestGenerationResponse(BaseModel):
    test_method_name: str
    test_method_body: str
    explanation: str
