"""Workflow prompts for the agent's LLM calls.

System prompt is kept stable so it qualifies for provider-side prompt caching.
All volatile content (issue description, file contents) goes in user messages.

Tool calling: only `"auto"` is supported by Meta API (verified 2026-10-26).
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are an expert Java/Spring Boot software engineer.
Your job is to analyze a bug report, identify the root cause in the provided \
source files, propose a minimal fix, and generate a regression test.

Always respond in the exact JSON structure requested.
Never include markdown fences around JSON — return raw JSON only.
Never modify files that are not directly related to the reported bug.
Never include secrets, credentials, or environment variables in your output.\
"""

CACHE_KEY_SYSTEM = "swe-agent-system-v1"


def make_diagnosis_prompt(
    issue_description: str,
    file_contents: dict[str, str],
) -> str:
    """Prompt asking for root-cause diagnosis and patch plan.

    Returns a user-message string. Volatile content is appended AFTER
    the stable system prompt so the prefix can be cached.
    """
    files_block = "\n\n".join(
        f"=== {path} ===\n{content}" for path, content in file_contents.items()
    )
    return f"""\
## Issue description

{issue_description}

## Source files

{files_block}

## Task

Respond with a single JSON object with these fields:
{{
  "root_cause": "<one-paragraph explanation of the bug>",
  "affected_files": ["<path relative to repo root>", ...],
  "fix_description": "<what the fix does>",
  "unified_diff": "<complete unified diff ready to apply with git apply>",
  "confidence": "high" | "medium" | "low"
}}

The unified_diff must:
- Use proper unified diff format (--- a/path, +++ b/path, @@ lines)
- Only modify the minimum necessary lines
- Leave all other code unchanged
"""


def make_test_generation_prompt(
    issue_description: str,
    fix_description: str,
    patch_diff: str,
    existing_test_file: str,
    existing_test_content: str,
) -> str:
    """Prompt asking for a regression test for the fixed behaviour."""
    return f"""\
## Issue that was fixed

{issue_description}

## Fix applied

{fix_description}

## Diff applied

{patch_diff}

## Existing test file ({existing_test_file})

{existing_test_content}

## Task

Respond with a single JSON object:
{{
  "test_method_name": "<name of the new JUnit 5 test method>",
  "test_method_body": "<complete @Test method body to append to the existing test class>",
  "explanation": "<why this test exercises the fixed code path>"
}}

Rules:
- Do NOT wrap the body in a class — only the method body.
- The method name must not already exist in the existing test file.
- Use assertThrows / assertEquals from JUnit 5 Assertions only.
- Do not import anything not already imported in the existing test file.
"""
