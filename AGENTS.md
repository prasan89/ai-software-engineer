# AI Software Engineer Agent — Repository Guidelines

## Purpose

This repository implements an AI Software Engineer Agent for the Meta Global AI
Developer Hackathon. The agent accepts a GitHub issue, analyzes a Java/Spring Boot
codebase, implements a fix, generates regression tests, and verifies the solution.

A mandatory differentiator is the **Engineering Knowledge & Cost Optimizer**: exact
duplicate detection, verified fix reuse, repository-aware cache invalidation,
resumable checkpoints, model routing, and measurable token savings.

---

## Architecture

```
ai-software-engineer-agent/
  agent/                   # Core Python package
    api/                   # FastAPI application and route handlers
    providers/             # Provider-neutral LLM interface + adapters
    engineering_knowledge/ # Deduplication, fix reuse, cache invalidation
    cost_optimizer/        # Budget, usage accounting, model routing
    workflow/              # Typed states, run events, orchestrator
    repo/                  # Repository cloning, file indexing, worktree ops
    patch/                 # Patch generation and application
    testgen/               # Regression test generation
    sandbox/               # Docker-based isolated execution
    verification/          # Test run interpretation and outcome reporting
  tests/                   # Pytest test suite (mirrors agent/ structure)
  .env.example             # Placeholder env vars — NO real secrets
  pyproject.toml           # Package metadata, tool config
  Dockerfile               # Agent container image
  docker-compose.yml       # Local development services
```

Dependencies flow inward: domain types → engineering_knowledge/cost_optimizer →
providers → workflow → api. The api layer must not import from workflow internals
directly; it drives the workflow through the orchestrator interface.

---

## LLM Providers

The primary provider is **Meta's official Model API** (Llama family, endpoint
`https://api.llama.com/v1/`). A provider-neutral interface (`agent/providers/base.py`)
allows Anthropic Claude and other providers to be added as optional fallbacks.

Rules:
- Never hard-code a model identifier without a corresponding constant and a comment
  citing the documentation version it was verified against.
- Never fall over to a paid fallback provider unless `FALLBACK_PROVIDER_ENABLED=true`
  and a `FALLBACK_BUDGET_USD` is set.
- Usage (prompt tokens, completion tokens, cost) must be recorded for every call.
- All API keys live in environment variables. Never commit real keys.

---

## Engineering Knowledge & Cost Optimizer

This subsystem is the primary hackathon differentiator. Any change affecting its
contracts requires explicit review.

Interfaces that must remain stable across refactors:

| Interface | Location | Purpose |
|-----------|----------|---------|
| `IssueFingerprinter` | `engineering_knowledge/fingerprint.py` | Exact deduplication |
| `FixStore` | `engineering_knowledge/fix_store.py` | Verified fix storage + compatibility |
| `CacheInvalidator` | `engineering_knowledge/cache.py` | Repository-aware invalidation |
| `CheckpointManager` | `engineering_knowledge/checkpoint.py` | Resumable runs |
| `UsageTracker` | `cost_optimizer/usage.py` | Per-call token + cost accounting |
| `ModelRouter` | `cost_optimizer/router.py` | Complexity-based model selection |
| `CacheMetrics` | `cost_optimizer/metrics.py` | Hit rate, reuse savings reporting |

Cache keys must cover all result-affecting inputs. Define invalidation triggers and
TTLs explicitly. Validate freshness before reporting cached state as current.

---

## Security

- All repository code runs inside a Docker sandbox (`agent/sandbox/`). Never execute
  untrusted code directly on the host.
- Sandbox containers must have no network access, a read-only filesystem except for
  a bounded work volume, and a CPU/memory/time limit.
- Repository paths passed from user input must be validated and canonicalized before
  any filesystem operation. Reject path traversal attempts.
- GitHub tokens are write-scoped only to the target repository. Never persist tokens
  beyond the lifetime of a single run.
- Secrets must never appear in logs, structured events, or error messages.
- Docker image builds must pin base image digests for reproducibility.

---

## Testing

Test categories and pytest markers:

| Marker | Meaning |
|--------|---------|
| `unit` | Pure logic, no I/O, no network |
| `integration` | Real filesystem, no network |
| `sandbox` | Requires Docker daemon |
| `meta_api` | Requires `META_API_KEY` env var |
| `slow` | >5 s, excluded from the default run |

Default `pytest` run: `unit` tests only. CI runs `unit + integration`.

Rules:
- Tests must not call external APIs unless marked `meta_api`.
- Tests must not modify files outside `tmp/` or pytest's `tmp_path` fixture.
- Mock only at provider boundaries, not inside domain logic.
- Every public interface in `engineering_knowledge/` and `cost_optimizer/` must have
  unit tests before any workflow code depends on it.
- Do not fabricate test results. Report skips and failures honestly.

---

## Development Commands

```bash
# Install dependencies
pip install -e ".[dev]"

# Lint (auto-fix)
ruff check --fix agent/ tests/

# Format
ruff format agent/ tests/

# Type check
mypy agent/

# Tests (unit only, default)
pytest

# Tests (unit + integration)
pytest -m "unit or integration"

# All tests (requires Docker + META_API_KEY)
pytest -m "unit or integration or sandbox or meta_api"

# Coverage
pytest --cov=agent --cov-report=term-missing
```

---

## Commits and Pull Requests

- Imperative commit subject, ≤72 chars.
- Never commit: `.env`, real API keys, sandbox working directories, `checkpoints/`,
  `tmp/`, `.cache/`, `worktrees/`, binaries, or `__pycache__/`.
- AI-assisted contributions are welcome. A human must review the diff and own the
  final merge.
- Do not merge a PR that has failing tests or unresolved type errors.

---

## Scope Control

This is a hackathon project. Before adding a service, library, or abstraction, ask:
does this directly support the demo path (issue → fix → verified PR)? If not, defer
it. Three similar lines are better than a premature abstraction.
