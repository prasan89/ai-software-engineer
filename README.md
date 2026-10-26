# AI Software Engineer Agent

An AI agent that accepts a GitHub issue, analyzes a Java/Spring Boot codebase,
implements a fix, generates regression tests, and verifies the solution.

Built for the **Meta Global AI Developer Hackathon**.

Primary differentiator: **Engineering Knowledge & Cost Optimizer** — exact
duplicate detection, verified fix reuse, repository-aware cache invalidation,
resumable checkpoints, model routing, and measurable token savings.

---

## Prerequisites

- Python 3.11+
- Docker (for sandbox execution of untrusted repository code)
- A Meta developer account with a Model API key ([dev.meta.ai](https://dev.meta.ai/))
- A GitHub personal access token (repo:read scope)

---

## Setup

```bash
# 1. Clone
git clone <repo-url>
cd ai-software-engineer-agent

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install with dev dependencies
pip install -e ".[dev]"

# 4. Configure environment
cp .env.example .env
# Edit .env — add META_API_KEY and GITHUB_TOKEN (never commit .env)
```

---

## Development Commands

```bash
# Run unit tests (default — no network, no Docker required)
pytest

# Run with coverage
pytest --cov=agent --cov-report=term-missing

# Run unit + integration tests
pytest -m "unit or integration"

# Lint (auto-fix)
ruff check --fix agent/ tests/

# Format
ruff format agent/ tests/

# Type check
mypy agent/

# Start the API server (development)
uvicorn agent.api.app:app --reload --host 0.0.0.0 --port 8000
```

---

## Architecture

```
agent/
  api/                   FastAPI app, routes, settings
  providers/             Provider-neutral LLM interface + Meta adapter
  engineering_knowledge/ Fingerprinting, fix store, cache, checkpoints
  cost_optimizer/        Usage tracking, model routing, metrics
  workflow/              Typed states, run events
  repo/                  Repository cloning, file indexing, tree hashing
  patch/                 Patch generation and application
  testgen/               Regression test generation
  sandbox/               Docker-isolated execution
  verification/          Test result parsing and outcome reporting
tests/
  unit/                  Pure logic tests (no I/O, no network)
  integration/           Filesystem tests (no network)
```

See [AGENTS.md](AGENTS.md) for architecture rules, security requirements,
testing conventions, and development commands.

---

## Provider Notes

**Primary provider:** Meta Model API (`https://api.meta.ai/v1`)

Verified model identifiers and pricing (2026-10-26):

| Model | Input ($/1M) | Output ($/1M) | Use |
|---|---|---|---|
| `muse-wonder-1.0` | $4.00 | $20.00 | Complex multi-file changes |
| `muse-spark-1.3` | $1.25 | $4.25 | Typical bug fixes |
| `muse-spark-1.3-contributor` | $0.10 | $0.20 | Simple lookups, test gen |

The agent routes tasks to the cheapest sufficient model automatically.
Fallback to other providers is opt-in and requires explicit budget configuration.

> **Unresolved:** The exact request field name for enabling prompt caching
> is not documented in the public Meta API docs as of 2026-10-26. The response
> field `cached_tokens` is confirmed. This will be verified before the
> cache-prefix optimization is implemented.

---

## Status

Scaffold complete. The following modules are stubs pending implementation:
- `agent/repo/` — repository cloning and file indexing
- `agent/patch/` — patch generation and application
- `agent/testgen/` — regression test generation
- `agent/sandbox/` — Docker sandbox execution
- `agent/verification/` — test result parsing
- `agent/api/routes.py` — POST /runs endpoint and SSE event stream
- `agent/workflow/` — orchestrator wiring all stages together
