# AI Software Engineer Agent

An AI agent that accepts a GitHub issue, analyzes a Java/Spring Boot codebase,
implements a fix, generates regression tests, and verifies the solution inside a
secure Docker sandbox.

Built for the **Meta Global AI Developer Hackathon**.

Primary differentiator: **Engineering Knowledge & Cost Optimizer** — exact
duplicate detection, verified fix reuse, repository-aware cache invalidation,
resumable checkpoints, model routing, and measurable token savings.

---

## What is implemented

| Component | Status |
|---|---|
| `agent/providers/meta.py` | Complete — Meta Model API adapter with retry, budget, cached-token accounting |
| `agent/engineering_knowledge/` | Complete — fingerprinting, fix store, cache invalidation, checkpoints |
| `agent/cost_optimizer/` | Complete — usage tracking, model routing, metrics |
| `agent/workflow/orchestrator.py` | Complete — full issue → fix → verify pipeline with retry |
| `agent/sandbox/docker_executor.py` | Complete — Docker-isolated Maven execution |
| `agent/sandbox/local_executor.py` | Complete — host executor, opt-in trusted fixtures only |
| `agent/verification/__init__.py` | Complete — maps all sandbox outcomes to VerificationResult |
| `agent/repo/__init__.py` | Complete — file indexing, tree hashing |
| `agent/patch/__init__.py` | Complete — unified-diff application via `git apply` |
| `agent/api/routes.py` | Partial — health endpoint only; POST /runs and SSE stream not yet implemented |
| `agent/testgen/__init__.py` | Stub — test generation is handled inline in the orchestrator |

**Live Meta inference:** Not yet verified in this environment (META_API_KEY not set).
Four live API tests exist in `tests/unit/providers/test_meta_http.py` (marker `meta_api`);
they are skipped when the key is absent with an explicit skip message.

---

## Prerequisites

- Python 3.11+
- Docker (required for untrusted-repository sandbox execution)
- A Meta developer account with a Model API key ([dev.meta.ai](https://dev.meta.ai/))
- A GitHub personal access token (repo:read scope)
- Maven 3.x + JDK 21 on the host (for integration tests that use LocalSandbox)

---

## Setup

```bash
# 1. Clone
git clone https://github.com/prasan89/ai-software-engineer
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
# Run unit tests (default — no network, no Docker, no Maven required)
pytest

# Run with coverage
pytest --cov=agent --cov-report=term-missing

# Run unit + integration tests
# Requires: Maven 3.x + JDK 21 on host, fixtures/spring-calc present
pytest -m "unit or integration"

# Run Docker sandbox integration tests
# Requires: Docker daemon running, maven:3.9.9-eclipse-temurin-21-alpine image
pytest -m sandbox -v

# Run live Meta API tests (requires META_API_KEY in environment)
META_API_KEY=<key> pytest -m meta_api -v

# Lint (auto-fix)
ruff check --fix agent/ tests/

# Format
ruff format agent/ tests/

# Type check
mypy agent/
```

---

## Docker Configuration

The sandbox uses:
```
maven:3.9.9-eclipse-temurin-21-alpine
```

Security parameters (all enforced by default):

| Flag | Value | Purpose |
|---|---|---|
| `--network` | `none` | Blocks all outbound network from Maven plugins |
| `--read-only` | — | Prevents writes to the container FS |
| `--cap-drop` | `ALL` | Drops all Linux capabilities |
| `--security-opt` | `no-new-privileges` | Prevents privilege escalation |
| `--user` | `1000:1000` | Runs as non-root |
| `--memory` | `1g` | Hard memory cap |
| `--memory-swap` | `1g` | Disables swap |
| `--cpus` | `1.0` | One CPU core maximum |
| `--pids-limit` | `128` | Fork-bomb protection |
| `--rm` | — | Container auto-removed on exit |
| No `docker.sock` | — | Container cannot access Docker daemon |

**Maven offline mode:** The sandbox requires a pre-seeded Maven local repository
for offline builds (`--network=none`). Pass `maven_local_repo` to
`DockerSandbox.run_maven_tests()`, or set `MAVEN_LOCAL_REPO` env var.

Pre-seed the local repository:
```bash
mvn -f fixtures/spring-calc/pom.xml dependency:go-offline
```

### Residual risks (documented, not claimed solved)

1. tmpfs mounts are uncapped on cgroup v1 hosts — a build could exhaust host RAM.
2. The Maven image is not pinned to a digest in default config — pin
   `SandboxConfig.image` to `@sha256:...` for production use.
3. Maven plugins run as uid 1000 inside the container; a plugin exploit could
   still read files accessible to that uid within the container.
4. Docker is invoked as a subprocess of the agent — if the agent process is
   compromised, so is Docker access.

---

## Architecture

```
agent/
  api/                   FastAPI app, routes, settings
  providers/             Provider-neutral LLM interface + Meta adapter
  engineering_knowledge/ Fingerprinting, fix store, cache, checkpoints
  cost_optimizer/        Usage tracking, model routing, metrics
  workflow/              Orchestrator, typed states, run events
  repo/                  Repository indexing, tree hashing
  patch/                 Patch generation and application
  testgen/               Stub (test generation handled in orchestrator)
  sandbox/               Docker-isolated execution (DockerSandbox)
  verification/          Test result interpretation
tests/
  unit/                  Pure logic tests (no I/O, no network, no Docker)
  integration/           Filesystem tests (Maven on host, trusted fixtures)
  sandbox/               Docker integration tests (requires Docker daemon)
```

### Sandbox routing

- **Untrusted repositories** (any repo from GitHub or user input): `DockerSandbox`.
  If Docker is unavailable, the run fails with `SandboxResult.sandbox_error=True`.
  There is no fallback to host execution.
- **Trusted fixtures** (version-controlled test data in this repo): integration
  tests inject `LocalSandbox(trusted=True)` via the `sandbox_runner` parameter.
  `LocalSandbox` requires the caller to set `trusted=True` explicitly.

### Workflow contract

`VerificationResult` distinguishes five outcomes:

| Outcome | Field | `passed` |
|---|---|---|
| All tests pass | — | `True` |
| Tests failed | `tests_failed > 0` | `False` |
| Execution timed out | `timed_out=True` | `False` |
| OOM killed | `oom_killed=True` | `False` |
| Sandbox unavailable | `sandbox_unavailable=True` | `False` |

A sandbox or verification failure **never** propagates as a successful fix.

---

## Test markers

| Marker | What it needs | Command |
|---|---|---|
| `unit` | Nothing | `pytest` |
| `integration` | Maven + JDK 21 on host | `pytest -m integration` |
| `sandbox` | Docker daemon | `pytest -m sandbox` |
| `meta_api` | `META_API_KEY` env var | `META_API_KEY=<key> pytest -m meta_api` |
| `slow` | Time (>30 s) | `pytest -m slow` |

---

## Provider notes

**Primary provider:** Meta Model API (`https://api.meta.ai/v1`)

Verified model identifiers and pricing (2026-10-26):

| Model | Input ($/1M) | Output ($/1M) | Use |
|---|---|---|---|
| `muse-wonder-1.0` | $4.00 | $20.00 | Complex multi-file changes |
| `muse-spark-1.3` | $1.25 | $4.25 | Typical bug fixes |
| `muse-spark-1.3-contributor` | $0.10 | $0.20 | Simple lookups, test gen |

> **Unresolved:** The exact request field name for enabling prompt caching
> is not documented in the public Meta API docs as of 2026-10-26. The response
> field `cached_tokens` is confirmed. Live inference has not been verified in
> this environment.

---

See [AGENTS.md](AGENTS.md) for architecture rules, security requirements,
testing conventions, and development commands.
