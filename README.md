# AI Software Engineer Agent

An AI agent that accepts a Java bug description, analyzes the codebase,
implements a fix with Meta muse-spark-1.3, generates a regression test,
and verifies the solution inside a secure Docker sandbox.

Built for the **Meta Global AI Developer Hackathon**.

**Core differentiator:** Engineering Knowledge & Cost Optimizer — exact duplicate
detection, verified fix reuse (zero additional LLM calls on repeat issues),
repository-aware cache invalidation, resumable checkpoints, model routing,
and measurable token savings.

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/prasan89/ai-software-engineer
cd ai-software-engineer-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Set your Meta API key
export META_API_KEY=your-key-here  # or add to ~/.zshrc

# 3. Pre-seed the Maven dependency cache (one time, ~60 s)
mvn -f fixtures/spring-calc/pom.xml dependency:go-offline

# 4. Launch the demo
./demo.sh            # starts web server at http://localhost:8000
# or
./demo.sh --e2e      # CLI three-scenario demo (no browser needed)
```

---

## Demo Walkthrough (2–3 minutes)

Open `http://localhost:8000`. The Spring Boot NPE fixture is pre-loaded.

| Step | Action | What to watch |
|---|---|---|
| 1 | Click **Start Run** | 7 pipeline stages execute in real time |
| 2 | Wait ~35 s | Diff appears: `if (divisor == null) throw IAE` |
| 3 | See **Test Results** | 6 Maven tests pass inside Docker |
| 4 | Check **LLM Usage** | 2 calls, ~$0.021 total cost |
| 5 | Click **Repeat Issue** | Fix reused from store — 0 LLM calls, $0.000 |
| 6 | Check **Engineering Memory** | Cache MISS → HIT transition shown |

**First 30 seconds:** The "Repeat Issue" result is the core differentiator —
identical issue, zero provider calls, identical verified outcome.

### CLI demo (all three scenarios)

```
./demo.sh --e2e
```

Expected output:
```
SCENARIO A: success=True  llm_calls=2  tests_run=6  cost=$0.021
SCENARIO B: success=True  llm_calls=0  fix_reused=YES  cost=$0.000
SCENARIO C: success=True  llm_calls=2  stale cache rejected
```

---

## What's Implemented

| Component | Status |
|---|---|
| `agent/providers/meta.py` | Complete — Meta muse-spark-1.3 adapter, retry, budget, cost accounting |
| `agent/engineering_knowledge/` | Complete — fingerprinting, fix store (with test preservation), cache invalidation, checkpoints |
| `agent/cost_optimizer/` | Complete — usage tracking, model routing, metrics |
| `agent/workflow/orchestrator.py` | Complete — 7-stage pipeline, retry, event callbacks |
| `agent/sandbox/docker_executor.py` | Complete — Docker-isolated Maven; `--network=none`, `--read-only`, `--cap-drop ALL` |
| `agent/patch/__init__.py` | Complete — LLM-tolerant diff application (`patch -p1 -l -F 3`, hunk repair) |
| `agent/api/routes.py` | Complete — `POST /runs`, `GET /runs/{id}/events` (SSE), `GET /runs/{id}`, `/fixtures` |
| `agent/api/static/index.html` | Complete — single-page UI, no build step required |
| `scripts/e2e_demo.py` | Complete — three-scenario CLI demo |
| `demo.sh` | Complete — single-command server + e2e launcher |

**Verified results (2026-10-26):**
- Scenario A: `success=True`, 6/6 Docker tests pass, ~$0.021 (2 LLM calls, 37s)
- Scenario B: fix reused, 0 LLM calls, $0.000 — **100% cost saving on repeat**
- Scenario C: stale cache correctly invalidated, fresh analysis runs

---

## Prerequisites

- Python 3.11+
- Docker (required for sandbox; `docker pull maven:3.9.9-eclipse-temurin-21-alpine`)
- Meta developer account with Model API key ([dev.meta.ai](https://dev.meta.ai/))
- Maven 3.x + JDK 21 on the host (for `mvn dependency:go-offline` only)

---

## Development Commands

```bash
# Unit tests (no network, no Docker)
pytest

# Unit + integration tests
pytest -m "unit or integration"

# Sandbox tests (requires Docker)
pytest -m sandbox -v

# Live Meta API tests (requires META_API_KEY)
pytest -m meta_api -v

# Lint (auto-fix)
ruff check --fix agent/ tests/

# Format
ruff format agent/ tests/

# Type check
mypy agent/

# Full quality gate
pytest -m "unit and not meta_api" && ruff check agent/ tests/ && mypy agent/
```

---

## Architecture

```
agent/
  api/                   FastAPI app (routes, run manager, static UI)
  providers/             Meta muse-spark-1.3 adapter + provider interface
  engineering_knowledge/ Fingerprinting, fix store, cache, checkpoints
  cost_optimizer/        Usage tracking, model routing, metrics
  workflow/              Orchestrator, typed states, run events
  repo/                  Repository indexing, tree hashing
  patch/                 LLM-tolerant unified diff application
  sandbox/               Docker-isolated Maven execution
  verification/          Test result interpretation
fixtures/
  spring-calc/           Spring Boot NPE fixture (5 tests + buggy divide())
scripts/
  e2e_demo.py            Three-scenario CLI demo
tests/
  unit/                  Pure logic (no I/O, no network, no Docker)
  integration/           Real filesystem, local Maven
  sandbox/               Docker integration
```

### Pipeline stages (WorkflowStage)

```
intake → repo_analysis → root_cause → patch →
test_generation → sandbox_execution → verification → done
```

Events are emitted at each stage boundary and streamed to clients via
`GET /runs/{id}/events` (Server-Sent Events).

### Engineering Knowledge flow

```
Issue arrives
  → fingerprint_from_raw() produces issue fingerprint
  → fix_store.check_compatibility(fp, repo, tree_hash)
      COMPATIBLE  → apply saved patch + re-inject test → verify → done (0 LLM calls)
      MISMATCH    → fresh LLM analysis
                  → diagnosis LLM call (muse-spark-1.3, 4096 max_tokens)
                  → apply_patch() with hunk repair + patch -p1 -l -F 3
                  → test_gen LLM call (2048 max_tokens)
                  → inject regression test into test class
                  → docker_executor.run_maven_tests()
                  → fix_store.save(patch + test_file_contents)
```

### Sandbox security (all enforced by default)

| Flag | Value |
|---|---|
| `--network` | `none` |
| `--read-only` | enabled |
| `--cap-drop` | `ALL` |
| `--security-opt` | `no-new-privileges` |
| `--user` | `1000:1000` |
| `--memory` | `1g` |
| `--cpus` | `1.0` |
| `--pids-limit` | `128` |
| No `docker.sock` | enforced |

Docker is unavailable → `SandboxResult.sandbox_error=True`, `passed=False`.
No fallback to host execution.

### Documented residual risks

1. Maven image not pinned to a digest in default config — pin `SandboxConfig.image`
   to `@sha256:...` for production use.
2. Maven plugins run as uid 1000 inside the container.
3. Docker is invoked as a subprocess; agent compromise = Docker access.

---

## Test markers

| Marker | Needs | Command |
|---|---|---|
| `unit` | Nothing | `pytest` |
| `integration` | Maven + JDK 21 | `pytest -m integration` |
| `sandbox` | Docker | `pytest -m sandbox` |
| `meta_api` | `META_API_KEY` | `pytest -m meta_api` |
| `slow` | Time (>30 s) | `pytest -m slow` |

---

## Provider notes

**Primary:** Meta Model API (`https://api.meta.ai/v1`)

| Model | Cached $/1M | Input $/1M | Output $/1M | Use |
|---|---|---|---|---|
| `muse-spark-1.3` | $0.15 | $1.25 | $4.25 | Default (diagnosis + test gen) |
| `muse-spark-1.3-contributor` | $0.002 | $0.10 | $0.20 | Hackathon contributor tier |
| `muse-wonder-1.0` | $0.20 | $4.00 | $20.00 | Complex repos (future) |

`muse-spark-1.3` uses ~148–2000 internal reasoning tokens per request.
`max_tokens` must be ≥ 512 for any call; the orchestrator uses 4096 for
diagnosis and 2048 for test generation.

When `content=None` (finish_reason=length), `MetaProvider` raises
`ProviderError` with a clear message rather than crashing Pydantic validation.
