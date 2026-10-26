#!/usr/bin/env bash
# demo.sh — single-command demo launcher for the AI Software Engineer Agent
#
# Usage:
#   ./demo.sh            # start server + open browser
#   ./demo.sh --e2e      # run the 3-scenario CLI e2e demo (no browser)
#   ./demo.sh --help     # show this help
#
# Requirements:
#   - Python virtual environment at .venv/  (pip install -e ".[dev]")
#   - META_API_KEY in environment or ~/.zshrc
#   - Docker daemon running (for sandbox execution)
#   - ~/.m2/repository pre-seeded:
#       mvn -f fixtures/spring-calc/pom.xml dependency:go-offline

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$REPO_DIR/.venv"
PORT="${PORT:-8000}"

# ── colours ────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'
BOLD='\033[1m'; RESET='\033[0m'

ok()   { echo -e "${GREEN}✓${RESET} $*"; }
warn() { echo -e "${YELLOW}⚠${RESET} $*"; }
err()  { echo -e "${RED}✗${RESET} $*" >&2; }
info() { echo -e "${CYAN}▸${RESET} $*"; }

# ── help ───────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
demo.sh — AI Software Engineer Agent

  ./demo.sh            Start the web server and open the browser demo
  ./demo.sh --e2e      Run the 3-scenario CLI e2e demo (A/B/C)
  ./demo.sh --help     Show this message

Environment:
  META_API_KEY   Meta developer API key (required)
  PORT           HTTP port (default 8000)
  MAVEN_LOCAL_REPO  Path to pre-seeded Maven repo (default ~/.m2/repository)
EOF
  exit 0
fi

# ── preflight checks ───────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}AI Software Engineer Agent — Demo Launcher${RESET}"
echo "────────────────────────────────────────────"

# Source zshrc if available (dev convenience)
[[ -f ~/.zshrc ]] && source ~/.zshrc 2>/dev/null || true

if [[ -z "${META_API_KEY:-}" ]]; then
  err "META_API_KEY is not set."
  echo "  Export it in your shell or add it to ~/.zshrc:"
  echo "  export META_API_KEY=your-key-here"
  exit 1
fi
ok "META_API_KEY found"

if [[ ! -d "$VENV" ]]; then
  err "Virtual environment not found at $VENV"
  echo "  Run: python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'"
  exit 1
fi
ok "Python venv at $VENV"

if ! docker info > /dev/null 2>&1; then
  warn "Docker is not running — sandbox tests will show 'Docker unavailable'"
else
  ok "Docker available"
fi

M2_REPO="${MAVEN_LOCAL_REPO:-$HOME/.m2/repository}"
if [[ ! -d "$M2_REPO" ]]; then
  warn "Maven local repo not found at $M2_REPO"
  echo "  Pre-seed with: mvn -f fixtures/spring-calc/pom.xml dependency:go-offline"
else
  ok "Maven local repo at $M2_REPO"
fi

echo ""

# ── e2e CLI mode ───────────────────────────────────────────────────────────
if [[ "${1:-}" == "--e2e" ]]; then
  info "Running 3-scenario CLI e2e demo…"
  echo ""
  cd "$REPO_DIR"
  exec "$VENV/bin/python3" scripts/e2e_demo.py
fi

# ── web server mode ────────────────────────────────────────────────────────
info "Starting web server on port $PORT…"
echo ""
echo -e "  ${BOLD}Open your browser at:${RESET}  ${CYAN}http://localhost:$PORT${RESET}"
echo ""
echo -e "  ${BOLD}Demo walkthrough (2-3 min):${RESET}"
echo "  1. The fixture is pre-loaded: Spring Boot NPE in divide()"
echo "  2. Click 'Start Run'   → watch 7 pipeline stages execute"
echo "  3. See the diff, 6 Docker test results, LLM cost (~\$0.014)"
echo "  4. Click 'Repeat Issue' → fix reused, 0 LLM calls, \$0.000"
echo "  5. The Engineering Memory panel shows cache MISS → HIT"
echo ""
echo -e "  ${YELLOW}Ctrl-C to stop the server${RESET}"
echo ""

cd "$REPO_DIR"
exec "$VENV/bin/uvicorn" agent.api.app:app \
  --host 0.0.0.0 \
  --port "$PORT" \
  --log-level info
