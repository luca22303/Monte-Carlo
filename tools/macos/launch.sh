#!/bin/bash
# Start the Monte Carlo dashboard in the background (no terminal window) and open it.
#   launch.sh            update, start if not running, open the browser
#   launch.sh --service  update, then run in the foreground (used by the login item / launchd)
# Log file: <repo>/logs/dashboard.log
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PORT="${MC_PORT:-8501}"
URL="http://localhost:$PORT"
LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/dashboard.log"
export PATH="/opt/homebrew/bin:/usr/local/bin:/Library/Frameworks/Python.framework/Versions/Current/bin:$PATH"

open_browser() {
  if command -v open >/dev/null 2>&1; then open "$URL"; elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL"; fi
}
running() { curl -fs "$URL/_stcore/health" >/dev/null 2>&1; }

if [ "${1:-}" != "--service" ] && running; then
  open_browser
  exit 0
fi

{
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') starting ==="
  cd "$REPO" || exit 1
  # 1. update the code (never asks for a password; on failure the current version is used)
  GIT_TERMINAL_PROMPT=0 git pull --ff-only || echo "update skipped (offline or local changes)"
  # 2. Python environment (created on first run)
  if [ ! -x .venv/bin/python ]; then
    python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' || {
      echo "Python 3.11+ is required"; exit 1; }
    python3 -m venv .venv
  fi
  # 3. install libraries only when requirements.txt changed
  REQ_HASH="$(shasum requirements.txt 2>/dev/null || sha1sum requirements.txt)"
  if [ "$REQ_HASH" != "$(cat .venv/.requirements_hash 2>/dev/null)" ]; then
    .venv/bin/pip install -q -r requirements.txt && echo "$REQ_HASH" > .venv/.requirements_hash
  fi
} >> "$LOG" 2>&1

CMD=(.venv/bin/streamlit run app/streamlit_app.py --server.headless true --server.port "$PORT" --server.address localhost
     --browser.gatherUsageStats false)
cd "$REPO" || exit 1
if [ "${1:-}" = "--service" ]; then
  exec "${CMD[@]}" >> "$LOG" 2>&1
fi

nohup "${CMD[@]}" >> "$LOG" 2>&1 &
for _ in $(seq 1 60); do
  running && break
  sleep 1
done
open_browser
