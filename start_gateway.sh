#!/usr/bin/env bash
# Start the muse2api gateway with direct foreground streaming and strict cleanup.
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# 1. Stop any existing instance first (guarantees a single instance).
./stop_gateway.sh 2>/dev/null || true

# 2. Ensure directories exist and clear stale browser locks.
mkdir -p data/logs data/browser-profile
rm -f data/browser-profile/Singleton* 2>/dev/null || true

echo "Starting muse2api gateway on http://127.0.0.1:18610..."
echo "Dashboard: http://127.0.0.1:18610/dashboard"

# --- Background mode (detached) --------------------------------------------
if [ "${BACKGROUND:-0}" = "1" ]; then
    nohup .venv/bin/python -m muse2api >/dev/null 2>&1 &
    PID=$!
    echo "$PID" > data/gateway.pid
    echo "Gateway running in background with PID: $PID"
    echo "Logs: $DIR/data/logs/gateway.log"
    exit 0
fi

# --- Foreground mode (default) ---------------------------------------------
echo "Logs streaming below (Ctrl+C to stop)."
echo "------------------------------------------------------------"

# Record PID so stop_gateway.sh can cleanly terminate it if invoked externally.
echo "$$" > data/gateway.pid

cleanup() {
    rm -f data/gateway.pid 2>/dev/null || true
    ./stop_gateway.sh >/dev/null 2>&1 || true
    exit 0
}
trap cleanup INT TERM

# Exec directly replaces the bash shell with Python.
# Output streams natively to the terminal, and setup_logging mirrors to data/logs/gateway.log.
exec .venv/bin/python -m muse2api