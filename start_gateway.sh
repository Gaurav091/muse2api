#!/usr/bin/env bash
# Start the muse2api gateway.
#
# Default: FOREGROUND — logs stream live in this terminal and the process is
# tied to this task. Press Ctrl+C (or stop the task) to shut it down.
# On every start, any previous instance is stopped first so only ONE instance
# ever runs.
#
# Background (old behaviour): BACKGROUND=1 bash ./start_gateway.sh
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Always clean up any existing instance first (guarantees a single instance).
./stop_gateway.sh 2>/dev/null || true

# Ensure dirs exist and clear stale browser locks.
mkdir -p data/logs data/browser-profile
rm -f data/browser-profile/Singleton* 2>/dev/null || true

echo "Starting muse2api gateway on http://127.0.0.1:18610..."
echo "Dashboard: http://127.0.0.1:18610/dashboard"

# --- Background mode (detached, logs to file only) --------------------------
if [ "${BACKGROUND:-0}" = "1" ]; then
    nohup .venv/bin/python -m muse2api > data/logs/gateway.log 2>&1 &
    PID=$!
    echo "$PID" > data/gateway.pid
    echo "Gateway running in background with PID: $PID"
    echo "Logs: $DIR/data/logs/gateway.log"
    exit 0
fi

# --- Foreground mode (default): stream logs live ---------------------------
echo "Logs streaming below (Ctrl+C to stop)."
echo "------------------------------------------------------------"

# tee keeps a copy in data/logs/gateway.log while also printing to the terminal.
# Process substitution lets us capture the real Python PID for the pidfile.
.venv/bin/python -m muse2api > >(tee data/logs/gateway.log) 2>&1 &
APP_PID=$!
echo "$APP_PID" > data/gateway.pid

# Ensure the gateway is stopped if this script is interrupted.
trap 'kill "$APP_PID" 2>/dev/null || true' INT TERM

# Block in the foreground so the terminal stays attached to this instance.
wait "$APP_PID"