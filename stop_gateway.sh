#!/usr/bin/env bash
# Stop the muse2api gateway and all associated Chromium browser processes cleanly.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# 1. Stop by recorded PID file if present
if [ -f data/gateway.pid ]; then
  PID=$(cat data/gateway.pid)
  if kill -0 "$PID" 2>/dev/null; then
    echo "Stopping muse2api gateway (PID $PID)..."
    kill -15 "$PID" 2>/dev/null || true
    sleep 0.5
    kill -9 "$PID" 2>/dev/null || true
  fi
  rm -f data/gateway.pid
fi

# 2. Kill any orphan muse2api Python processes
pkill -9 -f "python.*-m muse2api" 2>/dev/null || true
pkill -9 -f "auto_login.py" 2>/dev/null || true

# 3. Kill anything bound to gateway port 18610
lsof -ti :18610 2>/dev/null | xargs kill -9 2>/dev/null || true

# 4. Kill anything bound to browser CDP port 19210 and lingering browser renderers
lsof -ti :19210 2>/dev/null | xargs kill -9 2>/dev/null || true
pkill -9 -f "Google Chrome.*browser-profile" 2>/dev/null || true
pkill -9 -f "Google Chrome Helper.*browser-profile" 2>/dev/null || true

# 5. Clear stale browser singleton locks
rm -f data/browser-profile/Singleton* 2>/dev/null || true

echo "Gateway stopped."
