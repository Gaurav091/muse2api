#!/usr/bin/env python3
"""Automated login and cookie extractor for muse2api.

Launches visible Chrome to https://muse.ai/ with CDP enabled, monitors for session cookies
(hatch_sess and hatch_native_auth_device), automatically imports them into the gateway pool,
and activates the account without any manual copying.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import httpx

ADMIN_URL = os.environ.get("MUSE2API_ADMIN_URL", "http://127.0.0.1:18610/admin/accounts")
ADMIN_KEY = os.environ.get("MUSE2API_ADMIN_KEY", "sk-muse2api-admin-secret")
LOGIN_CDP_PORT = 19222
CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PROFILE_DIR = Path(__file__).parent / "data" / "login-profile"


def _get_cdp_cookies_sync(port: int) -> dict[str, str]:
    """Check for cookies on a CDP port synchronously using websockets."""
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/json")
        with urllib.request.urlopen(req, timeout=2) as r:
            targets = json.loads(r.read().decode("utf-8"))
    except Exception:
        return {}

    page = next(
        (t for t in targets if t.get("type") == "page" and "muse.ai" in t.get("url", "")),
        next((t for t in targets if t.get("type") == "page"), None),
    )
    if not page or not page.get("webSocketDebuggerUrl"):
        return {}

    ws_url = page["webSocketDebuggerUrl"]

    import websockets.sync.client as ws_client

    try:
        with ws_client.connect(ws_url, close_timeout=2) as ws:
            ws.send(json.dumps({
                "id": 1,
                "method": "Network.getCookies",
                "params": {"urls": ["https://muse.ai"]},
            }))
            resp = json.loads(ws.recv(timeout=3))
            raw_cookies = resp.get("result", {}).get("cookies", [])
            return {c["name"]: c["value"] for c in raw_cookies if "name" in c and "value" in c}
    except Exception:
        return {}


def register_account(cookies: dict[str, str], label: str = "primary") -> bool:
    """Send extracted cookies to the muse2api gateway."""
    headers = {"Authorization": f"Bearer {ADMIN_KEY}", "Content-Type": "application/json"}
    payload = {"label": label, "cookies": cookies}
    try:
        resp = httpx.post(ADMIN_URL, headers=headers, json=payload, timeout=15.0)
        if resp.status_code in (200, 201):
            print(f"✅ Account '{label}' registered successfully in muse2api!")
            return True
        print(f"❌ Gateway registration failed ({resp.status_code}): {resp.text}")
        return False
    except Exception as e:
        print(f"❌ Failed to reach gateway at {ADMIN_URL}: {e}")
        return False


def run_auto_login(timeout: int = 180, label: str = "primary") -> bool:
    """Launch visible Chrome and wait for user login."""
    print("==================================================")
    print("         muse2api Automatic Session Login         ")
    print("==================================================")

    # 1. First check if port 19210 already has valid cookies
    existing = _get_cdp_cookies_sync(19210)
    if existing.get("hatch_sess") and existing.get("hatch_native_auth_device"):
        print("⚡ Found active session cookies already present in running browser!")
        return register_account(existing, label)

    # 2. Check if login profile already has cookies on port 19222
    existing_19222 = _get_cdp_cookies_sync(LOGIN_CDP_PORT)
    if existing_19222.get("hatch_sess") and existing_19222.get("hatch_native_auth_device"):
        print("⚡ Found active session cookies in existing login window!")
        return register_account(existing_19222, label)

    # 3. Launch Chrome with login profile and remote debugging
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    chrome_cmd = [
        CHROME_PATH,
        f"--remote-debugging-port={LOGIN_CDP_PORT}",
        f"--user-data-dir={PROFILE_DIR}",
        "--no-first-run",
        "--no-default-browser-check",
        "https://muse.ai/",
    ]
    print(f"🌐 Opening Chrome for muse.ai login...")
    print(f"👉 Please log in to your muse.ai account in the opened browser window.")
    print(f"⏳ Waiting for login completion (timeout {timeout}s)...")

    proc = subprocess.Popen(
        chrome_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    start_time = time.time()
    try:
        while time.time() - start_time < timeout:
            cookies = _get_cdp_cookies_sync(LOGIN_CDP_PORT)
            if cookies.get("hatch_sess") and cookies.get("hatch_native_auth_device"):
                print("\n🎉 Login detected! Extracting session cookies...")
                ok = register_account(cookies, label)
                if ok:
                    print("🔒 Session is now active and protected by automatic keepalive.")
                return ok
            time.sleep(1.5)
        print("\n⏱️ Login timed out. Please try running `python auto_login.py` again.")
        return False
    finally:
        # Give user 2 seconds to view success then clean up process
        time.sleep(2)
        try:
            proc.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    lbl = sys.argv[1] if len(sys.argv) > 1 else "primary"
    success = run_auto_login(label=lbl)
    sys.exit(0 if success else 1)
