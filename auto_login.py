#!/usr/bin/env python3
"""Automated login and cookie extractor for muse2api.

Launches visible Chrome to https://muse.ai/ with CDP enabled, monitors for session cookies
(hatch_sess and hatch_native_auth_device), automatically imports them into the gateway pool,
and activates the account without any manual copying.
"""

import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import websockets.exceptions
import websockets.sync.client as ws_client

ADMIN_URL = os.environ.get("MUSE2API_ADMIN_URL", "http://127.0.0.1:18610/admin/accounts")
LOGIN_CDP_PORT = 19222
GATEWAY_CDP_PORT = 19210
PROFILE_DIR = Path(__file__).parent / "data" / "login-profile"

HATCH_SESS = "hatch_sess"
HATCH_NATIVE_AUTH_DEVICE = "hatch_native_auth_device"
DEFAULT_ACCOUNT_LABEL = "primary"
MUSE_SITE_URL = "https://muse.ai/"
WIN32_PLATFORM = "win32"
UTF_8 = "utf-8"


def _load_admin_key() -> str:
    """Retrieve gateway admin authorization key without hardcoding secrets."""
    env_key = os.environ.get("MUSE2API_ADMIN_KEY")
    if env_key:
        return env_key
    env_file = Path(__file__).parent / ".env"
    if env_file.is_file():
        try:
            for line in env_file.read_text(encoding=UTF_8).splitlines():
                if line.startswith("MUSE2API_ADMIN_KEY="):
                    return line.split("=", 1)[1].strip().strip('"\'')
        except OSError:
            pass
    return ""


def get_chrome_path() -> str:
    """Find Google Chrome or Chromium executable across macOS, Windows, and Linux."""
    env_path = os.environ.get("MUSE2API_CHROMIUM_PATH") or os.environ.get("CHROME_PATH")
    if env_path and (Path(env_path).is_file() or shutil.which(env_path)):
        return env_path

    candidates = [
        # macOS
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        # Windows
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%PROGRAMFILES%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%PROGRAMFILES(X86)%\Google\Chrome\Application\chrome.exe"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        "chrome.exe",
        "msedge.exe",
        # Linux
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/snap/bin/chromium",
    ]
    for c in candidates:
        if c and (Path(c).is_file() or shutil.which(c)):
            return c
    return "chrome" if sys.platform == WIN32_PLATFORM else "google-chrome"


CHROME_PATH = get_chrome_path()


def _get_cdp_cookies_sync(port: int) -> dict[str, str]:
    """Check for cookies on a CDP port synchronously using websockets."""
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/json", timeout=2.0)
        targets = r.json()
    except (httpx.HTTPError, OSError, json.JSONDecodeError):
        return {}

    page = next(
        (t for t in targets if t.get("type") == "page" and "muse.ai" in t.get("url", "")),
        next((t for t in targets if t.get("type") == "page"), None),
    )
    if not page or not page.get("webSocketDebuggerUrl"):
        return {}

    ws_url = page["webSocketDebuggerUrl"]

    try:
        with ws_client.connect(ws_url, close_timeout=2) as ws:
            ws.send(json.dumps({
                "id": 1,
                "method": "Network.getCookies",
                "params": {"urls": [MUSE_SITE_URL]},
            }))
            resp = json.loads(ws.recv(timeout=3))
            raw_cookies = resp.get("result", {}).get("cookies", [])
            return {c["name"]: c["value"] for c in raw_cookies if "name" in c and "value" in c}
    except (OSError, json.JSONDecodeError, websockets.exceptions.WebSocketException):
        return {}


def register_account(cookies: dict[str, str], label: str = DEFAULT_ACCOUNT_LABEL) -> bool:
    """Send extracted cookies to the muse2api gateway."""
    admin_key = _load_admin_key()
    headers = {"Authorization": f"Bearer {admin_key}", "Content-Type": "application/json"}
    payload = {"label": label, "cookies": cookies}
    try:
        resp = httpx.post(ADMIN_URL, headers=headers, json=payload, timeout=15.0)
    except httpx.HTTPError as e:
        print(f"❌ Failed to reach gateway at {ADMIN_URL}: {e}")
        return False

    if resp.status_code in (200, 201):
        print(f"✅ Account '{label}' registered successfully in muse2api!")
        return True
    print(f"❌ Gateway registration failed ({resp.status_code}): {resp.text}")
    return False


def _check_existing_sessions(label: str) -> bool:
    """Check if gateway CDP or login profile port already has valid session cookies."""
    endpoints = (
        (GATEWAY_CDP_PORT, "running browser"),
        (LOGIN_CDP_PORT, "existing login window"),
    )
    for port, desc in endpoints:
        existing = _get_cdp_cookies_sync(port)
        if existing.get(HATCH_SESS) and existing.get(HATCH_NATIVE_AUTH_DEVICE):
            print(f"⚡ Found active session cookies already present in {desc}!")
            return register_account(existing, label)
    return False


def _wait_for_login(proc: subprocess.Popen[Any], timeout: int, label: str) -> bool:
    """Wait for user to finish login in the launched Chrome instance."""
    start_time = time.time()
    try:
        while time.time() - start_time < timeout:
            cookies = _get_cdp_cookies_sync(LOGIN_CDP_PORT)
            if cookies.get(HATCH_SESS) and cookies.get(HATCH_NATIVE_AUTH_DEVICE):
                print("\n🎉 Login detected! Extracting session cookies...")
                ok = register_account(cookies, label)
                if ok:
                    print("🔒 Session is now active and protected by automatic keepalive.")
                return ok
            time.sleep(1.5)
        print("\n⏱️ Login timed out. Please try running `python auto_login.py` again.")
        return False
    finally:
        time.sleep(2)
        with contextlib.suppress(OSError):
            proc.terminate()


def run_auto_login(timeout: int = 600, label: str = DEFAULT_ACCOUNT_LABEL) -> bool:
    """Launch visible Chrome and wait for user login."""
    print("==================================================")
    print("         muse2api Automatic Session Login         ")
    print("==================================================")

    if _check_existing_sessions(label):
        return True

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    chrome_cmd = [
        CHROME_PATH,
        f"--remote-debugging-port={LOGIN_CDP_PORT}",
        f"--user-data-dir={PROFILE_DIR}",
        "--no-first-run",
        "--no-default-browser-check",
        MUSE_SITE_URL,
    ]
    print("🌐 Opening Chrome for muse.ai login...")
    print("👉 Please log in to your muse.ai account in the opened browser window.")
    print(f"⏳ Waiting for login completion (timeout {timeout}s)...")

    sub_kwargs: dict[str, Any] = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform != WIN32_PLATFORM:
        sub_kwargs["start_new_session"] = True
    else:
        sub_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    proc = subprocess.Popen(chrome_cmd, **sub_kwargs)
    return _wait_for_login(proc, timeout, label)


if __name__ == "__main__":
    lbl = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ACCOUNT_LABEL
    success = run_auto_login(label=lbl)
    sys.exit(0 if success else 1)
