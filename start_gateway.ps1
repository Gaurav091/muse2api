<#
.SYNOPSIS
    Start the muse2api gateway with live log streaming and strict single-process cleanup (Windows native, PowerShell).
#>
param(
    [switch]$Background
)

$ErrorActionPreference = "Stop"
$Port = 18610

# Resolve repo root (this script lives at the repo root).
Set-Location $PSScriptRoot

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "          Starting muse2api Gateway (Single Instance)       " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Strictly terminate ANY previous instance, orphan helper, or browser on 19210
& "$PSScriptRoot\stop_gateway.ps1" | Out-Null

# 2. Ensure dirs exist and clear stale browser locks.
New-Item -ItemType Directory -Force -Path "data\logs", "data\browser-profile" | Out-Null
Remove-Item "data\browser-profile\Singleton*" -Force -ErrorAction SilentlyContinue
"" | Out-File -FilePath "data\logs\gateway.log" -Encoding utf8

# 3. Locate the Python interpreter (prefer repo venv).
$Python = "python"
if (Test-Path ".venv\Scripts\python.exe") {
    $Python = ".venv\Scripts\python.exe"
}

# 4. Start the gateway
$proc = Start-Process -FilePath $Python `
    -ArgumentList "-m", "muse2api" `
    -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput "data\logs\gateway.log" `
    -RedirectStandardError "data\logs\gateway.err.log"
$proc.Id | Out-File -FilePath "data\gateway.pid" -Encoding ascii

# 5. Wait up to 10 seconds for the gateway HTTP server to bind
$bound = $false
for ($i = 0; $i -lt 20; $i++) {
    try {
        $resp = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/v1/models" -UseBasicParsing -TimeoutSec 1 -ErrorAction SilentlyContinue
        if ($resp.StatusCode -eq 200) {
            $bound = $true
            break
        }
    } catch {
        # Keep waiting
    }
    Start-Sleep -Milliseconds 500
}

if ($bound) {
    Write-Host "🟢 Gateway running with PID: $($proc.Id)" -ForegroundColor Green
    Write-Host "🌐 API Base:   http://127.0.0.1:$Port/v1"
    Write-Host "📊 Dashboard:  http://127.0.0.1:$Port/dashboard"
    Write-Host "Logs: $PSScriptRoot\data\logs\gateway.log"

    # Auto-verify active session and auto-recover if needed
    & $Python -c @"
import httpx, subprocess, sys

ADMIN_KEY = 'sk-muse2api-admin-secret'
try:
    resp = httpx.get(
        'http://127.0.0.1:$Port/admin/accounts',
        headers={'Authorization': f'Bearer {ADMIN_KEY}'},
        timeout=5.0
    )
    accounts = resp.json().get('data', [])
    active = [a for a in accounts if a.get('status') == 'active' and a.get('enabled')]
    if active:
        lbl = active[0].get('label', 'primary')
        print(f'✅ Session auto-grabbed: active account \"{lbl}\" loaded and ready for Hermes.')
    else:
        print('⚠️ No active session found. Auto-launching login recovery...')
        subprocess.Popen([sys.executable, 'auto_login.py'])
except Exception as e:
    print(f'ℹ️ Gateway check note: {e}')
"@
} else {
    Write-Host "❌ Gateway failed to bind within 10s. Check logs in data\logs\" -ForegroundColor Red
    & "$PSScriptRoot\stop_gateway.ps1" | Out-Null
    exit 1
}

if ($Background) {
    Write-Host "Gateway running in background (-Background). Logs: $PSScriptRoot\data\logs\gateway.log"
    exit 0
}

Write-Host "------------------------------------------------------------"
Write-Host "📡 Live logs streaming below (Press Ctrl+C to stop gateway):" -ForegroundColor Yellow
Write-Host "------------------------------------------------------------"

try {
    Get-Content -Path "data\logs\gateway.log" -Wait -Tail 30
} finally {
    Write-Host "`n[muse2api] Shutting down gateway cleanly..." -ForegroundColor Cyan
    & "$PSScriptRoot\stop_gateway.ps1" | Out-Null
}
