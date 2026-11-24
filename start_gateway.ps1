<#
.SYNOPSIS
    Start the muse2api gateway (Windows native, PowerShell).

.DESCRIPTION
    Default: FOREGROUND — logs stream live in this console and the process is
    tied to this task. Press Ctrl+C (or stop the task) to shut it down.
    On every start, any previous instance is stopped first so only ONE runs.

    Background (detached, logs to file only):
        powershell -File start_gateway.ps1 -Background
#>
param(
    [switch]$Background
)

$ErrorActionPreference = "Stop"
$Port = 18610

# Resolve repo root (this script lives at the repo root).
Set-Location $PSScriptRoot

Write-Host "[muse2api] Stopping any existing instance..." -ForegroundColor Cyan
& "$PSScriptRoot\stop_gateway.ps1" | Out-Null

# Ensure dirs exist and clear stale browser locks.
New-Item -ItemType Directory -Force -Path "data\logs", "data\browser-profile" | Out-Null
Remove-Item "data\browser-profile\Singleton*" -Force -ErrorAction SilentlyContinue

# Locate the Python interpreter (prefer repo venv).
$Python = "python"
if (Test-Path ".venv\Scripts\python.exe") {
    $Python = ".venv\Scripts\python.exe"
}

Write-Host "[muse2api] Starting gateway on http://127.0.0.1:$Port..." -ForegroundColor Green
Write-Host "[muse2api] Dashboard: http://127.0.0.1:$Port/dashboard"

# --- Background mode (detached, logs to file only) --------------------------
if ($Background) {
    $proc = Start-Process -FilePath $Python `
        -ArgumentList "-m", "muse2api" `
        -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput "data\logs\gateway.out.log" `
        -RedirectStandardError "data\logs\gateway.err.log"
    $proc.Id | Out-File -FilePath "data\gateway.pid" -Encoding ascii
    Write-Host "[muse2api] Running in background with PID: $($proc.Id)"
    Write-Host "[muse2api] Logs: $PSScriptRoot\data\logs\gateway.out.log"
    exit 0
}

# --- Foreground mode (default): stream logs live ----------------------------
Write-Host "[muse2api] Logs streaming below (Ctrl+C to stop)." -ForegroundColor Yellow
Write-Host "------------------------------------------------------------"

# -NoNewWindow inherits this console so logs stream here; -PassThru gives us
# the PID; we also record it so a later start can stop this instance.
$proc = Start-Process -FilePath $Python `
    -ArgumentList "-m", "muse2api" `
    -PassThru -NoNewWindow
$proc.Id | Out-File -FilePath "data\gateway.pid" -Encoding ascii

try {
    Wait-Process -Id $proc.Id
} finally {
    # Ensure the gateway is stopped if this script is interrupted.
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
}