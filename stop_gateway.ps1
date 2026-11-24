<#
.SYNOPSIS
    Stop the muse2api gateway (Windows native, PowerShell).
#>
$ErrorActionPreference = "SilentlyContinue"
$Port = 18610

# Stop by recorded PID file.
if (Test-Path "data\gateway.pid") {
    $pidText = (Get-Content "data\gateway.pid" -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($pidText -and [int]::TryParse($pidText.Trim(), [ref]$null)) {
        $procId = [int]$pidText.Trim()
        if (Get-Process -Id $procId -ErrorAction SilentlyContinue) {
            Write-Host "[muse2api] Stopping gateway (PID $procId)..." -ForegroundColor Yellow
            Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
        }
    }
    Remove-Item "data\gateway.pid" -Force -ErrorAction SilentlyContinue
}

# Kill anything still bound to the port (orphans / stale instances).
try {
    $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    foreach ($conn in $conns) {
        if ($conn.OwningProcess -gt 0) {
            Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
        }
    }
} catch {
    # Fallback: match by command line.
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like "*-m muse2api*" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

Write-Host "[muse2api] Gateway stopped."