<#
.SYNOPSIS
    Stop the muse2api gateway and all associated browser processes (Windows native, PowerShell).
#>
$ErrorActionPreference = "SilentlyContinue"
$Port = 18610
$CdpPort = 19210

# 1. Stop by recorded PID file.
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

# 2. Kill anything bound to gateway port 18610 and CDP port 19210.
try {
    foreach ($p in @($Port, $CdpPort)) {
        $conns = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue
        foreach ($conn in $conns) {
            if ($conn.OwningProcess -gt 0) {
                Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
            }
        }
    }
} catch {
    # Fallback to process matching
}

# 3. Kill lingering Python gateway instances and browser helper processes
Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*-m muse2api*" -or $_.CommandLine -like "*auto_login.py*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

Get-CimInstance Win32_Process -Filter "Name='chrome.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*browser-profile*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

# 4. Clear stale browser singleton locks
Remove-Item "data\browser-profile\Singleton*" -Force -ErrorAction SilentlyContinue

Write-Host "[muse2api] Gateway stopped."
