$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$frontend = Join-Path $root 'jarvis\frontend'
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir 'jarvis-frontend.log'
Set-Location $frontend

$pnpm = Get-Command pnpm -ErrorAction SilentlyContinue
if (-not $pnpm) {
    'pnpm not found on PATH' | Add-Content $log
    exit 1
}

while ($true) {
    "[{0}] starting frontend" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | Add-Content $log
    & $pnpm.Source dev 2>&1 | ForEach-Object { $_ | Add-Content $log }
    "[{0}] frontend exited with code {1}; restarting in 5s" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $LASTEXITCODE | Add-Content $log
    Start-Sleep -Seconds 5
}