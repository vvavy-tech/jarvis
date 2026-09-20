$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir 'jarvis-agent.log'
Set-Location $root

$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) {
    'uv not found on PATH' | Add-Content $log
    exit 1
}

while ($true) {
    "[{0}] starting agent" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | Add-Content $log
    & $uv.Source run python src/agent.py dev 2>&1 | ForEach-Object { $_ | Add-Content $log }
    "[{0}] agent exited with code {1}; restarting in 5s" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $LASTEXITCODE | Add-Content $log
    Start-Sleep -Seconds 5
}