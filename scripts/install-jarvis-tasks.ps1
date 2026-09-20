$ErrorActionPreference = 'Stop'

if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Start-Process -FilePath 'powershell.exe' -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`"" -Verb RunAs -Wait
    exit
}

$root = Split-Path -Parent $PSScriptRoot
$agentScript = Join-Path $root 'scripts\start-jarvis-agent.ps1'
$frontendScript = Join-Path $root 'scripts\start-jarvis-frontend.ps1'
$agentArg = "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$agentScript`""
$frontendArg = "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$frontendScript`""

$trigger = New-ScheduledTaskTrigger -AtLogOn
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew

$actionAgent = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $agentArg
$actionFrontend = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $frontendArg

Register-ScheduledTask -TaskName 'Jarvis Agent' -Action $actionAgent -Trigger $trigger -Principal $principal -Settings $settings -Description 'LiveKit voice agent (uv run python src/agent.py dev)' -Force
Register-ScheduledTask -TaskName 'Jarvis Frontend' -Action $actionFrontend -Trigger $trigger -Principal $principal -Settings $settings -Description 'Jarvis voice assistant web frontend (next dev)' -Force

Start-ScheduledTask -TaskName 'Jarvis Agent'
Start-ScheduledTask -TaskName 'Jarvis Frontend'

Write-Host 'Jarvis tasks registered and started.'