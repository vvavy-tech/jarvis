$root = Split-Path -Parent $PSScriptRoot
& (Join-Path $root 'scripts\stop-jarvis.ps1')

foreach ($name in @('Jarvis Agent', 'Jarvis Frontend')) {
    if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $name -Confirm:$false
    }
}

Write-Host 'Jarvis auto-start removed.'