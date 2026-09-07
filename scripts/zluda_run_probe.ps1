& (Join-Path $PSScriptRoot 'zluda_common.ps1') -ScriptName 'zluda_probe.py' -ScriptArguments $args
exit $global:LASTEXITCODE
