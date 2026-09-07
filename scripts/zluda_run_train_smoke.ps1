& (Join-Path $PSScriptRoot 'zluda_common.ps1') -ScriptName 'training_smoke.py' -ScriptArguments $args
exit $global:LASTEXITCODE
