& (Join-Path $PSScriptRoot 'zluda_common.ps1') -ScriptName 'train_tinystories.py' -ScriptArguments $args
exit $global:LASTEXITCODE
