& (Join-Path $PSScriptRoot 'zluda_common.ps1') -ScriptName 'cublas_ctypes_test.py' -ScriptArguments $args
exit $global:LASTEXITCODE
