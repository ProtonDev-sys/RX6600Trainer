$env:PATH = "C:\Program Files\AMD\ROCm\6.2\bin;C:\Users\HTD\zluda;" + $env:PATH
$env:DISABLE_ADDMM_CUDA_LT = '1'

$tlog = 'C:\Users\HTD\AppData\Local\Temp\opencode\train_smoke_result.txt'
Remove-Item $tlog -ErrorAction SilentlyContinue

$p = Start-Process -FilePath 'C:\Users\HTD\zluda\zluda.exe' -ArgumentList '--','C:\Users\HTD\Downloads\Projects\Cig\.venv\Scripts\python.exe','C:\Users\HTD\Downloads\Projects\Cig\training_smoke.py' -RedirectStandardOutput 'C:\Users\HTD\AppData\Local\Temp\opencode\train_smoke_stdout.txt' -RedirectStandardError 'C:\Users\HTD\AppData\Local\Temp\opencode\train_smoke_stderr.txt' -Wait -PassThru
Write-Output ("zluda.exe exit code: " + $p.ExitCode)

Write-Output '--- result ---'
if (Test-Path $tlog) { Get-Content $tlog } else { Write-Output '(none)' }
Write-Output '--- stdout ---'
if (Test-Path 'C:\Users\HTD\AppData\Local\Temp\opencode\train_smoke_stdout.txt') { Get-Content 'C:\Users\HTD\AppData\Local\Temp\opencode\train_smoke_stdout.txt' } else { Write-Output '(none)' }
Write-Output '--- stderr ---'
if (Test-Path 'C:\Users\HTD\AppData\Local\Temp\opencode\train_smoke_stderr.txt') { Get-Content 'C:\Users\HTD\AppData\Local\Temp\opencode\train_smoke_stderr.txt' } else { Write-Output '(none)' }