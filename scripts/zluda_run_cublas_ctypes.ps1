$env:PATH = "C:\Program Files\AMD\ROCm\6.2\bin;C:\Users\HTD\zluda;" + $env:PATH
$env:DISABLE_ADDMM_CUDA_LT = '1'

$tlog = 'C:\Users\HTD\AppData\Local\Temp\opencode\cublas_ctypes_log.txt'
Remove-Item $tlog -ErrorAction SilentlyContinue

$p = Start-Process -FilePath 'C:\Users\HTD\zluda\zluda.exe' -ArgumentList '--','C:\Users\HTD\Downloads\Projects\Cig\.venv\Scripts\python.exe','C:\Users\HTD\Downloads\Projects\Cig\cublas_ctypes_test.py' -RedirectStandardOutput 'C:\Users\HTD\AppData\Local\Temp\opencode\cublas_ctypes_stdout.txt' -RedirectStandardError 'C:\Users\HTD\AppData\Local\Temp\opencode\cublas_ctypes_stderr.txt' -Wait -PassThru
Write-Output ("zluda.exe exit code: " + $p.ExitCode)

Write-Output '--- test log ---'
if (Test-Path $tlog) { Get-Content $tlog } else { Write-Output '(none)' }
Write-Output '--- stderr ---'
if (Test-Path 'C:\Users\HTD\AppData\Local\Temp\opencode\cublas_ctypes_stderr.txt') { Get-Content 'C:\Users\HTD\AppData\Local\Temp\opencode\cublas_ctypes_stderr.txt' } else { Write-Output '(none)' }