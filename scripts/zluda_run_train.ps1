$env:PATH = "C:\Program Files\AMD\ROCm\6.2\bin;C:\Users\HTD\zluda;" + $env:PATH
$env:DISABLE_ADDMM_CUDA_LT = '1'

$live = 'C:\Users\HTD\AppData\Local\Temp\opencode\train_live.txt'
Remove-Item $live -ErrorAction SilentlyContinue

$argsList = @('--','C:\Users\HTD\Downloads\Projects\Cig\.venv\Scripts\python.exe','C:\Users\HTD\Downloads\Projects\Cig\train_tinystories.py') + $args
& 'C:\Users\HTD\zluda\zluda.exe' $argsList 2>&1 | Tee-Object -FilePath $live
Write-Output ("zluda.exe exit code: " + $LASTEXITCODE)