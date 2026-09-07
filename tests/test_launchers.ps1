# Hardware-free behavior checks. Run with Windows PowerShell 5.1 or PowerShell 7.
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$tempRoot = Join-Path ([IO.Path]::GetTempPath()) ('RX6600 launcher tests ' + [guid]::NewGuid().ToString('N'))
$savedLocation = Get-Location
$environmentNames = @(
    'PATH', 'RX6600_PYTHON', 'ZLUDA_PATH', 'HIP_PATH', 'DISABLE_ADDMM_CUDA_LT',
    'TORCH_BLAS_PREFER_CUBLASLT', 'TORCH_BLAS_PREFER_HIPBLASLT', 'NVIDIA_TF32_OVERRIDE',
    'RX6600_TEST_RECORD', 'RX6600_TEST_EXIT'
)
$originalEnvironment = @{}
foreach ($name in $environmentNames) {
    $originalEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}

function Assert-Equal($Actual, $Expected, [string]$Message) {
    if ($Actual -cne $Expected) { throw "$Message. Expected <$Expected>, received <$Actual>." }
}

function Invoke-LauncherCase([string]$Wrapper, [object[]]$Forwarded = @()) {
    if (Test-Path -LiteralPath $env:RX6600_TEST_RECORD) {
        Remove-Item -LiteralPath $env:RX6600_TEST_RECORD
    }
    $before = @{}
    foreach ($name in $environmentNames) {
        $before[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
    }
    $beforeLocation = (Get-Location).Path
    $errorFile = Join-Path $tempRoot 'stderr.txt'
    $LASTEXITCODE = 99 # Native status must survive a caller's local variable of the same name.
    $output = @(& (Join-Path $fixtureScripts $Wrapper) @Forwarded 2> $errorFile)
    $code = $global:LASTEXITCODE
    Assert-Equal (Get-Location).Path $beforeLocation 'Launcher changed the caller location'
    foreach ($name in $environmentNames) {
        Assert-Equal ([Environment]::GetEnvironmentVariable($name, 'Process')) $before[$name] "Launcher changed parent $name"
    }
    $record = @{}
    if (Test-Path -LiteralPath $env:RX6600_TEST_RECORD) {
        foreach ($line in Get-Content -LiteralPath $env:RX6600_TEST_RECORD) {
            $key, $value = $line.Split(':', 2)
            $record[$key] = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($value))
        }
    }
    return @{ Code = $code; Output = $output; Record = $record; Error = (Get-Content -LiteralPath $errorFile -Raw) }
}

try {
    $fixtureRoot = Join-Path $tempRoot 'repo with spaces'
    $fixtureScripts = Join-Path $fixtureRoot 'scripts'
    $zludaRoot = Join-Path $tempRoot 'ZLUDA install'
    $hipRoot = Join-Path $tempRoot 'HIP SDK'
    $hipBin = Join-Path $hipRoot 'bin'
    $library = Join-Path $hipBin 'rocblas\library'
    $pythonDir = Join-Path $tempRoot 'Python install'
    foreach ($directory in @($fixtureScripts, $zludaRoot, $library, $pythonDir)) {
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
    }
    $wrappers = @{
        'zluda_run_train.ps1' = 'train_tinystories.py'
        'zluda_run_train_smoke.ps1' = 'training_smoke.py'
        'zluda_run_probe.ps1' = 'zluda_probe.py'
        'zluda_run_cublas_ctypes.ps1' = 'cublas_ctypes_test.py'
    }
    foreach ($name in @($wrappers.Keys) + @('zluda_common.ps1')) {
        Copy-Item -LiteralPath (Join-Path $repoRoot "scripts\$name") -Destination $fixtureScripts
    }
    foreach ($name in $wrappers.Values) { Set-Content -LiteralPath (Join-Path $fixtureRoot $name) -Value '# fixture' }
    foreach ($name in @('cublas.dll', 'nvcuda.dll', 'zluda_redirect.dll')) {
        Set-Content -LiteralPath (Join-Path $zludaRoot $name) -Value 'fixture'
    }
    foreach ($name in @('amdhip64_6.dll', 'rocblas.dll', 'rocsolver.dll')) {
        Set-Content -LiteralPath (Join-Path $hipBin $name) -Value 'fixture'
    }
    $kernel = Join-Path $library 'Kernels.so-000-gfx1032.hsaco'
    $metadata = Join-Path $library 'TensileLibrary.dat'
    Set-Content -LiteralPath $kernel -Value 'fixture'
    Set-Content -LiteralPath $metadata -Value 'fixture'
    Set-Content -LiteralPath (Join-Path $pythonDir 'python.exe') -Value 'fixture'

    # Compile a real native child, so tests exercise PowerShell's argument serialization.
    $source = Join-Path $tempRoot 'FakeZluda.cs'
    @'
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
public static class FakeZluda {
    static string Field(string key, string value) {
        return key + ":" + Convert.ToBase64String(Encoding.UTF8.GetBytes(value ?? "<unset>"));
    }
    public static int Main(string[] args) {
        var lines = new List<string>();
        lines.Add(Field("cwd", Environment.CurrentDirectory));
        lines.Add(Field("argc", args.Length.ToString()));
        for (int i = 0; i < args.Length; i++) lines.Add(Field("arg" + i, args[i]));
        foreach (var name in new[] { "PATH", "RX6600_PYTHON", "ZLUDA_PATH", "HIP_PATH",
            "DISABLE_ADDMM_CUDA_LT", "TORCH_BLAS_PREFER_CUBLASLT", "TORCH_BLAS_PREFER_HIPBLASLT",
            "NVIDIA_TF32_OVERRIDE" }) lines.Add(Field(name, Environment.GetEnvironmentVariable(name)));
        File.WriteAllLines(Environment.GetEnvironmentVariable("RX6600_TEST_RECORD"), lines);
        Console.WriteLine("native stdout");
        Console.Error.WriteLine("native stderr");
        return int.Parse(Environment.GetEnvironmentVariable("RX6600_TEST_EXIT"));
    }
}
'@ | Set-Content -LiteralPath $source
    $compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
    & $compiler /nologo /target:exe "/out:$(Join-Path $zludaRoot 'zluda.exe')" $source
    if ($LASTEXITCODE -ne 0) { throw 'Could not compile native launcher fixture.' }

    Set-Location -LiteralPath $tempRoot
    $env:RX6600_PYTHON = '.\Python install\python.exe'
    $env:ZLUDA_PATH = '.\ZLUDA install'
    $env:HIP_PATH = '.\HIP SDK'
    $env:DISABLE_ADDMM_CUDA_LT = 'parent value'
    $env:TORCH_BLAS_PREFER_CUBLASLT = '1'
    $env:TORCH_BLAS_PREFER_HIPBLASLT = '1'
    $env:NVIDIA_TF32_OVERRIDE = '1'
    $env:RX6600_TEST_RECORD = Join-Path $tempRoot 'child.txt'
    $env:RX6600_TEST_EXIT = '0'
    $forwarded = @('--steps', '2', '--data', 'Tiny Stories.txt', 'a"quoted"value', '', 'trailing space\')
    foreach ($wrapper in $wrappers.Keys) {
        $result = Invoke-LauncherCase $wrapper $forwarded
        Assert-Equal $result.Code 0 "${wrapper} failed: $($result.Error)"
        Assert-Equal ($result.Output -join '') 'native stdout' 'Native stdout was lost'
        if ($result.Error -notmatch 'native stderr') { throw 'Native stderr was lost.' }
        Assert-Equal $result.Record.cwd $fixtureRoot 'Child working directory was incorrect'
        $expected = @('--', (Join-Path $pythonDir 'python.exe'), (Join-Path $fixtureRoot $wrappers[$wrapper])) + $forwarded
        Assert-Equal $result.Record.argc ([string]$expected.Count) 'Wrong native argument count'
        for ($i = 0; $i -lt $expected.Count; $i++) {
            Assert-Equal $result.Record["arg$i"] $expected[$i] "Wrong native argument $i"
        }
        Assert-Equal $result.Record.HIP_PATH $hipRoot 'HIP_PATH was not resolved'
        Assert-Equal $result.Record.ZLUDA_PATH $zludaRoot 'ZLUDA_PATH was not resolved'
        Assert-Equal $result.Record.RX6600_PYTHON (Join-Path $pythonDir 'python.exe') 'Python path was not resolved'
        Assert-Equal $result.Record.PATH "$hipBin;$zludaRoot;$env:PATH" 'DLL search path was incorrect'
        Assert-Equal $result.Record.DISABLE_ADDMM_CUDA_LT '1' 'Legacy cuBLAS was not selected'
        Assert-Equal $result.Record.TORCH_BLAS_PREFER_CUBLASLT '<unset>' 'cuBLASLt preference was not cleared'
        Assert-Equal $result.Record.TORCH_BLAS_PREFER_HIPBLASLT '<unset>' 'hipBLASLt preference was not cleared'
        Assert-Equal $result.Record.NVIDIA_TF32_OVERRIDE '0' 'TF32 was not disabled'
    }

    $env:RX6600_TEST_EXIT = '37'
    Assert-Equal (Invoke-LauncherCase 'zluda_run_train.ps1').Code 37 'Native nonzero status was lost'
    $shellPath = (Get-Process -Id $PID).Path
    $ErrorActionPreference = 'Continue'
    $shellOutput = @(& $shellPath -NoProfile -ExecutionPolicy Bypass -File (Join-Path $fixtureScripts 'zluda_run_train.ps1') --steps 2 --data 'Tiny Stories.txt' 2>&1)
    $ErrorActionPreference = 'Stop'
    Assert-Equal $LASTEXITCODE 37 'The documented -File command lost the native status'
    $env:RX6600_TEST_EXIT = '0'
    # Unset parent variables must also remain absent after a successful child.
    foreach ($name in @('DISABLE_ADDMM_CUDA_LT', 'TORCH_BLAS_PREFER_CUBLASLT', 'TORCH_BLAS_PREFER_HIPBLASLT', 'NVIDIA_TF32_OVERRIDE')) {
        [Environment]::SetEnvironmentVariable($name, [NullString]::Value, 'Process')
    }
    Assert-Equal (Invoke-LauncherCase 'zluda_run_probe.ps1').Code 0 'Unset parent environment case failed'

    $defaultPythonDir = Join-Path $fixtureRoot '.venv\Scripts'
    New-Item -ItemType Directory -Path $defaultPythonDir -Force | Out-Null
    Set-Content -LiteralPath (Join-Path $defaultPythonDir 'python.exe') -Value 'fixture'
    $env:RX6600_PYTHON = $null
    $result = Invoke-LauncherCase 'zluda_run_probe.ps1'
    Assert-Equal $result.Code 0 'Default repository Python failed'
    Assert-Equal $result.Record.arg1 (Join-Path $defaultPythonDir 'python.exe') 'Default Python used the caller directory'
    $env:RX6600_PYTHON = '.\Python install\python.exe'

    # Each missing dependency must fail before executing the native process.
    foreach ($missing in @(
        (Join-Path $pythonDir 'python.exe'), (Join-Path $zludaRoot 'zluda.exe'),
        (Join-Path $zludaRoot 'cublas.dll'), (Join-Path $hipBin 'amdhip64_6.dll'),
        (Join-Path $hipBin 'rocblas.dll'), (Join-Path $hipBin 'rocsolver.dll'), $kernel, $metadata
    )) {
        $bytes = [IO.File]::ReadAllBytes($missing)
        Remove-Item -LiteralPath $missing
        try {
            $result = Invoke-LauncherCase 'zluda_run_probe.ps1'
            Assert-Equal $result.Code 1 "Missing $missing did not fail"
            Assert-Equal $result.Record.Count 0 "Missing $missing still launched a child"
            if ($result.Error -notmatch 'SETUP.md') { throw "Missing $missing did not provide setup guidance." }
        }
        finally { [IO.File]::WriteAllBytes($missing, $bytes) }
    }

    $fakeExecutable = Join-Path $zludaRoot 'zluda.exe'
    [IO.File]::WriteAllText($fakeExecutable, 'not an executable')
    $result = Invoke-LauncherCase 'zluda_run_probe.ps1'
    Assert-Equal $result.Code 1 'Native startup failure did not fail'
    Assert-Equal $result.Record.Count 0 'Invalid executable unexpectedly started'
    Write-Output "Launcher tests passed on PowerShell $($PSVersionTable.PSVersion)."
}
finally {
    Set-Location -LiteralPath $savedLocation.Path
    foreach ($name in $environmentNames) {
        $value = $originalEnvironment[$name]
        if ($null -eq $value) { $value = [NullString]::Value }
        [Environment]::SetEnvironmentVariable($name, $value, 'Process')
    }
    if (Test-Path -LiteralPath $tempRoot) {
        $resolvedTempRoot = (Resolve-Path -LiteralPath $tempRoot).ProviderPath
        $expectedTempRoot = [IO.Path]::GetFullPath($tempRoot)
        if ($resolvedTempRoot -ne $expectedTempRoot -or -not $expectedTempRoot.StartsWith([IO.Path]::GetTempPath(), [StringComparison]::OrdinalIgnoreCase)) {
            throw 'Refusing to remove an unexpected fixture directory.'
        }
        Remove-Item -LiteralPath $resolvedTempRoot -Recurse -Force
    }
}
