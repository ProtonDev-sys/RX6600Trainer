param(
    [Parameter(Mandatory = $true)]
    [string]$ScriptName,
    [object[]]$ScriptArguments = @()
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$environmentNames = @(
    'PATH', 'RX6600_PYTHON', 'ZLUDA_PATH', 'HIP_PATH',
    'DISABLE_ADDMM_CUDA_LT', 'TORCH_BLAS_PREFER_CUBLASLT',
    'TORCH_BLAS_PREFER_HIPBLASLT', 'NVIDIA_TF32_OVERRIDE'
)
$savedEnvironment = @{}
foreach ($name in $environmentNames) {
    $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
$locationPushed = $false
$exitCode = 1

function Resolve-LauncherPath([string]$Name, [string]$Default, [string]$PathType) {
    $value = [Environment]::GetEnvironmentVariable($Name, 'Process')
    if ([string]::IsNullOrWhiteSpace($value)) { $value = $Default }
    if (-not (Test-Path -LiteralPath $value -PathType $PathType)) {
        throw "$Name path does not exist: $value. Configure $Name and follow SETUP.md."
    }
    return (Resolve-Path -LiteralPath $value).ProviderPath
}

function Assert-LauncherFile([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required file is missing: $Path. Follow SETUP.md to prepare the RX 6600 environment."
    }
}

try {
    $python = Resolve-LauncherPath 'RX6600_PYTHON' (Join-Path $repoRoot '.venv\Scripts\python.exe') 'Leaf'
    $zludaRoot = Resolve-LauncherPath 'ZLUDA_PATH' (Join-Path $env:USERPROFILE 'zluda') 'Container'
    $hipRoot = Resolve-LauncherPath 'HIP_PATH' 'C:\Program Files\AMD\ROCm\6.2' 'Container'
    $hipBin = Join-Path $hipRoot 'bin'
    $zluda = Join-Path $zludaRoot 'zluda.exe'
    $scriptPath = Join-Path $repoRoot $ScriptName
    Assert-LauncherFile $scriptPath
    foreach ($name in @('zluda.exe', 'cublas.dll', 'nvcuda.dll', 'zluda_redirect.dll')) {
        Assert-LauncherFile (Join-Path $zludaRoot $name)
    }
    foreach ($name in @('amdhip64_6.dll', 'rocblas.dll', 'rocsolver.dll')) {
        Assert-LauncherFile (Join-Path $hipBin $name)
    }

    $libraryPath = Join-Path $hipBin 'rocblas\library'
    if (-not (Test-Path -LiteralPath $libraryPath -PathType Container)) {
        throw "rocBLAS library directory is missing: $libraryPath. Install the gfx1032 package described in SETUP.md."
    }
    $libraryFiles = @(Get-ChildItem -LiteralPath $libraryPath -File -Recurse)
    $metadata = @($libraryFiles | Where-Object { $_.Name -like 'TensileLibrary*.dat' -or $_.Name -like 'TensileLibrary*.yaml' })
    $kernels = @($libraryFiles | Where-Object { $_.Name -like '*gfx1032*.hsaco' -or $_.Name -like '*gfx1032*.co' })
    if ($metadata.Count -eq 0 -or $kernels.Count -eq 0) {
        throw "rocBLAS gfx1032 kernels or TensileLibrary metadata are missing in $libraryPath. Install the RX 6600 package described in SETUP.md."
    }

    $env:RX6600_PYTHON = $python
    $env:ZLUDA_PATH = $zludaRoot
    $env:HIP_PATH = $hipRoot
    $env:PATH = "$hipBin;$zludaRoot;$env:PATH"
    $env:DISABLE_ADDMM_CUDA_LT = '1'
    $env:TORCH_BLAS_PREFER_CUBLASLT = $null
    $env:TORCH_BLAS_PREFER_HIPBLASLT = $null
    $env:NVIDIA_TF32_OVERRIDE = '0'
    Push-Location -LiteralPath $repoRoot
    $locationPushed = $true

    # Native stderr remains visible; a child's nonzero status is returned unchanged.
    $ErrorActionPreference = 'Continue'
    $PSNativeCommandUseErrorActionPreference = $false
    $arguments = @('--', $python, $scriptPath) + $ScriptArguments
    # Windows PowerShell serializes native arguments using its legacy quoting rules.
    if ($PSVersionTable.PSVersion -lt [version]'7.3') {
        $arguments = @($arguments | ForEach-Object {
            $escaped = [regex]::Replace([string]$_, '(\\*)"', '$1$1\"')
            '"' + [regex]::Replace($escaped, '(\\+)$', '$1$1') + '"'
        })
    }
    else {
        $PSNativeCommandArgumentPassing = 'Standard'
    }
    $global:LASTEXITCODE = 1
    & $zluda @arguments
    $exitCode = $global:LASTEXITCODE
}
catch {
    Write-Error "RX 6600 launcher: $($_.Exception.Message)" -ErrorAction Continue
    $exitCode = 1
}
finally {
    if ($locationPushed) { Pop-Location }
    foreach ($name in $environmentNames) {
        $value = $savedEnvironment[$name]
        if ($null -eq $value) { $value = [NullString]::Value }
        [Environment]::SetEnvironmentVariable($name, $value, 'Process')
    }
}
exit $exitCode
