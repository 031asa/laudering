param([string]$Output = '')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$gpuPython = Join-Path $projectRoot '.venv-gnn-gpu/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $gpuPython)) { throw 'Project GPU environment is missing; see README.' }
if (-not $Output) { $Output = Join-Path $projectRoot ('result/gnn_gpu_smoke_' + (Get-Date -Format 'yyyyMMdd_HHmmss')) }
$previousPythonPath = $env:PYTHONPATH
$previousEncoding = $env:PYTHONIOENCODING
Push-Location $projectRoot
try {
    $env:PYTHONPATH = Join-Path $projectRoot 'scripts'
    $env:PYTHONIOENCODING = 'utf-8'
    & $gpuPython -m gnn_features.smoke_model --device cuda --output $Output
    if ($LASTEXITCODE -ne 0) { throw "GPU acceptance failed (exit $LASTEXITCODE)." }
} finally {
    $env:PYTHONPATH = $previousPythonPath
    $env:PYTHONIOENCODING = $previousEncoding
    Pop-Location
}
