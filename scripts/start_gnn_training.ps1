$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$trainPython = Join-Path $projectRoot '.venv-gnn-gpu/Scripts/python.exe'
$outputRoot = Join-Path $projectRoot 'result/gnn_train_gpu_v1'
if (-not (Test-Path -LiteralPath $trainPython)) { throw 'Project GPU Python is missing.' }
$cacheRoot = Join-Path $projectRoot 'data/gnn_train_cache_parquet'
$cacheState = Get-Content -LiteralPath (Join-Path $cacheRoot 'status.json') -Raw | ConvertFrom-Json
if ($cacheState.status -ne 'completed' -or $cacheState.verified_targets -ne 20318992) { throw 'Complete verified Train cache is required.' }
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$stdoutPath = Join-Path $outputRoot ("run_$stamp.stdout.log")
$stderrPath = Join-Path $outputRoot ("run_$stamp.stderr.log")
$oldPythonPath = $env:PYTHONPATH
$oldEncoding = $env:PYTHONIOENCODING
try {
    $env:PYTHONPATH = Join-Path $projectRoot 'scripts'
    $env:PYTHONIOENCODING = 'utf-8'
    $process = Start-Process -FilePath $trainPython -ArgumentList @('-m', 'gnn_features.train_model') -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
    [pscustomobject]@{LauncherPid=$process.Id; StatusFile=(Join-Path $outputRoot 'status.json'); Stdout=$stdoutPath; Stderr=$stderrPath}
} finally {
    $env:PYTHONPATH = $oldPythonPath
    $env:PYTHONIOENCODING = $oldEncoding
}
