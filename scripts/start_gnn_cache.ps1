$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$cachePython = Join-Path $projectRoot '.venv-gnn-gpu/Scripts/python.exe'
$outputRoot = Join-Path $projectRoot 'data/gnn_train_cache_parquet'
if (-not (Test-Path -LiteralPath $cachePython)) { throw 'Project Python environment is missing.' }
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$stdoutPath = Join-Path $outputRoot ("run_$stamp.stdout.log")
$stderrPath = Join-Path $outputRoot ("run_$stamp.stderr.log")
$oldPythonPath = $env:PYTHONPATH
$oldEncoding = $env:PYTHONIOENCODING
try {
    $env:PYTHONPATH = Join-Path $projectRoot 'scripts'
    $env:PYTHONIOENCODING = 'utf-8'
    $process = Start-Process -FilePath $cachePython -ArgumentList @('-m', 'gnn_features.cache_train') -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
    [pscustomobject]@{Pid=$process.Id; StatusFile=(Join-Path $outputRoot 'status.json'); Stdout=$stdoutPath; Stderr=$stderrPath}
} finally {
    $env:PYTHONPATH = $oldPythonPath
    $env:PYTHONIOENCODING = $oldEncoding
}
