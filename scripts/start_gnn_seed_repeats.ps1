param([switch]$Resume, [switch]$AcceptOnly)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$repeatPython = Join-Path $projectRoot '.venv-gnn-gpu/Scripts/python.exe'
$outputRoot = Join-Path $projectRoot 'result/gnn_seed_repeats'
if (-not (Test-Path -LiteralPath $repeatPython)) { throw 'Project GPU Python is missing.' }
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$stdoutPath = Join-Path $outputRoot ("run_$stamp.stdout.log")
$stderrPath = Join-Path $outputRoot ("run_$stamp.stderr.log")
$previousPythonPath = $env:PYTHONPATH
$previousEncoding = $env:PYTHONIOENCODING
try {
    $env:PYTHONPATH = Join-Path $projectRoot 'scripts'
    $env:PYTHONIOENCODING = 'utf-8'
    $arguments = @('-m', 'gnn_features.repeat_pipeline')
    if ($Resume) { $arguments += '--resume' }
    if ($AcceptOnly) { $arguments += '--accept-only' }
    $process = Start-Process -FilePath $repeatPython -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
    [pscustomobject]@{LauncherPid=$process.Id; StatusFile=(Join-Path $outputRoot 'status.json'); Stdout=$stdoutPath; Stderr=$stderrPath}
} finally {
    $env:PYTHONPATH = $previousPythonPath
    $env:PYTHONIOENCODING = $previousEncoding
}
