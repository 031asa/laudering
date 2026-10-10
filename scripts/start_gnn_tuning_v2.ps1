param([switch]$Resume)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$tunePython = Join-Path $projectRoot '.venv-gnn-gpu/Scripts/python.exe'
$outputRoot = Join-Path $projectRoot 'result/gnn_tuning_v2'
if (-not (Test-Path -LiteralPath $tunePython)) { throw 'Project GPU Python is missing.' }
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$stdoutPath = Join-Path $outputRoot ("run_$stamp.stdout.log")
$stderrPath = Join-Path $outputRoot ("run_$stamp.stderr.log")
$oldPythonPath = $env:PYTHONPATH
$oldEncoding = $env:PYTHONIOENCODING
try {
    $env:PYTHONPATH = Join-Path $projectRoot 'scripts'
    $env:PYTHONIOENCODING = 'utf-8'
    $arguments = @('-m', 'gnn_features.v2_pipeline')
    if ($Resume) { $arguments += '--resume' }
    $process = Start-Process -FilePath $tunePython -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
    [pscustomobject]@{LauncherPid=$process.Id; StatusFile=(Join-Path $outputRoot 'status.json'); Stdout=$stdoutPath; Stderr=$stderrPath}
} finally {
    $env:PYTHONPATH = $oldPythonPath
    $env:PYTHONIOENCODING = $oldEncoding
}
