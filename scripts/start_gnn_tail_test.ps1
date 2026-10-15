param([switch]$Resume)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$testPython = Join-Path $projectRoot '.venv-gnn-gpu/Scripts/python.exe'
$outputRoot = Join-Path $projectRoot 'result/gnn_tail_test'
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$oldPath = $env:PYTHONPATH
$oldEncoding = $env:PYTHONIOENCODING
try {
    $env:PYTHONPATH = $PSScriptRoot
    $env:PYTHONIOENCODING = 'utf-8'
    $arguments = @('-m','gnn_tail_test.pipeline')
    if ($Resume) { Write-Host 'Resuming only verified tail artifacts.' }
    $process = Start-Process -FilePath $testPython -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $outputRoot "run_$stamp.stdout.log") -RedirectStandardError (Join-Path $outputRoot "run_$stamp.stderr.log") -PassThru
    [pscustomobject]@{LauncherPid=$process.Id;StatusFile=(Join-Path $outputRoot 'status.json')}
} finally {
    $env:PYTHONPATH = $oldPath
    $env:PYTHONIOENCODING = $oldEncoding
}
