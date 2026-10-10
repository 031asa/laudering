param(
    [ValidateSet('build', 'run', 'query', 'verify', 'test')][string]$Job = 'query',
    [string]$Python = 'C:\anaconda\python.exe',
    [string]$Archive = 'data/gnn_v1_dev',
    [string]$Output = 'result/gnn_v1_dev',
    [int]$NodeId = 0,
    [string]$Cutoff = '2022-09-04T12:00:00'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
$previousPath = $env:PYTHONPATH
$previousEncoding = $env:PYTHONIOENCODING
try {
    $env:PYTHONPATH = $PSScriptRoot
    $env:PYTHONIOENCODING = 'utf-8'
    switch ($Job) {
        'build' { & $Python -m gnn_features.build --output $Archive --end '2022-09-04T13:00:00' }
        'run' { & $Python -m gnn_features.run --archive $Archive --output $Output }
        'query' { & $Python -m gnn_features.query --archive $Archive --node-id $NodeId --cutoff $Cutoff }
        'verify' { & $Python -m gnn_features.verify --archive $Archive --output $Output }
        'test' { & $Python -m pytest tests/test_gnn_features.py -q }
    }
    if ($LASTEXITCODE -ne 0) { throw "GNN feature job failed: $Job" }
}
finally {
    $env:PYTHONPATH = $previousPath
    $env:PYTHONIOENCODING = $previousEncoding
    Pop-Location
}
