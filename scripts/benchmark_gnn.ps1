param(
    [ValidateSet('reference', 'rolling', 'spark')][string]$Mode = 'rolling',
    [string]$Output = 'result/gnn_benchmark_rolling',
    [int]$Partitions = 8,
    [int]$Cores = 2,
    [string]$NodeReference = '',
    [switch]$WarmRepeat,
    [string]$RuntimeRoot = 'C:\Users\28118\Documents\hive_spark_project'
)
$ErrorActionPreference = 'Stop'
$python = Join-Path $RuntimeRoot '.venv\Scripts\python.exe'
$java = Join-Path $RuntimeRoot '.runtime\jdk17\jdk-17.0.19+10'
$jar = Join-Path $RuntimeRoot '.runtime\jars\hadoop-bare-naked-local-fs-0.1.0.jar'
foreach ($required in @($python, (Join-Path $java 'bin\java.exe'), $jar)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Required runtime path missing: $required" }
}
$env:JAVA_HOME = $java
$env:SPARK_HOME = Join-Path $RuntimeRoot '.venv\Lib\site-packages\pyspark'
$env:PYSPARK_PYTHON = $python
$env:PYSPARK_DRIVER_PYTHON = $python
$env:PYTHONPATH = $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
$env:SPARK_LOCAL_IP = '127.0.0.1'
$env:PYSPARK_SUBMIT_ARGS = "--driver-memory 2g --driver-class-path `"$jar`" pyspark-shell"
Push-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
try {
    $benchmarkArguments = @('-m', 'gnn_features.benchmark', '--mode', $Mode, '--output', $Output, '--partitions', $Partitions, '--cores', $Cores)
    if ($NodeReference) { $benchmarkArguments += @('--node-reference', $NodeReference) }
    if ($WarmRepeat) { $benchmarkArguments += '--warm-repeat' }
    & $python @benchmarkArguments
    if ($LASTEXITCODE -ne 0) { throw "Benchmark failed: $Mode" }
} finally { Pop-Location }
