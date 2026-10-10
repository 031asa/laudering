param(
    [string]$Archive = 'data/gnn_v1_day_20220904',
    [string]$Output = 'result/gnn_day_20220904',
    [string]$Date = '2022-09-04',
    [int]$Cores = 4,
    [int]$Partitions = 8,
    [switch]$Resume,
    [string]$RuntimeRoot = 'C:\Users\28118\Documents\hive_spark_project'
)
$ErrorActionPreference = 'Stop'
$python = Join-Path $RuntimeRoot '.venv\Scripts\python.exe'
$java = Join-Path $RuntimeRoot '.runtime\jdk17\jdk-17.0.19+10'
$jar = Join-Path $RuntimeRoot '.runtime\jars\hadoop-bare-naked-local-fs-0.1.0.jar'
foreach ($required in @($python, (Join-Path $java 'bin\java.exe'), $jar)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Required runtime path missing: $required" }
}
$settings = @{
    JAVA_HOME = $java
    SPARK_HOME = (Join-Path $RuntimeRoot '.venv\Lib\site-packages\pyspark')
    PYSPARK_PYTHON = $python
    PYSPARK_DRIVER_PYTHON = $python
    PYTHONPATH = $PSScriptRoot
    PYTHONIOENCODING = 'utf-8'
    PYTHONFAULTHANDLER = '1'
    SPARK_LOCAL_IP = '127.0.0.1'
    PYSPARK_SUBMIT_ARGS = "--driver-memory 2g --conf spark.python.worker.faulthandler.enabled=true --driver-class-path `"$jar`" pyspark-shell"
}
$previous = @{}
foreach ($key in $settings.Keys) { $previous[$key] = [Environment]::GetEnvironmentVariable($key); [Environment]::SetEnvironmentVariable($key, $settings[$key]) }
Push-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
try {
    $jobArguments = @('-m', 'gnn_features.daily', '--archive', $Archive, '--output', $Output, '--date', $Date, '--cores', $Cores, '--partitions', $Partitions)
    if ($Resume) { $jobArguments += '--resume' }
    & $python @jobArguments
    if ($LASTEXITCODE -ne 0) { throw 'Daily GNN feature job failed' }
} finally {
    Pop-Location
    foreach ($key in $settings.Keys) { [Environment]::SetEnvironmentVariable($key, $previous[$key]) }
}
