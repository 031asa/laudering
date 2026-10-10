param([string]$RuntimeRoot = 'C:\Users\28118\Documents\hive_spark_project',
      [ValidateSet('audit', 'find_duplicates', 'partition_dates', 'standardize')][string]$Job = 'audit')
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
$env:SPARK_LOCAL_IP = '127.0.0.1'
$env:PYSPARK_SUBMIT_ARGS = "--driver-memory 2g --driver-class-path `"$jar`" pyspark-shell"
& $python (Join-Path $PSScriptRoot ($Job + '.py'))
if ($LASTEXITCODE -ne 0) { throw "Audit failed with exit code $LASTEXITCODE" }
