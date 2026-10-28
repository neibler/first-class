param(
    [ValidateSet('api','ui','test','train','preflight','news_search','prepare_market','evaluate','infer')][string]$Task = 'api',
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$ExtraArgs
)
$ErrorActionPreference = 'Stop'
$runtimePath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $runtimePath)) {
    $runtimePath = Join-Path $PSScriptRoot '..\..\work\python\python.exe'
}
if (-not (Test-Path -LiteralPath $runtimePath)) {
    throw 'Create .venv and install requirements-dev.txt first. See LOCAL_INTEGRATION.md.'
}
& $runtimePath (Join-Path $PSScriptRoot 'local_run.py') $Task @ExtraArgs
if ($LASTEXITCODE -ne 0) { throw "Execution failed: $LASTEXITCODE" }
