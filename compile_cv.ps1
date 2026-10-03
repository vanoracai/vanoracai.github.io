param(
    [ValidateSet('zh', 'en', 'all')]
    [string]$Language = 'all',
    [switch]$Force
)
$ErrorActionPreference = 'Stop'

# Compile saved sources without regenerating or overwriting editor changes.
$taskPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    $taskPython = (Get-Command python -ErrorAction Stop).Source
}
$taskCompilerArgs = @((Join-Path $PSScriptRoot 'scripts/compile_cv.py'), '--language', $Language)
if ($Force) { $taskCompilerArgs += '--force' }
& $taskPython @taskCompilerArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
