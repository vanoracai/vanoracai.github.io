param([switch]$Build, [switch]$Check, [switch]$Watch, [switch]$SkipCV)
$ErrorActionPreference = 'Stop'
$taskPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    $taskPython = (Get-Command python -ErrorAction Stop).Source
}
$taskGeneratorArgs = @((Join-Path $PSScriptRoot 'scripts/generate_all.py'))
if ($Build) { $taskGeneratorArgs += '--build' }
if ($Check) { $taskGeneratorArgs += '--check' }
if ($Watch) { $taskGeneratorArgs += '--watch' }
if ($SkipCV) { $taskGeneratorArgs += '--skip-cv' }
& $taskPython @taskGeneratorArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
