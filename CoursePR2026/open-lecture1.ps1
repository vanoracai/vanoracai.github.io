param(
  [switch]$NoOpen
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$server = Join-Path $root "preview_server.py"
$url = "http://127.0.0.1:8000/?file=lecture_1.md"

function Test-PreviewServer {
  try {
    $response = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:8000/lecture_1.md" -TimeoutSec 2
    return $response.StatusCode -eq 200
  }
  catch {
    return $false
  }
}

if (-not (Test-Path -LiteralPath $server)) {
  throw "Cannot find preview_server.py in $root"
}

if (-not (Test-PreviewServer)) {
  $python = Get-Command python -ErrorAction SilentlyContinue
  if (-not $python) {
    throw "Python was not found on PATH. Install Python or run this from an environment where python is available."
  }

  Start-Process -FilePath $python.Source `
    -ArgumentList @($server) `
    -WorkingDirectory $root `
    -WindowStyle Hidden

  $ready = $false
  for ($i = 0; $i -lt 20; $i++) {
    Start-Sleep -Milliseconds 250
    if (Test-PreviewServer) {
      $ready = $true
      break
    }
  }

  if (-not $ready) {
    throw "Preview server did not start on http://127.0.0.1:8000/"
  }
}

Write-Host "Lecture 1 preview:"
Write-Host $url

if (-not $NoOpen) {
  Start-Process $url
}
