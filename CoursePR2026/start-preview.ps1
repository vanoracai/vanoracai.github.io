$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonw = "D:\miniconda3\pythonw.exe"
$server = Join-Path $root "preview_server.py"

Start-Process -FilePath $pythonw `
  -ArgumentList @($server) `
  -WorkingDirectory $root `
  -WindowStyle Hidden

Write-Host "Preview server started:"
Write-Host "http://127.0.0.1:8000/"
