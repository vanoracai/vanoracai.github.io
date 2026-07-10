$connections = netstat -ano | Select-String ":8000" | ForEach-Object {
  ($_ -split "\s+")[-1]
} | Where-Object {
  $_ -match "^\d+$" -and $_ -ne "0"
} | Sort-Object -Unique

foreach ($pidValue in $connections) {
  Stop-Process -Id ([int]$pidValue) -Force -ErrorAction SilentlyContinue
}

Write-Host "Stopped preview processes on port 8000."
