# Refresh the copies in this folder from their sources in docs/. Run from anywhere:
#   powershell -File frontend_handoff\sync.ps1
$root = Split-Path -Parent $PSScriptRoot
$files = "API_CONTRACT.md", "openapi.json", "BENCHMARKS.md", "weakness.md"
foreach ($f in $files) {
    Copy-Item -Path (Join-Path $root "docs\$f") -Destination (Join-Path $PSScriptRoot $f) -Force
    Write-Host "synced $f"
}
