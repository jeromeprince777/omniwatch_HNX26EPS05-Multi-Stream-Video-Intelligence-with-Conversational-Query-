# Start OmniWatch FastAPI and Web UI server (PowerShell)
param (
    [string]$HostIP = "127.0.0.1",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
Write-Host "[OmniWatch] Starting server on http://$HostIP`:$Port" -ForegroundColor Cyan
python -m uvicorn omniwatch.api.main:app --host $HostIP --port $Port --reload
