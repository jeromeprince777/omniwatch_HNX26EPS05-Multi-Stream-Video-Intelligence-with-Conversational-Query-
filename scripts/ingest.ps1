# Ingest multi-camera footage from manifest.yaml (PowerShell)
param (
    [Parameter(Position=0)]
    [string]$Manifest = "manifest.yaml",
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
Write-Host "[OmniWatch] Ingesting footage from: $Manifest with extra args: $ExtraArgs" -ForegroundColor Cyan
python -m omniwatch.ingest.pipeline $Manifest @ExtraArgs
