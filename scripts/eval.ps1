# Run automated evaluation and ablation benchmark (PowerShell)
param (
    [string]$Queries = "eval/queries/dev.jsonl",
    [string]$OutDir = "eval/results"
)

$ErrorActionPreference = "Stop"
Write-Host "[OmniWatch] Running evaluation harness on $Queries -> $OutDir" -ForegroundColor Cyan
python -m eval.run_eval --queries "$Queries" --out "$OutDir"
