# OmniWatch -- Sync Screenshots and Push to GitHub (PowerShell)
$ErrorActionPreference = "Continue"

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "  OmniWatch (HNX26EPS05) -- Sync Screenshots & Push to GitHub" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

# 1. Sync Screenshots
Write-Host "`n[1/3] Copying UI screenshots to docs\screenshots\ ..." -ForegroundColor Yellow
$Python = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }
& $Python scripts\sync_screenshots.py

# 2. Check Git Repo
Write-Host "`n[2/3] Configuring Git Repository ..." -ForegroundColor Yellow
if (-not (Test-Path ".git")) {
    git init
    git branch -M main
}

git remote remove origin 2>$null
git remote add origin "https://github.com/jeromeprince777/omniwatch_HNX26EPS05-Multi-Stream-Video-Intelligence-with-Conversational-Query-.git"

# 3. Commit & Push
Write-Host "`n[3/3] Committing and Pushing to GitHub ..." -ForegroundColor Yellow
git add .
git status --short
git commit -m "feat: complete OmniWatch platform with architectural flowcharts, diagrams, and UI screenshots"

Write-Host "`nPushing to GitHub (if login is required, your browser or terminal will prompt you)..." -ForegroundColor Green
git push -u origin main

Write-Host "`n=====================================================================" -ForegroundColor Cyan
Write-Host "  Done! Check your repository on GitHub." -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
