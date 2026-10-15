@echo off
title OmniWatch -- Sync Screenshots and Push to GitHub
cd /d "%~dp0"

echo =====================================================================
echo   OmniWatch (HNX26EPS05) -- Sync Screenshots & Push to GitHub
echo =====================================================================
echo.

echo [1/3] Copying UI screenshots to docs\screenshots\ ...
if not exist "docs\screenshots" mkdir "docs\screenshots"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" scripts\sync_screenshots.py
) else (
    python scripts\sync_screenshots.py
)

echo.
echo [2/3] Checking Git Repository Configuration ...
if not exist ".git" (
    echo Initializing new Git repository...
    git init
)

git config user.name "jeromeprince777"
git config user.email "jeromeprince777@users.noreply.github.com"
git branch -M main

echo Setting remote origin to https://github.com/jeromeprince777/omniwatch_HNX26EPS05-Multi-Stream-Video-Intelligence-with-Conversational-Query- ...
git remote remove origin >nul 2>&1
git remote add origin https://github.com/jeromeprince777/omniwatch_HNX26EPS05-Multi-Stream-Video-Intelligence-with-Conversational-Query-.git

echo.
echo [3/3] Staging, Committing, and Pushing to GitHub ...
git add .
git commit -m "feat: complete OmniWatch platform with architectural flowcharts, diagrams, and UI screenshots"

echo.
echo Pushing branch 'main' to GitHub...
git push -u origin main

echo.
echo =====================================================================
echo   Operation Complete! Check GitHub repository:
echo   https://github.com/jeromeprince777/omniwatch_HNX26EPS05-Multi-Stream-Video-Intelligence-with-Conversational-Query-
echo =====================================================================
