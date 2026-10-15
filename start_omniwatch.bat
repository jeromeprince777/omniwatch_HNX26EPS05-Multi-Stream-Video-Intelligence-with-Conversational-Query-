@echo off
title OmniWatch Tactical Surveillance Server
cd /d "%~dp0"
echo ========================================================
echo   OmniWatch -- Multi-Stream Video Intelligence HUD
echo ========================================================
echo Starting server on http://127.0.0.1:8000 ...
echo Press Ctrl+C to terminate the server.
echo ========================================================
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m uvicorn omniwatch.api.main:app --host 127.0.0.1 --port 8000 --reload
) else (
    python -m uvicorn omniwatch.api.main:app --host 127.0.0.1 --port 8000 --reload
)
pause
