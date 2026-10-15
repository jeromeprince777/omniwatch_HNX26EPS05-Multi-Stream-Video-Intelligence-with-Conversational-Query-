@echo off
title OmniWatch Live Query Tracking
cd /d "%~dp0"
set PYTHON=.venv\Scripts\python.exe
if not exist "%PYTHON%" set PYTHON=python

if "%1"=="" goto default_run

"%PYTHON%" omni_live.py %*
goto end

:default_run
echo.
echo ============================================================
echo  OmniWatch LIVE  --  GPU-accelerated real-time tracking
echo  http://127.0.0.1:8100
echo ============================================================
echo.
"%PYTHON%" omni_live.py ^
    --src Yokohama=data/public_cctv/yokohama_cctv.mp4 ^
    --src Tours=data/public_cctv/surveillance_tours.mp4 ^
    --loop ^
    --query "person carrying a bag" ^
    --port 8100

:end
