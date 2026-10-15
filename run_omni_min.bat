@echo off
title OmniWatch Minimal Video Search Tool
cd /d "%~dp0"
set PYTHON=.venv\Scripts\python.exe
if not exist "%PYTHON%" set PYTHON=python

if "%1"=="" goto usage

"%PYTHON%" omni_min.py %*
goto end

:usage
echo Usage:
echo   run_omni_min.bat index --cam Yokohama --video data/public_cctv/yokohama_cctv.mp4 --start "2026-10-09 09:00:00"
echo   run_omni_min.bat index --cam Tours --video data/public_cctv/surveillance_tours.mp4 --start "2026-10-09 09:05:00"
echo   run_omni_min.bat search "person carrying a bag"
echo   run_omni_min.bat search "red car"
echo   run_omni_min.bat sheet
echo.
pause

:end
