@echo off
cd /d "%~dp0"
title AShare Quant Task Scheduler Setup
echo ============================================================
echo [AShare Quant] Installing Daily Windows Scheduled Task...
echo ============================================================
.venv\Scripts\python.exe setup_task.py
echo.
pause
