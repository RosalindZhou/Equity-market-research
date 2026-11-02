@echo off
cd /d "%~dp0"
title AShare Quant Daily Pipeline
echo ============================================================
echo [AShare Quant] Running Daily Pipeline...
echo ============================================================
.venv\Scripts\python.exe daily_runner.py
echo.
pause
