@echo off
cd /d "%~dp0"
title AShare Quant Web Console
echo ============================================================
echo [AShare Quant] Starting Web Console at http://127.0.0.1:5000 ...
echo ============================================================
start "" "http://127.0.0.1:5000"
.venv\Scripts\python.exe src\web_app.py
pause
