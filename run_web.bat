@echo off
chcp 65001 >nul
title AShare Quant Web 控制台
echo ============================================================
echo 🚀 正在启动 AShare Quant 量化交互式 Web 控制台...
echo ============================================================

start "" http://127.0.0.1:5000
.venv\Scripts\python.exe src\web_app.py
pause
