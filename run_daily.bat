@echo off
chcp 65001 >nul
title AShare Quant 每日自动运行脚本
echo ============================================================
echo ⏰ 正在执行 AShare Quant 每日全量流水线...
echo ============================================================

cd /d "%~dp0"
.venv\Scripts\python.exe main.py --daily
.venv\Scripts\python.exe main.py --paper --paper-action step
.venv\Scripts\python.exe src\report_generator.py

echo ============================================================
echo 🎉 每日任务与静态网页快照已全部生成！
echo ============================================================
