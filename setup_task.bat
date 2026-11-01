@echo off
chcp 65001 >nul
title 安装 AShare Quant Windows 每日定时计划任务
echo ============================================================
echo ⚙️ 正在向 Windows 任务计划程序注册自动运行任务...
echo ============================================================

set SCRIPT_PATH=%~dp0run_daily.bat

echo 脚本目标路径: %SCRIPT_PATH%
echo 正在注册名为 [AShare_Quant_Daily_Auto] 的任务...
echo.

schtasks /create /tn "AShare_Quant_Daily_Auto" /tr "\"%SCRIPT_PATH%\"" /sc weekly /d MON,TUE,WED,THU,FRI /st 15:35 /f

if %ERRORLEVEL% equ 0 (
    echo.
    echo ============================================================
    echo ✅ 成功注册 Windows 任务计划！
    echo 任务将在每周一至周五自动执行，即使电脑无人在旁也会按时跑！
    echo 如需修改执行时间，请在 Windows [任务计划程序] 中调整。
    echo ============================================================
) else (
    echo.
    echo ⚠️ 注册计划任务需要管理员权限。如遇报错，请右键选择 [以管理员身份运行] 本文件。
)

pause
