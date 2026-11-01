@echo off
chcp 65001 >nul
title 安装 AShare Quant Windows 每日定时计划任务
echo ============================================================
echo ⚙️ 正在计算对应北京时间 15:35 的本地时间并注册计划任务...
echo ============================================================

cd /d "%~dp0"
set SCRIPT_PATH=%~dp0run_daily.bat

:: 通过 Python 动态计算当前本地时区对应北京时间 15:35 的具体时间点
for /f "tokens=*" %%i in ('.venv\Scripts\python.exe -c "from datetime import datetime, timezone, timedelta; bj_tz=timezone(timedelta(hours=8)); local_now=datetime.now().astimezone(); bj_dt=datetime.now(bj_tz).replace(hour=15, minute=35, second=0, microsecond=0); print(bj_dt.astimezone(local_now.tzinfo).strftime('%%H:%%M'))"') do set LOCAL_TIME=%%i

echo.
echo 目标对应北京时间: 15:35 (A股盘后清算就绪)
echo 当前电脑本地时间: %LOCAL_TIME% (自动适配加拿大时区)
echo 脚本目标路径: %SCRIPT_PATH%
echo 正在向 Windows 任务计划程序注册 [AShare_Quant_Daily_Auto] ...
echo.

schtasks /create /tn "AShare_Quant_Daily_Auto" /tr "\"%SCRIPT_PATH%\"" /sc weekly /d MON,TUE,WED,THU,FRI /st %LOCAL_TIME% /f

if %ERRORLEVEL% equ 0 (
    echo.
    echo ============================================================
    echo ✅ 成功注册 Windows 任务计划！
    echo 任务将在每周一至周五 本地时间 %LOCAL_TIME% 自动在后台静默运行！
    echo 自动抓取当日行情、计算因子、并在模拟盘中自动完成买入与卖出！
    echo 早上醒来直接打开网页即可查看最新的模拟盘收益与持仓！
    echo ============================================================
) else (
    echo.
    echo ⚠️ 注册计划任务需要管理员权限。如遇报错，请右键选择 [以管理员身份运行] 本文件。
)

pause
