"""
AShare Quant System - Windows 每日定时计划任务安装与配置器 (setup_task.py)
======================================================================
功能：
1. 自动根据当前操作系统的本地时区（例如加拿大东部时间 UTC-5），精准换算北京时间 15:35（A股盘后清算就绪）对应的本地运行时间。
2. 使用原生 Windows PowerShell API (Register-ScheduledTask) 注册计划任务：
   - 彻底解决 Windows CMD 批处理对于中文路径和时区计算的编码乱码问题。
   - 允许笔记本在电池供电下运行 (AllowStartIfOnBatteries)。
   - 若电脑休眠或关机，开机唤醒后自动补跑 (StartWhenAvailable)。
   - 每周一至周五定时在后台静默拉取行情、执行选股、模拟盘自动买卖并生成网页报表。
"""

import sys
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path

# 控制台 UTF-8 支持
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_EXE = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
RUNNER_SCRIPT = PROJECT_ROOT / "daily_runner.py"
TASK_NAME = "AShare_Quant_Daily_Auto"


def calculate_schedule_time():
    """计算对应北京时间 15:35 的本地时间"""
    bj_tz = timezone(timedelta(hours=8))
    local_tz = datetime.now().astimezone().tzinfo

    # 构造北京时间 15:35:00
    now_bj = datetime.now(bj_tz)
    target_bj = now_bj.replace(hour=15, minute=35, second=0, microsecond=0)

    # 转换至本地时区
    target_local = target_bj.astimezone(local_tz)
    return target_bj, target_local


def register_task() -> bool:
    print("=" * 65)
    print("⚙️  AShare Quant Windows 每日定时任务配置程序")
    print("=" * 65)

    if not PYTHON_EXE.exists():
        print(f"❌ 错误: 未找到虚拟环境 Python: {PYTHON_EXE}")
        print("请确保已安装虚拟环境 .venv！")
        return False

    if not RUNNER_SCRIPT.exists():
        print(f"❌ 错误: 未找到自动化脚本: {RUNNER_SCRIPT}")
        return False

    target_bj, target_local = calculate_schedule_time()
    local_time_str = target_local.strftime("%H:%M")
    local_tz_name = target_local.tzname() or "本地时区"

    print(f"📍 目标 A 股盘后时间 : 北京时间 15:35:00 (A股收盘结算完成)")
    print(f"📍 适配当前电脑时间 : {local_tz_name} {local_time_str} (周一至周五)")
    print(f"📁 执行工作目录     : {PROJECT_ROOT}")
    print(f"🐍 Python 解析器    : {PYTHON_EXE}")
    print(f"📜 自动运行主脚本   : {RUNNER_SCRIPT}")
    print("-" * 65)
    print("正在向 Windows 任务计划程序注册后台定时任务...")

    # 构建安全可靠的 PowerShell 注册脚本
    ps_script = f"""
    $OutputEncoding = [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $action = New-ScheduledTaskAction -Execute '{PYTHON_EXE.as_posix()}' -Argument '"{RUNNER_SCRIPT.as_posix()}"' -WorkingDirectory '{PROJECT_ROOT.as_posix()}';
    $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At '{local_time_str}';
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable;
    Register-ScheduledTask -TaskName "{TASK_NAME}" -Action $action -Trigger $trigger -Settings $settings -Force
    """

    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        if res.returncode == 0:
            print("\n" + "=" * 65)
            print("🎉 恭喜！Windows 每日全自动定时任务注册成功！")
            print("=" * 65)
            print(f"⏰ 运行时间: 每周一至周五 本地时间 {local_time_str} (即北京时间 15:35)")
            print("🚀 自动化任务内容:")
            print("   1. 自动增量拉取中证 500 最新行情")
            print("   2. 自动计算 101Alpha 等多因子截面排名")
            print("   3. 自动运行 LightGBM 预测明日 Top-5 股票")
            print("   4. 模拟盘自动到期轮动平仓，并自动下单买入 Top-5 标的")
            print("   5. 自动渲染最新的静态 HTML 离线报告与网页控制台数据")
            print("\n💡 您人在加拿大，完全无需熬夜盯盘：")
            print(f"   后台会在凌晨 {local_time_str} 自动完成一切操作，醒来直接双击 run_web.bat 打开网页查看！")
            print("=" * 65)
            return True
        else:
            print(f"\n❌ 注册失败: {res.stderr or res.stdout}")
            print("💡 提示: Windows 计划任务可能需要管理员权限。请尝试右键点击批处理文件，选择 [以管理员身份运行]！")
            return False

    except Exception as e:
        print(f"\n❌ 调用 PowerShell 注册失败: {e}")
        return False


if __name__ == "__main__":
    success = register_task()
    sys.exit(0 if success else 1)
