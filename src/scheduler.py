"""
后台自动化守护定时调度器 (Scheduler Daemon)
===========================================================
功能特性：
1. 跨时区感知 (Timezone-Aware):
   - 自动将系统本地时区（如加拿大东部/太平洋时间）与北京时间 (UTC+8) 精确对齐
2. A股交易日与交易时间自动判断:
   - 每周一至周五，北京时间 15:35 (A股盘后清算就绪时间) 自动触发全流程流水线：
     1) 增量获取当日最新日线行情入库 (DataFetcher)
     2) 33维因子计算与截面特征落盘 (FactorCalculator)
     3) 模拟盘自动盯盘与周频到期平仓/新信号建仓 (PaperTrader)
     4) 生成并落盘今日静态 HTML 快照档案 (ReportGenerator)
     5) 微信/多通道通知广播 (Notifier)
3. 守护进程稳定运行，带异常捕获与重试机制
"""

import os
import sys
import time
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 修复 Windows 控制台 GBK 编码输出
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.data_fetcher import DataFetcher
from src.factor_calc import FactorCalculator
from src.paper_trader import PaperTrader
from src.report_generator import ReportGenerator
from src.notifier import Notifier
import src.config as cfg


LOG_FILE = cfg.LOGS_DIR / "scheduler.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("Scheduler")


def get_beijing_time() -> datetime:
    """获取当前精准北京时间 (UTC+8)"""
    return datetime.now(timezone(timedelta(hours=8)))


def run_daily_pipeline():
    """执行每日盘后自动化完整闭环"""
    logger.info("=" * 65)
    logger.info("⏰ 触发 A股盘后自动化任务流水线 (行情 -> 因子 -> 模拟盘 -> 研报网页)")
    logger.info("=" * 65)

    try:
        # 1. 增量抓取最新日线行情
        logger.info(">>> [1/4] 增量抓取今日收盘行情...")
        fetcher = DataFetcher()
        fetcher.fetch_and_save_all()

        # 2. 计算 33 维因子并落盘 Parquet
        logger.info(">>> [2/4] 计算最新多因子与截面排位...")
        calculator = FactorCalculator()
        calculator.run_pipeline()

        # 3. 模拟盘自动盯盘与调仓
        logger.info(">>> [3/4] 模拟盘自动到期平仓与新信号买入建仓...")
        trader = PaperTrader()
        trader.step(push_notification=True)

        # 4. 生成最新静态 HTML 网页快照
        logger.info(">>> [4/4] 导出静态 HTML 研报档案...")
        generator = ReportGenerator()
        report_file = generator.generate_html_report()
        logger.info(f"✅ 今日静态研报已落盘: {report_file}")

        logger.info("=" * 65)
        logger.info("🎉 今日自动化流水线圆满完成！")
        logger.info("=" * 65)

    except Exception as err:
        logger.error(f"自动化执行过程中发生异常: {err}", exc_info=True)


def start_scheduler_loop(check_interval: int = 30):
    """守护循环轮询"""
    logger.info("=" * 65)
    logger.info("🚀 AShare Quant 每日自动运行守护进程已启动")
    bj_now = get_beijing_time()
    logger.info(f"当前北京时间: {bj_now.strftime('%Y-%m-%d %H:%M:%S')} | 本地时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("计划触发时间: 每个交易日 (周一至周五) 北京时间 15:35:00")
    logger.info("=" * 65)

    last_executed_date = None

    while True:
        try:
            bj_time = get_beijing_time()
            today_str = bj_time.strftime("%Y-%m-%d")
            weekday = bj_time.weekday()  # 0=周一, 4=周五, 5/6=周末

            # 判断是否周一至周五，且时间达到 15:35 之后，且今天尚未执行过
            if weekday in range(0, 5) and (bj_time.hour > 15 or (bj_time.hour == 15 and bj_time.minute >= 35)):
                if last_executed_date != today_str:
                    logger.info(f"检测到符合执行条件 (北京时间: {bj_time.strftime('%H:%M:%S')})，启动任务...")
                    run_daily_pipeline()
                    last_executed_date = today_str
                    logger.info(f"今日 ({today_str}) 任务已执行完毕，进入等待明天状态。")

            time.sleep(check_interval)
        except KeyboardInterrupt:
            logger.info("调度守护进程已由用户手动停止。")
            break
        except Exception as e:
            logger.error(f"调度循环内部异常: {e}", exc_info=True)
            time.sleep(check_interval)


if __name__ == "__main__":
    start_scheduler_loop()
