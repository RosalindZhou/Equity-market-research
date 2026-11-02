"""
AShare Quant System - 每日盘后自动化调度主引擎 (daily_runner.py)
============================================================
专为海外与自动化无人值守设计：
1. 盘后自动更新中证 500 最新行情数据入库
2. 自动增量计算多因子特征与截面百分位
3. LightGBM 模型预测明日 Top-5 股票目标
4. 模拟盘 (PaperTrader) 自动扫描持仓：
   - 满 5 交易日自动卖出轮动
   - 触发 -5% 硬止损或 +10% 止盈自动卖出
   - 释放资金后自动买入今日最新推荐 Top-5
5. 自动渲染最新的离线 HTML 量化研报 (reports/latest.html)
6. 全程写入 logs/daily_runner.log，即使无人盯盘也能完整追溯
"""

import os
import sys
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

# 将项目根目录添加至系统路径
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 控制台 UTF-8 支持
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 确保 logs 目录存在
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "daily_runner.log"

# 配置双重日志输出 (控制台 + 文件)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8", mode="a"),
    ],
)

logger = logging.getLogger("DailyRunner")

from src.data_fetcher import DataFetcher
from src.factor_calc import FactorCalculator
from src.model_train import ModelTrainer
from src.predictor import Predictor
from src.paper_trader import PaperTrader
from src.report_generator import ReportGenerator
import src.config as cfg


def get_dual_timezone_time() -> str:
    """获取北京与本地双时区时间字符串"""
    bj_tz = timezone(timedelta(hours=8))
    local_tz = datetime.now().astimezone().tzinfo
    now_utc = datetime.now(timezone.utc)
    bj_str = now_utc.astimezone(bj_tz).strftime("%Y-%m-%d %H:%M:%S")
    local_str = now_utc.astimezone(local_tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    return f"北京时间: {bj_str} | 本地时间: {local_str}"


def run_pipeline() -> bool:
    """执行完整的无人值守每日量化交易流水线"""
    logger.info("=" * 65)
    logger.info("🚀 [AShare Quant] 每日无人值守全自动流水线启动")
    logger.info(get_dual_timezone_time())
    logger.info("=" * 65)

    start_time = datetime.now()

    try:
        # ---------------------------------------------------------
        # 步骤 1: 增量拉取当日收盘行情
        # ---------------------------------------------------------
        logger.info("\n>>> [步骤 1/4] 正在增量更新中证 500 成分股最新收盘行情...")
        fetcher = DataFetcher()
        try:
            fetcher.fetch_and_save_all()
            logger.info("✅ 行情数据增量更新完毕")
        except Exception as e:
            logger.warning(f"⚠️ 行情抓取部分告警或受限 (如未开盘或网络波动): {e}")

        # ---------------------------------------------------------
        # 步骤 2: 计算多因子特征与横截面百分位
        # ---------------------------------------------------------
        logger.info("\n>>> [步骤 2/4] 计算量化多因子特征并衍生截面百分位...")
        calculator = FactorCalculator()
        calculator.run_pipeline()
        logger.info("✅ 多因子特征计算与缓存落盘完成")

        # 检查模型权重
        if not cfg.MODEL_PATH.exists():
            logger.info("⚠️ 未检测到生产模型，正在自动触发离线训练...")
            trainer = ModelTrainer()
            trainer.run_pipeline()

        # ---------------------------------------------------------
        # 步骤 3: 运行模型推断并生成最新 Top-5 股票推荐
        # ---------------------------------------------------------
        logger.info("\n>>> [步骤 3/4] 运行 LightGBM 推断明日 Alpha 选股目标...")
        predictor = Predictor(budget=cfg.TOTAL_BUDGET, top_n=cfg.TOP_N_PICKS)
        pred_df = predictor.run_pipeline(push_notification=True)
        logger.info(f"✅ 模型预测完成，已生成推荐股票 {len(pred_df)} 只并入库")

        # ---------------------------------------------------------
        # 步骤 4: 模拟盘自动轮动与买卖下单
        # ---------------------------------------------------------
        logger.info("\n>>> [步骤 4/4] 模拟盘 (PaperTrader) 自动执行清算与轮动交易...")
        trader = PaperTrader(default_capital=cfg.TOTAL_BUDGET)
        trader.step()
        logger.info("✅ 模拟盘自动卖出与买入换仓全部执行完毕")

        # ---------------------------------------------------------
        # 步骤 5: 渲染离线静态 HTML 研报
        # ---------------------------------------------------------
        logger.info("\n>>> 正在生成最新静态 HTML 离线研报...")
        rg = ReportGenerator()
        report_file = rg.generate_html_report()
        logger.info(f"✅ 静态 HTML 研报已成功生成: {report_file}")

        elapsed = (datetime.now() - start_time).total_seconds()
        summary = trader.get_account_summary()
        acc = summary["account"]

        logger.info("\n" + "=" * 65)
        logger.info(f"🎉 每日全自动流水线执行成功！耗时: {elapsed:.1f} 秒")
        logger.info(f"💰 模拟盘当前总资产: ¥{acc['total_equity']:,.2f} (可用现金: ¥{acc['cash']:,.2f})")
        logger.info(f"📈 累计总收益率: {acc['total_return_pct']:.2f}% | 当前持仓数: {len(summary['positions'])} 只")
        logger.info("🌐 打开 Web 控制台即可直接查看最新仪表盘与持仓明细！")
        logger.info("=" * 65)
        return True

    except Exception as e:
        logger.error(f"❌ 自动化流水线执行过程中发生异常: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)
