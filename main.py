"""
AShare Quant System - 一键全流程总控调度入口 (main.py)
===================================================
支持每日定时自动化流水线与命令行独立模块调度：
- 默认 / --daily: 执行每日盘后完整流水线 (行情更新 -> 因子计算 -> 信号预测 -> 微信推送)
- --train: 执行 LightGBM 模型离线重新训练与 IC 评估
- --predict: 仅执行今日最新特征预测与微信通知
- --fetch: 仅执行行情抓取更新
"""

import sys
import argparse
import logging
from pathlib import Path

# 将项目根目录添加至系统路径
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.data_fetcher import DataFetcher
from src.factor_calc import FactorCalculator
from src.model_train import ModelTrainer
from src.predictor import Predictor
from src.notifier import Notifier
from src.backtest import Backtester
from src.paper_trader import PaperTrader
from src.web_app import start_server
import src.config as cfg


def run_daily_pipeline():
    """执行每日盘后自动化流水线"""
    logging.info("==================================================")
    logging.info("🚀 启动 AShare_Quant_System 每日盘后自动化流水线")
    logging.info("==================================================")

    # 1. 增量更新行情数据
    logging.info("\n>>> 步骤 1/3: 更新最新市场行情数据...")
    fetcher = DataFetcher()
    fetcher.fetch_and_save_all()

    # 2. 计算因子并更新缓存
    logging.info("\n>>> 步骤 2/3: 极速计算多因子与横截面百分位排名...")
    calculator = FactorCalculator()
    calculator.run_pipeline()

    # 3. 检查模型并生成明日预测信号
    model_file = cfg.MODEL_PATH
    if not model_file.exists():
        logging.warning(f"未检测到生产模型权重文件 [{model_file}]，自动触发首次离线训练...")
        trainer = ModelTrainer()
        trainer.run_pipeline()

    # 4. 信号推断与微信推送
    logging.info("\n>>> 步骤 3/3: 执行模型预测并推送到手机端...")
    predictor = Predictor(budget=cfg.TOTAL_BUDGET, top_n=cfg.TOP_N_PICKS)
    predictor.run_pipeline(push_notification=True)

    logging.info("\n==================================================")
    logging.info("🎉 每日盘后量化流水线圆满完成！祝投资顺利！")
    logging.info("==================================================")


def main():
    parser = argparse.ArgumentParser(description="A股中低频量化研究系统调度引擎")
    parser.add_argument("--fetch", action="store_true", help="仅执行数据抓取与入库")
    parser.add_argument("--calc", action="store_true", help="仅执行多因子计算与特征衍生")
    parser.add_argument("--train", action="store_true", help="执行 LightGBM 模型离线重新训练")
    parser.add_argument("--predict", action="store_true", help="仅执行最新实盘预测与微信推送")
    parser.add_argument("--test-notify", action="store_true", help="测试微信消息推送通道")
    parser.add_argument("--backtest", action="store_true", help="执行历史区间回测与回滚收益分析")
    parser.add_argument("--start", type=str, default="2024-01-01", help="回测起点日期 (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2026-10-22", help="回测终点日期 (YYYY-MM-DD)")
    parser.add_argument("--no_stop", action="store_true", help="回测时关闭盘中止损止盈盯盘逻辑 (纯5日到期轮动)")
    parser.add_argument("--paper", action="store_true", help="执行实盘模拟盘操作")
    parser.add_argument("--paper-action", type=str, choices=["init", "status", "step"], default="status", help="模拟盘操作: init, status, step")
    parser.add_argument("--capital", type=float, default=cfg.TOTAL_BUDGET, help="模拟盘资金 (默认 100000)")
    parser.add_argument("--web", action="store_true", help="启动交互式量化交易 Web 控制台")
    parser.add_argument("--port", type=int, default=5000, help="Web 控制台端口 (默认 5000)")

    args = parser.parse_args()

    # 配置根日志级别
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.web:
        start_server(port=args.port)
    elif args.fetch:
        DataFetcher().fetch_and_save_all()
    elif args.calc:
        FactorCalculator().run_pipeline()
    elif args.train:
        # 若特征缓存不存在，自动触发因子计算
        if not cfg.FEATURES_PARQUET_PATH.exists():
            logging.info("未检测到特征矩阵缓存，正在自动先执行多因子计算与特征衍生...")
            FactorCalculator().run_pipeline()
        ModelTrainer().run_pipeline()
    elif args.predict:
        # 若模型不存在，自动触发训练
        if not cfg.MODEL_PATH.exists():
            logging.info("未检测到生产模型文件，正在自动触发离线训练...")
            if not cfg.FEATURES_PARQUET_PATH.exists():
                FactorCalculator().run_pipeline()
            ModelTrainer().run_pipeline()
        Predictor(budget=cfg.TOTAL_BUDGET, top_n=cfg.TOP_N_PICKS).run_pipeline(push_notification=True)
    elif args.backtest:
        Backtester(
            start_date=args.start,
            end_date=args.end,
            use_stop_loss=not args.no_stop,
        ).run()
    elif args.paper:
        trader = PaperTrader(default_capital=args.capital)
        if args.paper_action == "init":
            trader.init_account(args.capital)
        elif args.paper_action == "step":
            trader.step()
        else:
            trader.print_status()
    elif args.test_notify:
        Notifier().broadcast(
            title="【系统测试】AShare Quant 微信推送通道验证",
            content="这是一条测试消息，说明您的微信推送配置完全正常！\n- 时间: 2026-11-01\n- 状态: 运行正常",
        )
    else:
        # 默认执行每日全量自动化流水线
        run_daily_pipeline()


if __name__ == "__main__":
    main()
