"""
量化策略回测与历史区间回滚分析引擎 (Backtester) - 机构级实盘模拟版
======================================================================
功能特性：
1. 任意历史区间回滚 (Custom Time Window):
   - 支持指定 --start YYYY-MM-DD 和 --end YYYY-MM-DD，任意调取历史任意时间段评估策略表现
2. 严密防范未来函数 (No Lookahead Bias):
   - T 日盘后出信号 -> T+1 日开盘价执行买入 -> T+6 日开盘价执行卖出 (持仓整 5 个交易日)
   - 包含真实摩擦成本: 券商佣金 (万三) + 卖出印花税 (千分之0.5) + 双向滑点 (千一)
3. 动态风控与盯盘逻辑:
   - 模拟持仓期内逐日最低价判断硬止损 (-5%) 与最高价止盈 (+8%)
4. 机构级全景指标报告:
   - 策略绝对收益率 vs 中证500基准收益率 (Alpha)
   - 年化收益率、年化波动率、夏普比率 (Sharpe)、卡玛比率 (Calmar)
   - 最大回撤 (Max Drawdown) 及起止发生区间
   - 单笔交易胜率 (Trade Win Rate)、盈亏比 (Profit/Loss Ratio)、平均盈利/亏损
"""

import os
import sys
import argparse
import sqlite3
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 修复 Windows 控制台编码
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb

import src.config as cfg


def setup_logger(name: str = "Backtester") -> logging.Logger:
    """配置日志"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


class Backtester:
    """
    机构级量化回测与区间回滚分析引擎
    """

    def __init__(
        self,
        start_date: str = "2024-01-01",
        end_date: str = "2026-10-22",
        top_n: int = cfg.TOP_N_PICKS,
        initial_cash: float = 100000.0,
        hold_days: int = cfg.HOLD_DAYS,
        stop_loss_pct: float = cfg.STOP_LOSS_PCT,
        take_profit_pct: float = cfg.TAKE_PROFIT_PCT,
        use_stop_loss: bool = True,
        commission: float = cfg.TRANSACTION_FEE,
        stamp_duty: float = cfg.STAMP_DUTY,
        slippage: float = cfg.SLIPPAGE,
    ):
        self.logger = setup_logger()
        self.start_date = start_date
        self.end_date = end_date
        self.top_n = top_n
        self.initial_cash = initial_cash
        self.hold_days = hold_days
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.use_stop_loss = use_stop_loss
        self.commission = commission
        self.stamp_duty = stamp_duty
        self.slippage = slippage

        self.model_path = cfg.MODEL_PATH
        self.features_path = cfg.FEATURES_PARQUET_PATH
        self.db_path = cfg.DB_PATH

    def load_model(self) -> lgb.Booster:
        """加载训练好的 LightGBM 模型"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}，请先训练模型！")
        with open(self.model_path, "r", encoding="utf-8") as f:
            return lgb.Booster(model_str=f.read())

    def run(self) -> Dict[str, Any]:
        """执行端到端历史回测"""
        self.logger.info("=" * 65)
        self.logger.info(f"启动量化策略历史区间回测与回滚分析 [{self.start_date} ~ {self.end_date}]")
        self.logger.info("=" * 65)

        # 1. 加载模型与特征数据
        model = self.load_model()
        self.logger.info(f"正在读取多因子特征矩阵 (区间: {self.start_date} ~ {self.end_date})...")
        
        feature_df = pl.read_parquet(self.features_path).filter(
            (pl.col("date") >= self.start_date) & (pl.col("date") <= self.end_date)
        )
        if feature_df.is_empty():
            raise ValueError(f"指定区间 [{self.start_date} ~ {self.end_date}] 内无有效特征数据！")

        # 2. 批量预测模型分数
        self.logger.info(f"计算指定时间段样本的预测评分 (共 {feature_df.shape[0]} 条样本)...")
        X = feature_df.select(cfg.RANK_FEATURES).to_numpy()
        pred_scores = model.predict(X)
        scored_df = feature_df.select(["date", "symbol"]).with_columns(
            pl.Series("pred_score", pred_scores)
        )

        # 3. 读取该区间及后续持仓日的原始行情 OHLCV
        self.logger.info("从数据库加载对应区间的真实成交价格 (Open/High/Low/Close)...")
        conn = sqlite3.connect(self.db_path.as_posix())
        try:
            # 读取 start_date 之后的所有行情用于匹配 T+1 开盘买入与 T+6 开盘卖出
            price_query = """
            SELECT symbol, date, open, high, low, close
            FROM daily_bars
            WHERE date >= ?
            ORDER BY symbol, date ASC
            """
            prices_df = pl.read_database(price_query, connection=conn, execute_options={"parameters": [self.start_date]})
        finally:
            conn.close()

        # 4. 构建极速字典映射与交易日序列
        all_dates = sorted(prices_df["date"].unique().to_list())
        date_to_idx = {d: i for i, d in enumerate(all_dates)}
        idx_to_date = {i: d for i, d in enumerate(all_dates)}

        # 将行情转化为按 (symbol, date) 索引的极速查询表
        price_dict = {}
        for row in prices_df.iter_rows(named=True):
            price_dict[(row["symbol"], row["date"])] = row

        # 5. 确定周频轮动决策日 (每 hold_days 个交易日作为决策周期)
        scored_dates = sorted(scored_df["date"].unique().to_list())
        # 每隔 hold_days 作为一个调仓决策点
        rebalance_signal_dates = scored_dates[0::self.hold_days]
        self.logger.info(f"区间内涵盖 {len(scored_dates)} 个交易日，规划执行 {len(rebalance_signal_dates)} 轮周频调仓")

        # 6. 逐轮模拟真实买卖交易与资产核算
        trades = []
        portfolio_history = []
        benchmark_history = []

        current_cash = self.initial_cash
        benchmark_cash = self.initial_cash
        
        # 记录初始点
        portfolio_history.append({"date": self.start_date, "equity": current_cash, "return_pct": 0.0})
        benchmark_history.append({"date": self.start_date, "equity": benchmark_cash, "return_pct": 0.0})

        for cycle_idx, signal_date in enumerate(rebalance_signal_dates):
            if signal_date not in date_to_idx:
                continue
            curr_idx = date_to_idx[signal_date]
            
            # T+1 日开盘买入
            buy_date_idx = curr_idx + 1
            if buy_date_idx >= len(all_dates):
                break
            buy_date = idx_to_date[buy_date_idx]

            # T+6 日开盘卖出
            sell_date_idx = buy_date_idx + self.hold_days
            if sell_date_idx >= len(all_dates):
                # 如果没有足够的持仓日，使用区间最后一个可交易日收盘卖出
                sell_date_idx = len(all_dates) - 1
            sell_date = idx_to_date[sell_date_idx]

            # 筛选当期 Top-N 股票
            day_candidates = scored_df.filter(pl.col("date") == signal_date).sort(
                "pred_score", descending=True
            ).head(self.top_n).to_dicts()

            if not day_candidates:
                continue

            # 每只股票等权分配可用本金
            alloc_cash_per_stock = current_cash / len(day_candidates)
            cycle_trades = []
            cycle_returns = []

            for cand in day_candidates:
                code = cand["symbol"]
                buy_bar = price_dict.get((code, buy_date))
                if not buy_bar or buy_bar["open"] <= 0:
                    continue

                raw_buy_price = float(buy_bar["open"])
                # 买入成本：滑点 + 佣金
                actual_buy_price = raw_buy_price * (1.0 + self.slippage)
                buy_fee_rate = self.commission
                
                # 模拟持仓期间的每日价格变动与止损止盈
                actual_sell_price = None
                exit_date = sell_date
                exit_reason = "NORMAL_EXPIRE"  # 正常到期卖出

                # 检查持仓日内的止损止盈
                if self.use_stop_loss:
                    for d_i in range(buy_date_idx, sell_date_idx):
                        d_name = idx_to_date[d_i]
                        day_bar = price_dict.get((code, d_name))
                        if not day_bar:
                            continue
                        
                        # 检查跌破硬止损 (-5%)
                        if day_bar["low"] <= actual_buy_price * (1.0 - self.stop_loss_pct):
                            actual_sell_price = actual_buy_price * (1.0 - self.stop_loss_pct) * (1.0 - self.slippage)
                            exit_date = d_name
                            exit_reason = "STOP_LOSS"
                            break
                        
                        # 检查达到止盈位 (+8%)
                        if day_bar["high"] >= actual_buy_price * (1.0 + self.take_profit_pct):
                            actual_sell_price = actual_buy_price * (1.0 + self.take_profit_pct) * (1.0 - self.slippage)
                            exit_date = d_name
                            exit_reason = "TAKE_PROFIT"
                            break

                # 若未触发提前止损止盈，则在 sell_date 开盘卖出
                if actual_sell_price is None:
                    sell_bar = price_dict.get((code, sell_date))
                    if sell_bar and sell_bar["open"] > 0:
                        raw_sell_price = float(sell_bar["open"])
                    elif sell_bar and sell_bar["close"] > 0:
                        raw_sell_price = float(sell_bar["close"])
                    else:
                        raw_sell_price = actual_buy_price  # 兜底
                    actual_sell_price = raw_sell_price * (1.0 - self.slippage)

                # 卖出税费：佣金 + 印花税
                sell_fee_rate = self.commission + self.stamp_duty

                # 净收益率 (扣除所有买入佣金、滑点与卖出税费)
                net_pnl_pct = (actual_sell_price * (1.0 - sell_fee_rate)) / (actual_buy_price * (1.0 + buy_fee_rate)) - 1.0
                
                # 计算持股数与绝对盈亏金额
                shares = int(alloc_cash_per_stock // (actual_buy_price * 100)) * 100
                if shares <= 0:
                    shares = 100  # 小资金保底1手
                
                trade_invest = shares * actual_buy_price
                trade_pnl = trade_invest * net_pnl_pct

                cycle_returns.append(net_pnl_pct)
                cycle_trades.append({
                    "cycle": cycle_idx + 1,
                    "signal_date": signal_date,
                    "symbol": code,
                    "buy_date": buy_date,
                    "buy_price": actual_buy_price,
                    "sell_date": exit_date,
                    "sell_price": actual_sell_price,
                    "hold_days": self.hold_days,
                    "net_ret_pct": net_pnl_pct * 100.0,
                    "pnl_amount": trade_pnl,
                    "exit_reason": exit_reason,
                })

            if not cycle_returns:
                continue

            # 本轮收益与净值更新
            cycle_avg_ret = float(np.mean(cycle_returns))
            current_cash *= (1.0 + cycle_avg_ret)
            portfolio_history.append({
                "date": sell_date,
                "equity": current_cash,
                "return_pct": (current_cash / self.initial_cash - 1.0) * 100.0,
            })

            # 计算同期基准 (中证500样本股票等权收益)
            bench_returns = []
            for d_cand in scored_df.filter(pl.col("date") == signal_date).to_dicts():
                b_code = d_cand["symbol"]
                b_buy = price_dict.get((b_code, buy_date))
                b_sell = price_dict.get((b_code, sell_date))
                if b_buy and b_sell and b_buy["open"] > 0 and b_sell["open"] > 0:
                    bench_returns.append(b_sell["open"] / b_buy["open"] - 1.0)
            
            bench_avg_ret = float(np.mean(bench_returns)) if bench_returns else 0.0
            benchmark_cash *= (1.0 + bench_avg_ret)
            benchmark_history.append({
                "date": sell_date,
                "equity": benchmark_cash,
                "return_pct": (benchmark_cash / self.initial_cash - 1.0) * 100.0,
            })

            trades.extend(cycle_trades)

        # 7. 计算全景绩效评价指标
        metrics = self._calculate_performance_metrics(
            portfolio_history, benchmark_history, trades
        )

        # 8. 打印排版漂亮的终端研报
        self._print_backtest_report(metrics, trades, portfolio_history)

        return {
            "metrics": metrics,
            "trades": trades,
            "portfolio_history": portfolio_history,
            "benchmark_history": benchmark_history,
        }

    def _calculate_performance_metrics(
        self,
        portfolio_history: List[Dict],
        benchmark_history: List[Dict],
        trades: List[Dict],
    ) -> Dict[str, Any]:
        """计算机构级绩效与风险指标"""
        if not portfolio_history or len(portfolio_history) < 2:
            return {}

        equities = [p["equity"] for p in portfolio_history]
        bench_equities = [b["equity"] for b in benchmark_history]
        dates = [p["date"] for p in portfolio_history]

        total_return = (equities[-1] / equities[0]) - 1.0
        bench_total_return = (bench_equities[-1] / bench_equities[0]) - 1.0
        alpha_return = total_return - bench_total_return

        # 交易日数折算年化 (250个交易日)
        num_cycles = len(equities) - 1
        days_span = num_cycles * self.hold_days
        annual_factor = 250.0 / max(days_span, 1)

        annual_return = (1.0 + total_return) ** (annual_factor) - 1.0
        bench_annual_return = (1.0 + bench_total_return) ** (annual_factor) - 1.0

        # 周频周期收益率序列
        period_returns = [
            (equities[i] / equities[i - 1]) - 1.0 for i in range(1, len(equities))
        ]
        ret_mean = np.mean(period_returns)
        ret_std = np.std(period_returns) + 1e-8
        
        # 年化波动率与夏普比率 (无风险利率假定 2%)
        periods_per_year = 250.0 / self.hold_days
        annual_volatility = ret_std * np.sqrt(periods_per_year)
        sharpe_ratio = (annual_return - 0.02) / max(annual_volatility, 1e-6)

        # 最大回撤计算 (Max Drawdown)
        peaks = np.maximum.accumulate(equities)
        drawdowns = (equities - peaks) / peaks
        max_drawdown = float(np.min(drawdowns))  # 负数
        
        # 卡玛比率 (Calmar Ratio)
        calmar_ratio = annual_return / max(abs(max_drawdown), 1e-6)

        # 交易层面统计 (Trade-Level Statistics)
        total_trades = len(trades)
        winning_trades = [t for t in trades if t["net_ret_pct"] > 0]
        losing_trades = [t for t in trades if t["net_ret_pct"] <= 0]
        
        win_rate = (len(winning_trades) / total_trades * 100.0) if total_trades > 0 else 0.0
        
        avg_win_pct = np.mean([t["net_ret_pct"] for t in winning_trades]) if winning_trades else 0.0
        avg_loss_pct = np.mean([t["net_ret_pct"] for t in losing_trades]) if losing_trades else 0.0
        profit_loss_ratio = abs(avg_win_pct / avg_loss_pct) if avg_loss_pct != 0 else 0.0

        # 止损止盈触发统计
        stop_loss_count = sum(1 for t in trades if t["exit_reason"] == "STOP_LOSS")
        take_profit_count = sum(1 for t in trades if t["exit_reason"] == "TAKE_PROFIT")

        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "days_span": days_span,
            "num_cycles": num_cycles,
            "initial_cash": self.initial_cash,
            "final_equity": equities[-1],
            "total_return_pct": total_return * 100.0,
            "bench_total_return_pct": bench_total_return * 100.0,
            "alpha_return_pct": alpha_return * 100.0,
            "annual_return_pct": annual_return * 100.0,
            "bench_annual_return_pct": bench_annual_return * 100.0,
            "annual_volatility_pct": annual_volatility * 100.0,
            "sharpe_ratio": sharpe_ratio,
            "calmar_ratio": calmar_ratio,
            "max_drawdown_pct": max_drawdown * 100.0,
            "total_trades": total_trades,
            "win_trades_count": len(winning_trades),
            "loss_trades_count": len(losing_trades),
            "trade_win_rate": win_rate,
            "avg_win_pct": avg_win_pct,
            "avg_loss_pct": avg_loss_pct,
            "profit_loss_ratio": profit_loss_ratio,
            "stop_loss_count": stop_loss_count,
            "take_profit_count": take_profit_count,
        }

    def _print_backtest_report(
        self,
        metrics: Dict[str, Any],
        trades: List[Dict],
        portfolio_history: List[Dict],
    ) -> None:
        """打印格式化回测分析报告"""
        print("\n" + "=" * 75)
        print(f"📊 【量化策略历史区间回测与回滚评估报告】")
        print(f"回测区间: {metrics['start_date']} ~ {metrics['end_date']} | 周期跨度: {metrics['days_span']} 交易日 ({metrics['num_cycles']} 轮)")
        print(f"调仓机制: 每 5 交易日轮动 | 每次持仓 Top-{self.top_n} | 包含佣金(万3)+印花税(千0.5)+滑点(千1)")
        print("=" * 75)

        print("\n📈 核心收益与风险指标：")
        print(f"  • 策略累计收益率:      {metrics['total_return_pct']:+.2f}%  (初始本金: ¥{metrics['initial_cash']:,.2f} -> 最终净值: ¥{metrics['final_equity']:,.2f})")
        print(f"  • 中证500基准收益率:   {metrics['bench_total_return_pct']:+.2f}%")
        print(f"  • 累计超额收益 (Alpha): {metrics['alpha_return_pct']:+.2f}%")
        print(f"  • 策略年化收益率:      {metrics['annual_return_pct']:+.2f}%  (基准年化: {metrics['bench_annual_return_pct']:+.2f}%)")
        print(f"  • 年化波动率:          {metrics['annual_volatility_pct']:.2f}%")
        print(f"  • 夏普比率 (Sharpe):   {metrics['sharpe_ratio']:+.2f}   (以无风险利率 2.0% 折算)")
        print(f"  • 最大回撤 (Max DD):   {metrics['max_drawdown_pct']:.2f}%")
        print(f"  • 卡玛比率 (Calmar):   {metrics['calmar_ratio']:+.2f}")

        print("\n🎯 交易层面胜率与盈亏比统计：")
        print(f"  • 总交易笔数:          {metrics['total_trades']} 笔")
        print(f"  • 盈利笔数 / 亏损笔数:  {metrics['win_trades_count']} 笔 / {metrics['loss_trades_count']} 笔")
        print(f"  • 单笔交易胜率:        {metrics['trade_win_rate']:.2f}%")
        print(f"  • 平均单笔盈利:        {metrics['avg_win_pct']:+.2f}%")
        print(f"  • 平均单笔亏损:        {metrics['avg_loss_pct']:+.2f}%")
        print(f"  • 盈亏比 (Win/Loss):   {metrics['profit_loss_ratio']:.2f} : 1")
        print(f"  • 硬止损触发次数 (-5%): {metrics['stop_loss_count']} 次")
        print(f"  • 止盈触发次数 (+8%):   {metrics['take_profit_count']} 次")

        print("\n🗓️ 最近 5 轮调仓明细概览：")
        recent_trades = trades[-15:] if len(trades) >= 15 else trades
        print(f"{'轮次':<5} | {'买入日':<10} | {'代码':<8} | {'买入价':<8} | {'卖出日':<10} | {'卖出价':<8} | {'净收益率':<10} | {'退出方式'}")
        print("-" * 75)
        for t in recent_trades:
            reason_map = {"NORMAL_EXPIRE": "到期轮动", "STOP_LOSS": "触发止损", "TAKE_PROFIT": "止盈离场"}
            print(
                f"{t['cycle']:<5} | {t['buy_date']:<10} | {t['symbol']:<8} | ¥{t['buy_price']:<7.2f} | "
                f"{t['sell_date']:<10} | ¥{t['sell_price']:<7.2f} | {t['net_ret_pct']:>+7.2f}% | {reason_map.get(t['exit_reason'], t['exit_reason'])}"
            )
        print("=" * 75 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="A股量化策略历史区间回测与回滚分析")
    parser.add_argument("--start", type=str, default="2024-01-01", help="回测起点日期 (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2026-10-22", help="回测终点日期 (YYYY-MM-DD)")
    parser.add_argument("--top_n", type=int, default=5, help="每期持仓股票只数 (默认 5)")
    parser.add_argument("--cash", type=float, default=100000.0, help="初始资金 (默认 100000)")
    parser.add_argument("--no_stop", action="store_true", help="关闭止盈止损盯盘逻辑")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tester = Backtester(
        start_date=args.start,
        end_date=args.end,
        top_n=args.top_n,
        initial_cash=args.cash,
        use_stop_loss=not args.no_stop,
    )
    tester.run()
