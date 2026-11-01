"""
实盘模拟盘记账与自动跟踪引擎 (PaperTrader) - 机构级实盘模拟版
======================================================================
功能特性：
1. 本地轻量化 SQLite 账本 (Virtual Account):
   - 自动维护 paper_account (资金与净值)、paper_positions (当前持仓)、paper_trades (历史交易流水)
2. 5 交易日严格周频轮动与自动换仓:
   - 每日开盘/收盘自动检查持仓天数，达到 5 日强制平仓释放资金
   - 盘中实时硬止损 (-5%) 与目标止盈 (+8%) 检查
3. 动态买入与资金规划:
   - 基于标准账户资金 (默认 100,000 元)，按 Top-5 组合等权与 A 股 100 股起买规则执行模拟买入
4. 盈亏与胜率实时核算:
   - 统计当前持仓浮动盈亏、已平仓胜率、盈亏比与累计收益率
   - 联动 Notifier 推送每日模拟盘资产变动研报至微信
"""

import os
import sys
import argparse
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 修复 Windows 控制台 GBK 编码输出 Emoji 乱码
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import polars as pl
import pandas as pd
import numpy as np

import src.config as cfg
from src.predictor import Predictor
from src.notifier import Notifier


def setup_logger(name: str = "PaperTrader") -> logging.Logger:
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


class PaperTrader:
    """
    实盘模拟盘自动交易与资金管理引擎
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        default_capital: float = cfg.TOTAL_BUDGET,
    ):
        self.logger = setup_logger()
        self.db_path = db_path or cfg.DB_PATH
        self.default_capital = default_capital
        self._init_database_tables()

    def _get_connection(self) -> sqlite3.Connection:
        """获取数据库连接"""
        return sqlite3.connect(self.db_path.as_posix())

    def _init_database_tables(self) -> None:
        """初始化模拟盘所需的数据库表结构"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # 1. 账户信息表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS paper_account (
            account_id TEXT PRIMARY KEY,
            initial_capital REAL,
            cash REAL,
            total_equity REAL,
            cumulative_pnl REAL,
            cumulative_return_pct REAL,
            updated_at TEXT
        )
        """)

        # 2. 当前持仓表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS paper_positions (
            symbol TEXT PRIMARY KEY,
            shares INTEGER,
            cost_price REAL,
            current_price REAL,
            entry_date TEXT,
            hold_days INTEGER,
            unrealized_pnl REAL,
            unrealized_pnl_pct REAL,
            updated_at TEXT
        )
        """)

        # 3. 交易流水表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS paper_trades (
            trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            symbol TEXT,
            side TEXT, -- BUY / SELL
            shares INTEGER,
            price REAL,
            amount REAL,
            fee REAL,
            pnl REAL,
            net_return_pct REAL,
            reason TEXT
        )
        """)

        conn.commit()
        conn.close()

    def init_account(self, capital: Optional[float] = None) -> None:
        """重置或初始化模拟盘账户"""
        init_cap = float(capital or self.default_capital)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM paper_account")
        cursor.execute("DELETE FROM paper_positions")
        cursor.execute("DELETE FROM paper_trades")

        cursor.execute("""
        INSERT INTO paper_account (account_id, initial_capital, cash, total_equity, cumulative_pnl, cumulative_return_pct, updated_at)
        VALUES ('default_portfolio', ?, ?, ?, 0.0, 0.0, ?)
        """, (init_cap, init_cap, init_cap, now_str))

        conn.commit()
        conn.close()

        self.logger.info(f"✅ 模拟盘账户初始化成功！初始可用本金: ¥{init_cap:,.2f}")

    def get_account_summary(self) -> Dict[str, Any]:
        """查询模拟盘账户总体资产与收益"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT initial_capital, cash, total_equity, cumulative_pnl, cumulative_return_pct, updated_at FROM paper_account WHERE account_id='default_portfolio'")
        row = cursor.fetchone()

        if not row:
            conn.close()
            # 若不存在则自动初始化
            self.init_account(self.default_capital)
            return self.get_account_summary()

        account = {
            "initial_capital": row[0],
            "cash": row[1],
            "total_equity": row[2],
            "cumulative_pnl": row[3],
            "cumulative_return_pct": row[4],
            "updated_at": row[5],
        }

        # 查询当前持仓
        positions = []
        cursor.execute("SELECT symbol, shares, cost_price, current_price, entry_date, hold_days, unrealized_pnl, unrealized_pnl_pct FROM paper_positions")
        for p in cursor.fetchall():
            positions.append({
                "symbol": p[0],
                "shares": p[1],
                "cost_price": p[2],
                "current_price": p[3],
                "entry_date": p[4],
                "hold_days": p[5],
                "unrealized_pnl": p[6],
                "unrealized_pnl_pct": p[7],
            })

        # 查询历史平仓胜率
        cursor.execute("SELECT COUNT(*), SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), SUM(pnl) FROM paper_trades WHERE side='SELL'")
        trades_stat = cursor.fetchone()
        total_closed = trades_stat[0] or 0
        win_closed = trades_stat[1] or 0
        total_realized_pnl = trades_stat[2] or 0.0
        win_rate = (win_closed / total_closed * 100.0) if total_closed > 0 else 0.0

        conn.close()

        return {
            "account": account,
            "positions": positions,
            "total_closed_trades": total_closed,
            "win_trades": win_closed,
            "trade_win_rate": win_rate,
            "realized_pnl": total_realized_pnl,
        }

    def print_status(self) -> str:
        """格式化输出模拟盘账户状态并在终端打印"""
        summary = self.get_account_summary()
        acc = summary["account"]
        positions = summary["positions"]

        report_lines = [
            "=" * 70,
            "💼 【A股量化模拟盘实盘跟踪账本】",
            f"账户初始本金: ¥{acc['initial_capital']:,.2f} | 当前总资产净值: ¥{acc['total_equity']:,.2f}",
            f"可用现金余额: ¥{acc['cash']:,.2f} | 累计总盈亏: ¥{acc['cumulative_pnl']:+,.2f} ({acc['cumulative_return_pct']:+.2f}%)",
            f"已平仓交易胜率: {summary['trade_win_rate']:.1f}% ({summary['win_trades']}/{summary['total_closed_trades']} 笔盈利) | 累计实现盈亏: ¥{summary['realized_pnl']:+,.2f}",
            "=" * 70,
        ]

        if not positions:
            report_lines.append("\n  [当前暂无持仓标的，资金全部处于现金状态，等待调仓信号执行]\n")
        else:
            report_lines.append("\n📌 当前持仓明细：")
            report_lines.append(f"{'代码':<8} | {'持股':<6} | {'成本价':<8} | {'现价':<8} | {'持仓日':<6} | {'浮动盈亏':<12} | {'收益率':<8}")
            report_lines.append("-" * 70)
            for p in positions:
                report_lines.append(
                    f"{p['symbol']:<8} | {p['shares']:<6} | ¥{p['cost_price']:<7.2f} | ¥{p['current_price']:<7.2f} | "
                    f"{p['hold_days']}天   | ¥{p['unrealized_pnl']:>+9.2f} | {p['unrealized_pnl_pct']:>+6.2f}%"
                )

        report_lines.append("=" * 70)
        report_text = "\n".join(report_lines)
        print("\n" + report_text + "\n")
        return report_text

    def step(self, push_notification: bool = True) -> str:
        """
        每日模拟盘自动调仓与盯盘执行引擎：
        1. 读取最新行情更新现有持仓现价与浮盈浮亏
        2. 检查是否触发止损 (-5%)、止盈 (+8%) 或持有满 5 天平仓
        3. 释放现金后，根据最新 Predictor 推荐的合格标的买入建仓
        4. 保存账本并推送通知
        """
        self.logger.info("=" * 60)
        self.logger.info("启动模拟盘每日调仓与资产盯盘流水线...")
        self.logger.info("=" * 60)

        conn = self._get_connection()
        cursor = conn.cursor()

        # 1. 查询账户与现有持仓
        cursor.execute("SELECT cash, initial_capital FROM paper_account WHERE account_id='default_portfolio'")
        acc_row = cursor.fetchone()
        if not acc_row:
            self.init_account()
            conn = self._get_connection()
            cursor = conn.cursor()
            acc_row = (self.default_capital, self.default_capital)

        current_cash, initial_cap = acc_row[0], acc_row[1]

        # 2. 查询最新市场收盘日
        cursor.execute("SELECT MAX(date) FROM daily_bars")
        latest_market_date = cursor.fetchone()[0]
        self.logger.info(f"市场最新交易日: {latest_market_date}")

        # 3. 检查并更新现有持仓
        cursor.execute("SELECT symbol, shares, cost_price, entry_date, hold_days FROM paper_positions")
        curr_positions = cursor.fetchall()

        trades_executed = []

        for pos in curr_positions:
            code, shares, cost_price, entry_date, hold_days = pos
            # 查询最新 bar
            cursor.execute("SELECT open, high, low, close FROM daily_bars WHERE symbol=? AND date=?", (code, latest_market_date))
            bar = cursor.fetchone()
            if not bar:
                continue

            b_open, b_high, b_low, b_close = bar
            new_hold_days = hold_days + 1
            unrealized_pnl = (b_close - cost_price) * shares
            unrealized_pnl_pct = (b_close / cost_price - 1.0) * 100.0

            # 检查卖出退出触发条件
            should_sell = False
            sell_reason = ""
            sell_price = b_close

            if b_low <= cost_price * (1.0 - cfg.STOP_LOSS_PCT):
                should_sell = True
                sell_reason = "硬止损触发 (-5%)"
                sell_price = round(cost_price * (1.0 - cfg.STOP_LOSS_PCT), 2)
            elif b_high >= cost_price * (1.0 + cfg.TAKE_PROFIT_PCT):
                should_sell = True
                sell_reason = "止盈触发 (+8%)"
                sell_price = round(cost_price * (1.0 + cfg.TAKE_PROFIT_PCT), 2)
            elif new_hold_days >= cfg.HOLD_DAYS:
                should_sell = True
                sell_reason = "持仓满5日到期轮动"
                sell_price = b_close

            if should_sell:
                # 执行卖出平仓
                sell_amount = sell_price * shares
                fee = sell_amount * (cfg.TRANSACTION_FEE + cfg.STAMP_DUTY + cfg.SLIPPAGE)
                net_sell_cash = sell_amount - fee
                realized_pnl = net_sell_cash - (cost_price * shares)
                net_ret_pct = (net_sell_cash / (cost_price * shares) - 1.0) * 100.0

                current_cash += net_sell_cash
                cursor.execute("DELETE FROM paper_positions WHERE symbol=?", (code,))
                cursor.execute("""
                INSERT INTO paper_trades (date, symbol, side, shares, price, amount, fee, pnl, net_return_pct, reason)
                VALUES (?, ?, 'SELL', ?, ?, ?, ?, ?, ?, ?)
                """, (latest_market_date, code, shares, sell_price, sell_amount, fee, realized_pnl, net_ret_pct, sell_reason))

                trades_executed.append(f"🔴 【卖出平仓】{code} {shares}股 @ ¥{sell_price:.2f} | 盈亏: ¥{realized_pnl:+.2f} ({net_ret_pct:+.2f}%) | 原因: {sell_reason}")
                self.logger.info(trades_executed[-1])
            else:
                # 更新持仓状态
                cursor.execute("""
                UPDATE paper_positions
                SET current_price=?, hold_days=?, unrealized_pnl=?, unrealized_pnl_pct=?, updated_at=?
                VALUES (?, ?, ?, ?, ?)
                WHERE symbol=?
                """, (b_close, new_hold_days, unrealized_pnl, unrealized_pnl_pct, latest_market_date, code))

        # 4. 生成最新推荐标的并模拟建仓
        predictor = Predictor(budget=initial_cap)
        pred_df = predictor.predict()
        plans = predictor.plan_portfolio(pred_df)

        target_buys = plans["global_top"]

        # 检查是否已有持仓，避免重复买入
        cursor.execute("SELECT symbol FROM paper_positions")
        existing_codes = {r[0] for r in cursor.fetchall()}

        max_slots = cfg.TOP_N_PICKS
        open_slots = max(1, max_slots - len(existing_codes))

        if open_slots > 0 and current_cash > 2000.0:
            avail_cash_per_stock = current_cash / open_slots

            for cand in target_buys:
                if len(existing_codes) >= max_slots:
                    break

                c_code = cand["symbol"]
                if c_code in existing_codes:
                    continue

                c_close = cand["close"]
                lot_cost = c_close * 100.0 * (1.0 + cfg.TRANSACTION_FEE + cfg.SLIPPAGE)

                if current_cash >= lot_cost and avail_cash_per_stock >= lot_cost:
                    # 计算可买手数
                    buy_shares = int(avail_cash_per_stock // lot_cost) * 100
                    if buy_shares <= 0:
                        buy_shares = 100

                    buy_price = round(c_close * (1.0 + cfg.SLIPPAGE), 2)
                    buy_amount = buy_price * buy_shares
                    buy_fee = buy_amount * (cfg.TRANSACTION_FEE + cfg.SLIPPAGE)
                    total_cost = buy_amount + buy_fee

                    if current_cash < total_cost:
                        continue

                    current_cash -= total_cost
                    existing_codes.add(c_code)

                    cursor.execute("""
                    INSERT INTO paper_positions (symbol, shares, cost_price, current_price, entry_date, hold_days, unrealized_pnl, unrealized_pnl_pct, updated_at)
                    VALUES (?, ?, ?, ?, ?, 0, 0.0, 0.0, ?)
                    """, (c_code, buy_shares, buy_price, c_close, latest_market_date, latest_market_date))

                    cursor.execute("""
                    INSERT INTO paper_trades (date, symbol, side, shares, price, amount, fee, pnl, net_return_pct, reason)
                    VALUES (?, ?, 'BUY', ?, ?, ?, ?, 0.0, 0.0, '信号入场建仓')
                    """, (latest_market_date, c_code, buy_shares, buy_price, buy_amount, buy_fee))

                    trades_executed.append(f"🟢 【买入建仓】{c_code} {buy_shares}股 @ ¥{buy_price:.2f} (占用资金: ¥{total_cost:,.2f})")
                    self.logger.info(trades_executed[-1])

        # 5. 核算最新总资产净值
        cursor.execute("SELECT SUM(shares * current_price) FROM paper_positions")
        market_val = cursor.fetchone()[0] or 0.0
        total_equity = current_cash + market_val
        cum_pnl = total_equity - initial_cap
        cum_ret_pct = (total_equity / initial_cap - 1.0) * 100.0

        cursor.execute("""
        UPDATE paper_account
        SET cash=?, total_equity=?, cumulative_pnl=?, cumulative_return_pct=?, updated_at=?
        WHERE account_id='default_portfolio'
        """, (current_cash, total_equity, cum_pnl, cum_ret_pct, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        conn.commit()
        conn.close()

        # 6. 生成并打印模拟盘状态
        status_report = self.print_status()

        if push_notification and trades_executed:
            notifier = Notifier()
            title = f"【模拟盘变动】{latest_market_date} 调仓与资产更新"
            content = f"### 🚀 模拟盘今日交易明细\n" + "\n".join([f"- {t}" for t in trades_executed]) + f"\n\n```text\n{status_report}\n```"
            notifier.broadcast(title=title, content=content)

        return status_report


def parse_args():
    parser = argparse.ArgumentParser(description="A股量化模拟盘自动交易与记账引擎")
    parser.add_argument("--action", type=str, choices=["init", "status", "step"], default="status", help="操作: init(初始化), status(查询账本), step(执行调仓)")
    parser.add_argument("--capital", type=float, default=cfg.TOTAL_BUDGET, help="初始本金 (默认 100000)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trader = PaperTrader(default_capital=args.capital)

    if args.action == "init":
        trader.init_account(args.capital)
    elif args.action == "step":
        trader.step()
    else:
        trader.print_status()
