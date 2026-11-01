"""
A股量化交互式 Web 控制台 (Web Console & Trader) - Flask 驱动
===========================================================
功能特性：
1. 双时区自适应时钟（北京时间 A股交易状态 vs 加拿大本地时间）
2. 模拟盘实盘账户资产、持仓与历史流水实时展现
3. 今日 AI 优质标的推荐 + 【🟢 一键模拟买入 (One-Click Buy)】
4. 当前持仓标的 + 【🔴 一键平仓 (Sell)】
5. 历史时间段交互式回测 (Backtest) 与 ECharts 净值曲线图表
6. 每日静态 HTML 快照归档查看
"""

import os
import sys
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

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

from flask import Flask, render_template, request, jsonify, send_from_directory

import src.config as cfg
from src.predictor import Predictor
from src.paper_trader import PaperTrader
from src.backtest import Backtester
from src.report_generator import ReportGenerator, REPORTS_DIR


app = Flask(__name__, template_folder=str(PROJECT_ROOT / "templates"))
app.config["SECRET_KEY"] = "ashare_quant_secret_2026"


def get_stock_name(symbol: str) -> str:
    """从数据库查询股票名称"""
    conn = sqlite3.connect(cfg.DB_PATH.as_posix())
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM stock_info WHERE symbol=?", (symbol,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else symbol


def get_db_connection() -> sqlite3.Connection:
    return sqlite3.connect(cfg.DB_PATH.as_posix())


@app.route("/")
def index():
    """主页面"""
    return render_template("index.html")


@app.route("/api/dashboard_data", methods=["GET"])
def get_dashboard_data():
    """获取仪表盘全景数据"""
    try:
        trader = PaperTrader()
        summary = trader.get_account_summary()
        acc = summary["account"]
        positions = summary["positions"]

        # 补全持仓股票名称与止损止盈价
        for p in positions:
            p["name"] = get_stock_name(p["symbol"])
            p["stop_loss"] = round(p["cost_price"] * (1.0 - cfg.STOP_LOSS_PCT), 2)
            p["take_profit"] = round(p["cost_price"] * (1.0 + cfg.TAKE_PROFIT_PCT), 2)

        # 获取今日 AI 预测推荐
        predictor = Predictor(budget=acc["cash"] if acc["cash"] > 300 else cfg.TOTAL_BUDGET)
        pred_df = predictor.predict()
        plans = predictor.plan_portfolio(pred_df)
        latest_date = str(pred_df["date"][0])

        for item in plans["global_top"]:
            item["name"] = get_stock_name(item["symbol"])
        for item in plans["budget_top"]:
            item["name"] = get_stock_name(item["symbol"])

        # 查询最近 20 笔历史交易流水
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
        SELECT date, symbol, side, shares, price, amount, fee, pnl, net_return_pct, reason
        FROM paper_trades
        ORDER BY trade_id DESC LIMIT 20
        """)
        trades = []
        for r in cursor.fetchall():
            trades.append({
                "date": r[0],
                "symbol": r[1],
                "name": get_stock_name(r[1]),
                "side": r[2],
                "shares": r[3],
                "price": r[4],
                "amount": r[5],
                "fee": r[6],
                "pnl": r[7],
                "net_return_pct": r[8],
                "reason": r[9],
            })
        conn.close()

        # 归档文件列表
        report_files = [f.name for f in REPORTS_DIR.glob("report_*.html")]
        report_files.sort(reverse=True)

        return jsonify({
            "status": "success",
            "latest_date": latest_date,
            "account": acc,
            "positions": positions,
            "summary": summary,
            "global_top": plans["global_top"],
            "budget_top": plans["budget_top"],
            "recent_trades": trades,
            "report_files": report_files,
        })
    except Exception as err:
        return jsonify({"status": "error", "message": str(err)}), 500


@app.route("/api/buy", methods=["POST"])
def one_click_buy():
    """一键模拟买入 API"""
    data = request.json or {}
    symbol = str(data.get("symbol", "")).zfill(6)
    if not symbol:
        return jsonify({"status": "error", "message": "股票代码不能为空！"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 查询最新价格
        cursor.execute("SELECT close FROM daily_bars WHERE symbol=? ORDER BY date DESC LIMIT 1", (symbol,))
        row = cursor.fetchone()
        if not row:
            return jsonify({"status": "error", "message": f"未查询到标的 [{symbol}] 的有效行情！"}), 404
        current_price = float(row[0])

        # 查询账户现金
        cursor.execute("SELECT cash, initial_capital FROM paper_account WHERE account_id='default_portfolio'")
        acc_row = cursor.fetchone()
        if not acc_row:
            PaperTrader().init_account()
            cursor.execute("SELECT cash, initial_capital FROM paper_account WHERE account_id='default_portfolio'")
            acc_row = cursor.fetchone()

        cash, init_cap = acc_row[0], acc_row[1]

        # 检查是否已持仓
        cursor.execute("SELECT symbol FROM paper_positions WHERE symbol=?", (symbol,))
        if cursor.fetchone():
            return jsonify({"status": "error", "message": f"标的 [{symbol}] 已在持仓中，请勿重复建仓！"}), 400

        # 计算买入手数 (默认 1手 或根据金额计算)
        shares = int(data.get("shares", 0))
        if shares <= 0:
            lot_cost = current_price * 100 * (1.0 + cfg.TRANSACTION_FEE + cfg.SLIPPAGE)
            shares = int(cash // lot_cost) * 100
            if shares <= 0:
                shares = 100  # 小资金保底1手

        buy_price = round(current_price * (1.0 + cfg.SLIPPAGE), 2)
        buy_amount = buy_price * shares
        fee = buy_amount * (cfg.TRANSACTION_FEE + cfg.SLIPPAGE)
        total_cost = buy_amount + fee

        if cash < total_cost and cash < buy_price * 100:
            return jsonify({"status": "error", "message": f"现金余额不足！买入需要 ¥{total_cost:.2f}，当前仅剩 ¥{cash:.2f}"}), 400

        # 扣减现金并记录持仓
        new_cash = max(0.0, cash - total_cost)
        now_date = datetime.now().strftime("%Y-%m-%d")

        cursor.execute("""
        INSERT INTO paper_positions (symbol, shares, cost_price, current_price, entry_date, hold_days, unrealized_pnl, unrealized_pnl_pct, updated_at)
        VALUES (?, ?, ?, ?, ?, 0, 0.0, 0.0, ?)
        """, (symbol, shares, buy_price, current_price, now_date, now_date))

        cursor.execute("""
        INSERT INTO paper_trades (date, symbol, side, shares, price, amount, fee, pnl, net_return_pct, reason)
        VALUES (?, ?, 'BUY', ?, ?, ?, ?, 0.0, 0.0, 'Web控制台一键买入')
        """, (now_date, symbol, shares, buy_price, buy_amount, fee))

        # 更新账户总资产
        cursor.execute("SELECT SUM(shares * current_price) FROM paper_positions")
        market_val = cursor.fetchone()[0] or 0.0
        total_equity = new_cash + market_val
        cum_pnl = total_equity - init_cap
        cum_ret = (total_equity / init_cap - 1.0) * 100.0

        cursor.execute("""
        UPDATE paper_account
        SET cash=?, total_equity=?, cumulative_pnl=?, cumulative_return_pct=?, updated_at=?
        WHERE account_id='default_portfolio'
        """, (new_cash, total_equity, cum_pnl, cum_ret, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        conn.commit()
        return jsonify({
            "status": "success",
            "message": f"成功买入 {get_stock_name(symbol)} ({symbol}) {shares} 股 @ ¥{buy_price:.2f}！",
            "cash": new_cash,
            "total_equity": total_equity,
        })
    finally:
        conn.close()


@app.route("/api/sell", methods=["POST"])
def one_click_sell():
    """一键模拟平仓卖出 API"""
    data = request.json or {}
    symbol = str(data.get("symbol", "")).zfill(6)
    reason = str(data.get("reason", "Web控制台手动一键平仓"))

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT shares, cost_price FROM paper_positions WHERE symbol=?", (symbol,))
        pos_row = cursor.fetchone()
        if not pos_row:
            return jsonify({"status": "error", "message": f"未持有标的 [{symbol}]！"}), 404

        shares, cost_price = pos_row[0], pos_row[1]

        # 查询最新收盘价
        cursor.execute("SELECT close FROM daily_bars WHERE symbol=? ORDER BY date DESC LIMIT 1", (symbol,))
        current_price = float(cursor.fetchone()[0])

        sell_price = round(current_price * (1.0 - cfg.SLIPPAGE), 2)
        sell_amount = sell_price * shares
        fee = sell_amount * (cfg.TRANSACTION_FEE + cfg.STAMP_DUTY + cfg.SLIPPAGE)
        net_cash = sell_amount - fee
        realized_pnl = net_cash - (cost_price * shares)
        net_ret_pct = (net_cash / (cost_price * shares) - 1.0) * 100.0

        cursor.execute("DELETE FROM paper_positions WHERE symbol=?", (symbol,))

        now_date = datetime.now().strftime("%Y-%m-%d")
        cursor.execute("""
        INSERT INTO paper_trades (date, symbol, side, shares, price, amount, fee, pnl, net_return_pct, reason)
        VALUES (?, ?, 'SELL', ?, ?, ?, ?, ?, ?, ?)
        """, (now_date, symbol, shares, sell_price, sell_amount, fee, realized_pnl, net_ret_pct, reason))

        # 更新现金与账户净值
        cursor.execute("SELECT cash, initial_capital FROM paper_account WHERE account_id='default_portfolio'")
        acc_row = cursor.fetchone()
        old_cash, init_cap = acc_row[0], acc_row[1]
        new_cash = old_cash + net_cash

        cursor.execute("SELECT SUM(shares * current_price) FROM paper_positions")
        market_val = cursor.fetchone()[0] or 0.0
        total_equity = new_cash + market_val
        cum_pnl = total_equity - init_cap
        cum_ret = (total_equity / init_cap - 1.0) * 100.0

        cursor.execute("""
        UPDATE paper_account
        SET cash=?, total_equity=?, cumulative_pnl=?, cumulative_return_pct=?, updated_at=?
        WHERE account_id='default_portfolio'
        """, (new_cash, total_equity, cum_pnl, cum_ret, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        conn.commit()
        return jsonify({
            "status": "success",
            "message": f"成功平仓 {get_stock_name(symbol)} ({symbol}) {shares} 股！实现盈亏: ¥{realized_pnl:+.2f} ({net_ret_pct:+.2f}%)",
            "cash": new_cash,
            "total_equity": total_equity,
        })
    finally:
        conn.close()


@app.route("/api/step", methods=["POST"])
def execute_daily_step():
    """触发每日模拟盘调仓步进"""
    try:
        trader = PaperTrader()
        trader.step(push_notification=False)
        # 生成静态报告
        ReportGenerator().generate_html_report()
        return jsonify({"status": "success", "message": "模拟盘自动盯盘与周频轮动执行完毕！"})
    except Exception as err:
        return jsonify({"status": "error", "message": str(err)}), 500


@app.route("/api/reset", methods=["POST"])
def reset_account():
    """重置模拟盘本金"""
    data = request.json or {}
    capital = float(data.get("capital", cfg.TOTAL_BUDGET))
    PaperTrader().init_account(capital)
    ReportGenerator().generate_html_report()
    return jsonify({"status": "success", "message": f"账户已成功重置！初始本金: ¥{capital:,.2f}"})


@app.route("/api/backtest", methods=["POST"])
def run_interactive_backtest():
    """交互式区间回测与图表数据生成"""
    data = request.json or {}
    start_date = data.get("start_date", "2024-01-01")
    end_date = data.get("end_date", "2024-12-31")
    top_n = int(data.get("top_n", 5))
    cash = float(data.get("cash", 100000.0))
    no_stop = bool(data.get("no_stop", True))

    try:
        tester = Backtester(
            start_date=start_date,
            end_date=end_date,
            top_n=top_n,
            initial_cash=cash,
            use_stop_loss=not no_stop,
        )
        res = tester.run()

        # 整理供 ECharts 渲染的曲线
        p_hist = res["portfolio_history"]
        b_hist = res["benchmark_history"]

        chart_dates = [p["date"] for p in p_hist]
        strategy_returns = [round(p["return_pct"], 2) for p in p_hist]
        benchmark_returns = [round(b["return_pct"], 2) for b in b_hist]

        # 补充标的名称
        trades = res["trades"]
        for t in trades:
            t["name"] = get_stock_name(t["symbol"])

        return jsonify({
            "status": "success",
            "metrics": res["metrics"],
            "chart": {
                "dates": chart_dates,
                "strategy": strategy_returns,
                "benchmark": benchmark_returns,
            },
            "recent_trades": trades[-20:],
        })
    except Exception as err:
        return jsonify({"status": "error", "message": str(err)}), 500


@app.route("/reports/<path:filename>")
def serve_report(filename):
    """查看归档的历史 HTML 研报"""
    return send_from_directory(REPORTS_DIR, filename)


def start_server(port: int = 5000, debug: bool = False):
    print("=" * 65)
    print("🚀 【AShare Quant 量化交互式 Web 控制台】已启动！")
    print(f"👉 请在浏览器打开访问: http://127.0.0.1:{port}")
    print("=" * 65)
    app.run(host="127.0.0.1", port=port, debug=debug)


if __name__ == "__main__":
    start_server()
