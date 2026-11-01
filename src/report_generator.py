"""
每日量化研报与静态 HTML 档案生成器 (ReportGenerator)
===========================================================
生成完全独立、零外部依赖（内置优雅 CSS）、离线可双击直接查看的每日静态 HTML 报告，
自动存入 reports/report_YYYYMMDD.html 以及 reports/latest.html，供海外投资者离线随时回看。
"""

import os
import sys
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.config as cfg
from src.predictor import Predictor
from src.paper_trader import PaperTrader


REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class ReportGenerator:
    """静态 HTML 量化研报生成器"""

    def __init__(self, db_path: Path = cfg.DB_PATH):
        self.db_path = db_path
        self.reports_dir = REPORTS_DIR

    def _get_stock_name(self, symbol: str) -> str:
        """从数据库查询股票中文简称"""
        conn = sqlite3.connect(self.db_path.as_posix())
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM stock_info WHERE symbol=?", (symbol,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else symbol

    def generate_html_report(self) -> Path:
        """生成并落盘最新的静态 HTML 档案报告"""
        trader = PaperTrader(db_path=self.db_path)
        summary = trader.get_account_summary()
        acc = summary["account"]
        positions = summary["positions"]

        predictor = Predictor(budget=acc["total_equity"])
        pred_df = predictor.predict()
        plans = predictor.plan_portfolio(pred_df)
        latest_date = str(pred_df["date"][0])

        # 为标的补充中文简称
        for item in plans["global_top"]:
            item["name"] = self._get_stock_name(item["symbol"])
        for pos in positions:
            pos["name"] = self._get_stock_name(pos["symbol"])

        # 生成 HTML 文本
        html_content = self._render_template(
            date=latest_date,
            acc=acc,
            positions=positions,
            summary=summary,
            global_top=plans["global_top"],
        )

        # 写入 YYYYMMDD 文件及 latest.html
        date_clean = latest_date.replace("-", "")
        report_file = self.reports_dir / f"report_{date_clean}.html"
        latest_file = self.reports_dir / "latest.html"

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        with open(latest_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        return report_file

    def _render_template(
        self,
        date: str,
        acc: Dict[str, Any],
        positions: List[Dict[str, Any]],
        summary: Dict[str, Any],
        global_top: List[Dict[str, Any]],
    ) -> str:
        # 收益颜色 class
        pnl_class = "text-up" if acc["cumulative_pnl"] >= 0 else "text-down"
        pnl_sign = "+" if acc["cumulative_pnl"] >= 0 else ""

        # 持仓行
        pos_rows = ""
        if not positions:
            pos_rows = '<tr><td colspan="8" class="empty-state">当前暂无持仓，全额现金储备，等待买入信号。</td></tr>'
        else:
            for p in positions:
                item_class = "text-up" if p["unrealized_pnl"] >= 0 else "text-down"
                sign = "+" if p["unrealized_pnl"] >= 0 else ""
                pos_rows += f"""
                <tr>
                    <td><strong>{p['symbol']}</strong></td>
                    <td>{p['name']}</td>
                    <td>{p['shares']} 股</td>
                    <td>¥{p['cost_price']:.2f}</td>
                    <td>¥{p['current_price']:.2f}</td>
                    <td>{p['hold_days']} / 5 天</td>
                    <td class="{item_class}">¥{sign}{p['unrealized_pnl']:.2f}</td>
                    <td class="{item_class}">{sign}{p['unrealized_pnl_pct']:.2f}%</td>
                </tr>
                """

        # 今日 Top-5 精选卡片
        top_cards = ""
        for item in global_top:
            top_cards += f"""
            <div class="stock-card highlight-card">
                <div class="card-badge">Top #{item['rank']}</div>
                <div class="card-header">
                    <h3>{item['name']} <span class="code">({item['symbol']})</span></h3>
                    <span class="price">¥{item['close']:.2f}</span>
                </div>
                <div class="card-body">
                    <div class="metric-row"><span>预测 Alpha：</span><strong class="text-up">{item['score']:+.4f}</strong></div>
                    <div class="metric-row"><span>建议买入区间：</span><span>¥{item['buy_range']}</span></div>
                    <div class="metric-row"><span>硬止损位 (-5%)：</span><span class="text-down">¥{item['stop_loss']:.2f}</span></div>
                    <div class="metric-row"><span>目标止盈 (+8%)：</span><span class="text-up">¥{item['take_profit']:.2f}</span></div>
                    <div class="metric-row"><span>规划建仓股数：</span><span><strong>{item['target_shares']} 股</strong> (约 ¥{item['target_amount']:,.2f})</span></div>
                </div>
            </div>
            """

        # 全局 Top-5 表格
        top5_rows = ""
        for item in global_top:
            top5_rows += f"""
            <tr>
                <td>#{item['rank']}</td>
                <td><strong>{item['symbol']}</strong></td>
                <td>{item['name']}</td>
                <td>¥{item['close']:.2f}</td>
                <td class="text-up">{item['score']:+.4f}</td>
                <td>¥{item['buy_range']}</td>
                <td class="text-down">¥{item['stop_loss']:.2f}</td>
                <td class="text-up">¥{item['take_profit']:.2f}</td>
                <td><strong>{item['target_shares']} 股</strong></td>
                <td>¥{item['target_amount']:,.2f} ({item['target_weight_pct']}%)</td>
            </tr>
            """

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>A股量化周频研报 - {date}</title>
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #243248;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-blue: #38bdf8;
            --color-up: #ef4444;   /* A股红色代表上涨 */
            --color-down: #10b981; /* A股绿色代表下跌 */
            --border-color: #334155;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Microsoft YaHei", sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-main);
            padding: 24px;
            line-height: 1.5;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
        }}
        .header h1 {{ font-size: 24px; color: #fff; display: flex; align-items: center; gap: 10px; }}
        .header .meta {{ color: var(--text-muted); font-size: 14px; margin-top: 6px; }}
        .timezone-badge {{
            background: rgba(56, 189, 248, 0.15);
            color: var(--accent-blue);
            padding: 8px 14px;
            border-radius: 8px;
            border: 1px solid rgba(56, 189, 248, 0.3);
            font-size: 13px;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .kpi-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 18px;
        }}
        .kpi-card .label {{ font-size: 13px; color: var(--text-muted); margin-bottom: 6px; }}
        .kpi-card .value {{ font-size: 24px; font-weight: bold; }}
        .text-up {{ color: var(--color-up); }}
        .text-down {{ color: var(--color-down); }}
        
        .section-title {{
            font-size: 18px;
            margin: 24px 0 14px 0;
            display: flex;
            align-items: center;
            gap: 8px;
            color: var(--accent-blue);
        }}
        .card-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .stock-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            position: relative;
        }}
        .highlight-card {{ border-color: rgba(56, 189, 248, 0.4); background: linear-gradient(180deg, #1e293b 0%, #172554 100%); }}
        .card-badge {{
            position: absolute;
            top: 14px;
            right: 14px;
            background: #0284c7;
            color: #fff;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
        }}
        .card-header {{ display: flex; align-items: baseline; gap: 8px; margin-bottom: 14px; }}
        .card-header h3 {{ font-size: 20px; }}
        .card-header .code {{ color: var(--text-muted); font-size: 14px; }}
        .card-header .price {{ margin-left: auto; font-size: 22px; font-weight: bold; color: #fff; }}
        .metric-row {{
            display: flex;
            justify-content: space-between;
            padding: 6px 0;
            border-bottom: 1px dashed rgba(255, 255, 255, 0.08);
            font-size: 14px;
        }}
        .metric-row:last-child {{ border-bottom: none; }}
        
        .table-container {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow-x: auto;
            margin-bottom: 24px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 14px;
        }}
        th, td {{ padding: 12px 16px; border-bottom: 1px solid var(--border-color); }}
        th {{ background: #172033; color: var(--text-muted); font-weight: 600; font-size: 13px; }}
        tr:hover {{ background: rgba(255, 255, 255, 0.02); }}
        .empty-state {{ text-align: center; color: var(--text-muted); padding: 30px; }}
        .badge {{ padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: 500; }}
        .badge-success {{ background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }}
        .badge-warning {{ background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }}
        
        .footer {{
            text-align: center;
            color: var(--text-muted);
            font-size: 13px;
            margin-top: 36px;
            padding-top: 18px;
            border-top: 1px solid var(--border-color);
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- 头部 -->
        <div class="header">
            <div>
                <h1>🎯 A股中低频量化研究研报</h1>
                <div class="meta">
                    最新行情交易日: <strong>{date}</strong> | 轮动周期: 5 个交易日 (周频) | 模型: LightGBM 33-Factor GBDT (含 WorldQuant 101)
                </div>
            </div>
            <div class="timezone-badge">
                🌍 跨时区自动运行 (加拿大与北京双时区协同，离线自动生成)
            </div>
        </div>

        <!-- KPI 概览 -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="label">💼 模拟账户总资产</div>
                <div class="value">¥{acc['total_equity']:,.2f}</div>
            </div>
            <div class="kpi-card">
                <div class="label">💵 可用现金余额</div>
                <div class="value">¥{acc['cash']:,.2f}</div>
            </div>
            <div class="kpi-card">
                <div class="label">📈 累计收益率</div>
                <div class="value {pnl_class}">{pnl_sign}{acc['cumulative_return_pct']:.2f}%</div>
            </div>
            <div class="kpi-card">
                <div class="label">🎯 历史交易胜率</div>
                <div class="value text-up">{summary['trade_win_rate']:.1f}%</div>
            </div>
        </div>

        <!-- 模拟盘持仓 -->
        <div class="section-title">📌 当前模拟持仓列表</div>
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>股票代码</th>
                        <th>名称</th>
                        <th>持股数</th>
                        <th>成本价</th>
                        <th>当前最新价</th>
                        <th>已持仓天数</th>
                        <th>浮动盈亏</th>
                        <th>收益率</th>
                    </tr>
                </thead>
                <tbody>
                    {pos_rows}
                </tbody>
            </table>
        </div>

        <!-- 今日 AI 选股 Top-5 组合卡片 -->
        <div class="section-title">🏆 今日 AI 量化精选标的 (Top-5 组合)</div>
        <div class="card-grid">
            {top_cards}
        </div>

        <!-- 全局 Top-5 明细表 -->
        <div class="section-title">📊 Top-5 规划明细与仓位权重</div>
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>排名</th>
                        <th>代码</th>
                        <th>名称</th>
                        <th>收盘价</th>
                        <th>预测 Alpha</th>
                        <th>建议买入区间</th>
                        <th>硬止损 (-5%)</th>
                        <th>目标止盈 (+8%)</th>
                        <th>规划股数</th>
                        <th>规划金额 (目标权重)</th>
                    </tr>
                </thead>
                <tbody>
                    {top5_rows}
                </tbody>
            </table>
        </div>

        <div class="footer">
            免责声明：本页面由 AShare_Quant_System 自动化流水线离线计算生成，仅供量化模型研究与模拟盘跟踪，不构成投资建议。
        </div>
    </div>
</body>
</html>"""
        return html


if __name__ == "__main__":
    generator = ReportGenerator()
    out = generator.generate_html_report()
    print(f"✅ 成功生成最新静态 HTML 研报: {out}")
