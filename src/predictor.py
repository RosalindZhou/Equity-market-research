"""
实盘预测与信号层 (Predictor) - 33 维 Alpha 模型适配版 (含 WorldQuant 101)
===================================================
1. 最新截面提取: 读取数据库最近 80 交易日量价，极速衍生 33 维 Alpha 因子并计算今日横截面 Rank。
2. 生产模型推断: 加载训练好的 33 维 LightGBM 模型 (src/lgb_model.txt) 执行前向评分。
3. 资金与风控引擎: 基于标准账户总资金与 Top-N 等权组合规划，
   计算建议买入价、规划持股数、仓位权重、建议止盈位与硬止损位。
4. 格式化排版: 输出高辨识度 Markdown 报告，自动调用 Notifier 广播至手机微信。
"""

import os
import sys
import sqlite3
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 修复 Windows 控制台 GBK 编码输出 Emoji 乱码问题
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

from src.notifier import Notifier
import src.config as cfg


# 路径配置
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
SRC_DIR = PROJECT_ROOT / "src"

DEFAULT_DB_PATH = DATA_DIR / "market_data.db"
DEFAULT_MODEL_PATH = SRC_DIR / "lgb_model.txt"
DEFAULT_LOG_PATH = LOGS_DIR / "model_run.log"


def setup_logger(log_file: Path = DEFAULT_LOG_PATH, name: str = "Predictor") -> logging.Logger:
    """配置统一日志记录器"""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class Predictor:
    """
    实盘信号生成与仓位规划引擎 (33 维 Alpha 特征 (含 WorldQuant 101))
    """

    FEATURE_COLS = cfg.RANK_FEATURES
    RAW_FACTORS = cfg.RAW_FACTORS

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        model_path: Optional[Union[str, Path]] = None,
        budget: float = cfg.TOTAL_BUDGET,
        top_n: int = cfg.TOP_N_PICKS,
    ):
        self.logger = setup_logger(name="Predictor")
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.budget = budget
        self.top_n = top_n

    def load_latest_market_data(self) -> pl.DataFrame:
        """从数据库读取足够计算 60 日滚动因子的近期行情数据"""
        if not self.db_path.exists():
            raise FileNotFoundError(f"数据库不存在: {self.db_path}")

        conn = sqlite3.connect(self.db_path.as_posix())
        try:
            # 查询最近 90 个交易日以覆盖 60 日动量和均线
            query = """
            SELECT symbol, date, open, high, low, close, volume, amount
            FROM daily_bars
            WHERE date >= (
                SELECT date FROM (
                    SELECT DISTINCT date FROM daily_bars ORDER BY date DESC LIMIT 90
                ) ORDER BY date ASC LIMIT 1
            )
            ORDER BY symbol, date ASC
            """
            df = pl.read_database(query, connection=conn)
            if df.is_empty():
                raise ValueError("未查询到有效近期行情记录！")

            self.logger.info(
                f"成功加载近期行情数据: 共 {df.shape[0]} 条记录, 涵盖 {df['symbol'].n_unique()} 只股票, "
                f"最新收盘日: {df['date'].max()}"
            )
            return df
        finally:
            conn.close()

    def compute_latest_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算最新交易日的全市场 33 维因子横截面 (含 WorldQuant 101)"""
        self.logger.info("正在按股票分组时序计算最新 33 维量价因子 (经典 + WorldQuant 101)...")

        prev_close = pl.col("close").shift(1).over("symbol")
        prev_high = pl.col("high").shift(1).over("symbol")
        prev_low = pl.col("low").shift(1).over("symbol")
        delta_close_1 = pl.col("close") - prev_close

        # 1. 短线反转与多周期动量
        rev_1 = ((pl.col("close") / prev_close) - 1.0).alias("rev_1")
        rev_3 = ((pl.col("close") / pl.col("close").shift(3).over("symbol")) - 1.0).alias("rev_3")
        mom_5 = ((pl.col("close") / pl.col("close").shift(5).over("symbol")) - 1.0).alias("mom_5")
        mom_20 = ((pl.col("close") / pl.col("close").shift(20).over("symbol")) - 1.0).alias("mom_20")
        mom_60 = ((pl.col("close") / pl.col("close").shift(60).over("symbol")) - 1.0).alias("mom_60")

        # 2. 均线与形态衍生
        ma5 = pl.col("close").rolling_mean(5).over("symbol")
        ma10 = pl.col("close").rolling_mean(10).over("symbol")
        ma20 = pl.col("close").rolling_mean(20).over("symbol")
        ma60 = pl.col("close").rolling_mean(60).over("symbol")

        bias_20 = ((pl.col("close") / (ma20 + 1e-8)) - 1.0).alias("bias_20")
        high_20_ratio = (pl.col("close") / (pl.col("high").rolling_max(20).over("symbol") + 1e-8)).alias("high_20_ratio")

        ma_alignment = (
            (pl.col("close") > ma5).cast(pl.Int32)
            + (ma5 > ma10).cast(pl.Int32)
            + (ma10 > ma20).cast(pl.Int32)
            + (ma20 > ma60).cast(pl.Int32)
        ).cast(pl.Float64).alias("ma_alignment")

        # 3. 波动率与极值压力
        log_ret = (pl.col("close") / (prev_close + 1e-8)).log()
        vol_20 = log_ret.rolling_std(20).over("symbol").alias("vol_20")

        downside_ret = pl.when(log_ret < 0.0).then(log_ret).otherwise(0.0)
        downside_vol_20 = (downside_ret.pow(2).rolling_mean(20).over("symbol").sqrt()).alias("downside_vol_20")

        tr1 = pl.col("high") - pl.col("low")
        tr2 = (pl.col("high") - prev_close).abs()
        tr3 = (pl.col("low") - prev_close).abs()
        tr = pl.max_horizontal(tr1, tr2, tr3)
        atr_20 = ((tr.rolling_mean(20).over("symbol")) / (pl.col("close") + 1e-8)).alias("atr_20")

        candle_body_top = pl.max_horizontal(pl.col("open"), pl.col("close"))
        candle_range = pl.col("high") - pl.col("low") + 1e-6
        upper_shadow = (pl.col("high") - candle_body_top) / candle_range
        shadow_upper_10 = upper_shadow.rolling_mean(10).over("symbol").alias("shadow_upper_10")

        candle_body_bottom = pl.min_horizontal(pl.col("open"), pl.col("close"))
        lower_shadow = (candle_body_bottom - pl.col("low")) / candle_range
        shadow_lower_10 = lower_shadow.rolling_mean(10).over("symbol").alias("shadow_lower_10")

        # 4. 微观结构与资金流
        vol_m5 = pl.col("volume").rolling_mean(5).over("symbol")
        vol_m20 = pl.col("volume").rolling_mean(20).over("symbol")
        vol_ratio_5_20 = (vol_m5 / (vol_m20 + 1e-8)).alias("vol_ratio_5_20")

        pv_corr_20 = pl.rolling_corr(rev_1, pl.col("volume"), window_size=20).over("symbol").fill_null(0.0).alias("pv_corr_20")
        amihud_liquidity = ((rev_1.abs() / (pl.col("amount") + 1.0)) * 1e8).rolling_mean(20).over("symbol").alias("amihud_liquidity")

        ret_overnight = (pl.col("open") / (prev_close + 1e-8)) - 1.0
        ret_intraday = (pl.col("close") / (pl.col("open") + 1e-8)) - 1.0
        overnight_intraday_spread = ((ret_overnight - ret_intraday).rolling_mean(10).over("symbol")).alias("overnight_intraday_spread")

        # 5. WorldQuant 101 Formulaic Alphas (16 维)
        ret = rev_1

        # wq_alpha_001
        vol_term = ret.rolling_std(20).over("symbol")
        inner_001 = pl.when(ret < 0.0).then(vol_term).otherwise(pl.col("close"))
        signed_pow_001 = inner_001.sign() * (inner_001.abs().pow(2.0))
        ts_argmax_5 = sum([(signed_pow_001 >= signed_pow_001.shift(i).over("symbol")).cast(pl.Float64) for i in range(5)]) / 5.0
        wq_alpha_001 = ts_argmax_5.alias("wq_alpha_001")

        # wq_alpha_002
        delta_log_vol = (pl.col("volume") + 1.0).log() - (pl.col("volume").shift(2).over("symbol") + 1.0).log()
        wq_alpha_002 = (-1.0 * pl.rolling_corr(delta_log_vol, ret_intraday, window_size=6).over("symbol").fill_null(0.0)).alias("wq_alpha_002")

        # wq_alpha_003
        wq_alpha_003 = (-1.0 * pl.rolling_corr(pl.col("open"), pl.col("volume"), window_size=10).over("symbol").fill_null(0.0)).alias("wq_alpha_003")

        # wq_alpha_004
        ts_rank_low_9 = sum([(pl.col("low") >= pl.col("low").shift(i).over("symbol")).cast(pl.Float64) for i in range(9)]) / 9.0
        wq_alpha_004 = (-1.0 * ts_rank_low_9).alias("wq_alpha_004")

        # wq_alpha_006
        wq_alpha_006 = (-1.0 * pl.rolling_corr(pl.col("open"), pl.col("volume"), window_size=10).over("symbol").fill_null(0.0)).alias("wq_alpha_006")

        # wq_alpha_009
        ts_min_5 = delta_close_1.rolling_min(5).over("symbol")
        ts_max_5 = delta_close_1.rolling_max(5).over("symbol")
        wq_alpha_009 = (pl.when(ts_min_5 > 0).then(delta_close_1).when(ts_max_5 < 0).then(delta_close_1).otherwise(-1.0 * delta_close_1)).alias("wq_alpha_009")

        # wq_alpha_012
        delta_vol_1 = pl.col("volume") - pl.col("volume").shift(1).over("symbol")
        wq_alpha_012 = (delta_vol_1.sign() * (-1.0 * delta_close_1)).alias("wq_alpha_012")

        # wq_alpha_014
        delta_ret_3 = ret - ret.shift(3).over("symbol")
        wq_alpha_014 = (-1.0 * delta_ret_3 * pl.rolling_corr(pl.col("open"), pl.col("volume"), window_size=10).over("symbol").fill_null(0.0)).alias("wq_alpha_014")

        # wq_alpha_020
        term1_020 = pl.col("open") - prev_high
        term2_020 = pl.col("open") - prev_close
        term3_020 = pl.col("open") - prev_low
        wq_alpha_020 = (-1.0 * term1_020 * term2_020 * term3_020).alias("wq_alpha_020")

        # wq_alpha_023
        high_m20 = pl.col("high").rolling_mean(20).over("symbol")
        delta_high_2 = pl.col("high") - pl.col("high").shift(2).over("symbol")
        wq_alpha_023 = pl.when((pl.col("high") > prev_high) & (pl.col("high") > high_m20)).then(-1.0 * delta_high_2).otherwise(0.0).alias("wq_alpha_023")

        # wq_alpha_033
        wq_alpha_033 = (-1.0 * (1.0 - (pl.col("open") / (pl.col("close") + 1e-8)))).alias("wq_alpha_033")

        # wq_alpha_041
        vwap = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
        geom_mean = (pl.col("high") * pl.col("low")).sqrt()
        wq_alpha_041 = (geom_mean - vwap).alias("wq_alpha_041")

        # wq_alpha_044
        wq_alpha_044 = (-1.0 * pl.rolling_corr(pl.col("high"), pl.col("volume"), window_size=5).over("symbol").fill_null(0.0)).alias("wq_alpha_044")

        # wq_alpha_053
        clv = (2.0 * pl.col("close") - pl.col("high") - pl.col("low")) / (pl.col("high") - pl.col("low") + 1e-6)
        wq_alpha_053 = (-1.0 * (clv - clv.shift(9).over("symbol"))).alias("wq_alpha_053")

        # wq_alpha_054
        wq_alpha_054 = (-1.0 * ((pl.col("low") - pl.col("close")) * (pl.col("open") / (pl.col("close") + 1e-8)).pow(5.0)) / (pl.col("low") - pl.col("high") - 1e-6)).alias("wq_alpha_054")

        # wq_alpha_101
        wq_alpha_101 = ((pl.col("close") - pl.col("open")) / (pl.col("high") - pl.col("low") + 0.001)).alias("wq_alpha_101")

        factor_df = df.with_columns([
            rev_1, rev_3, mom_5, mom_20, mom_60,
            bias_20, high_20_ratio, ma_alignment,
            vol_20, downside_vol_20, atr_20, shadow_upper_10, shadow_lower_10,
            vol_ratio_5_20, pv_corr_20, amihud_liquidity, overnight_intraday_spread,
            wq_alpha_001, wq_alpha_002, wq_alpha_003, wq_alpha_004, wq_alpha_006,
            wq_alpha_009, wq_alpha_012, wq_alpha_014, wq_alpha_020, wq_alpha_023,
            wq_alpha_033, wq_alpha_041, wq_alpha_044, wq_alpha_053, wq_alpha_054, wq_alpha_101,
        ])

        latest_date = factor_df["date"].max()
        self.logger.info(f"提取最新截面交易日 [{latest_date}] 进行百分位排名归一化...")

        latest_df = factor_df.filter(pl.col("date") == latest_date)

        rank_exprs = [
            (
                pl.col(factor).rank(method="average") / pl.col(factor).count()
            ).alias(f"{factor}_rank")
            for factor in self.RAW_FACTORS
        ]

        feature_df = latest_df.with_columns(rank_exprs).drop_nulls(subset=self.FEATURE_COLS)
        self.logger.info(f"最新截面有效打分标的数: {feature_df.shape[0]} 只")
        return feature_df

    def load_model(self) -> lgb.Booster:
        """加载训练好的 LightGBM 模型 (使用字符串避免 Windows C++ 路径编码问题)"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}，请先执行 model_train.py！")

        self.logger.info(f"正在从 [{self.model_path}] 加载 LightGBM 生产模型...")
        with open(self.model_path, "r", encoding="utf-8") as f:
            booster = lgb.Booster(model_str=f.read())
        return booster

    def predict(self) -> pl.DataFrame:
        """模型前向推断与看多排序"""
        df_raw = self.load_latest_market_data()
        feature_df = self.compute_latest_features(df_raw)
        model = self.load_model()

        X = feature_df.select(self.FEATURE_COLS).to_numpy()
        pred_scores = model.predict(X)

        result_df = feature_df.select(["symbol", "date", "close"]).with_columns(
            pl.Series("pred_score", pred_scores)
        ).sort("pred_score", descending=True)

        return result_df

    def plan_portfolio(self, pred_df: pl.DataFrame) -> Dict[str, Any]:
        """基于标准账户总资产与 Top-N 等权组合规则执行资金与仓位规划"""
        top_candidates = pred_df.head(self.top_n).to_dicts()
        target_cash_per_stock = self.budget / max(len(top_candidates), 1)

        global_top = []
        for row in top_candidates:
            global_top.append(self._format_plan_row(row, len(global_top) + 1, target_cash_per_stock))

        return {
            "global_top": global_top,
        }

    def _format_plan_row(self, row: Dict[str, Any], rank: int, target_cash: float) -> Dict[str, Any]:
        code = str(row["symbol"]).zfill(6)
        close = float(row["close"])
        score = float(row["pred_score"])

        lot_cost = round(close * 100, 2)
        stop_loss = round(close * (1.0 - cfg.STOP_LOSS_PCT), 2)
        take_profit = round(close * (1.0 + cfg.TAKE_PROFIT_PCT), 2)
        buy_low = round(close * 0.995, 2)
        buy_high = round(close * 1.010, 2)
        
        # 按等权目标资金折算整数手 (每手 100 股)
        target_shares = int(target_cash // (close * 100.0)) * 100 if close > 0 else 100
        if target_shares <= 0:
            target_shares = 100
        target_amount = round(target_shares * close, 2)
        target_weight_pct = round(target_amount / self.budget * 100.0, 1)

        return {
            "rank": rank,
            "symbol": code,
            "close": close,
            "score": score,
            "lot_cost": lot_cost,
            "target_shares": target_shares,
            "target_amount": target_amount,
            "target_weight_pct": target_weight_pct,
            "buy_range": f"{buy_low:.2f} ~ {buy_high:.2f}",
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

    def generate_report(self, plans: Dict[str, Any], latest_date: str) -> str:
        """排版生成专业周频量化选股报告"""
        report_lines = [
            f"## 🎯 A股量化周频精选研报 ({latest_date})",
            "",
            f"**基准指数**: 中证500 (`000905`) | **模型算法**: LightGBM 33-Factor GBDT (含 WorldQuant 101)",
            f"**规划总资产**: ¥{self.budget:,.2f} | **持仓轮动周期**: 5 个交易日 (周频轮动)",
            "",
            "### 🏆 全局模型综合最高分 Top-5 精选组合",
            "",
            "| 排名 | 代码 | 最新收盘 | 预测Alpha | 建议买入区间 | 硬止损 (-5%) | 目标止盈 (+8%) | 规划股数 | 规划金额 | 目标权重 |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        for p in plans["global_top"]:
            line = (
                f"| #{p['rank']} | **`{p['symbol']}`** | ¥{p['close']:.2f} | "
                f"`{p['score']:+.4f}` | ¥{p['buy_range']} | "
                f"**¥{p['stop_loss']:.2f}** | ¥{p['take_profit']:.2f} | "
                f"**{p['target_shares']} 股** | ¥{p['target_amount']:,.2f} | {p['target_weight_pct']}% |"
            )
            report_lines.append(line)

        report_lines.extend([
            "",
            "---",
            "### 🛡️ 机构级实盘纪律与执行提示",
            "1. **执行方式**: 建议次日 09:25 集合竞价观察开盘价，在建议区间内手动分笔挂单买入。",
            "2. **持仓周期**: 严格按照 **5 个交易日（周频轮动）** 持有，避免频繁追涨杀跌消耗磨损与交易佣金。",
            "3. **硬止损底线**: 盘中跌破 **硬止损价 (-5%)**，坚决无条件离场，保住本金。",
            "",
            "> *免责声明：本报告由量化算法模型离线计算生成，不构成投资建议，股市有风险，入市须谨慎。*",
        ])

        return "\n".join(report_lines)

    def run_pipeline(self, push_notification: bool = True) -> str:
        """执行预测与推送流水线"""
        self.logger.info("=" * 60)
        self.logger.info("启动实盘预测与信号规划引擎 (33 维 Alpha 含 WorldQuant 101)...")
        self.logger.info("=" * 60)

        pred_df = self.predict()
        latest_date = str(pred_df["date"][0])
        plans = self.plan_portfolio(pred_df)

        report = self.generate_report(plans, latest_date=latest_date)

        self.logger.info("明日调仓与选股研报已生成：")
        print("\n" + report + "\n")

        if push_notification:
            notifier = Notifier()
            title = f"【量化优选】{latest_date} 中证500周频调仓清单"
            notifier.broadcast(title=title, content=report)

        self.logger.info("=" * 60)
        self.logger.info("实盘预测与推送流程执行完毕！")
        self.logger.info("=" * 60)

        return report


if __name__ == "__main__":
    predictor = Predictor()
    try:
        predictor.run_pipeline(push_notification=True)
    except Exception as err:
        logging.getLogger("Predictor").error(f"实盘预测运行失败: {err}", exc_info=True)
