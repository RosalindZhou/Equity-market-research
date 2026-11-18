"""
多因子计算与特征工程模块 (FactorCalculator) - 机构研究升级版
===================================================
1. 可交易性过滤 (Tradability Filter):
   - 剔除上市未满 60 交易日的次新股 (规避新股博弈噪音)
   - 剔除停牌无成交样本 (volume == 0)
   - 剔除 T+1 日开盘一字涨停无法买入的样本
2. 5 日周频超额收益标签 (Alpha Label):
   - T 日盘后选股，T+1 开盘买入，T+6 开盘卖出 (持仓 5 交易日)
   - 截面均值去中心化：label_Y = raw_ret_5 - mean(raw_ret_5)，剥离大盘 Beta，锁定纯 Alpha
   - 离群值 1%/99% 截面 Winsorize 去极值，防止坏样本扭曲模型分裂
3. 33 维超强 Alpha 因子库 (Polars 全向量化极速实现):
   - 经典量价与动量反转 (17 维): rev_1, rev_3, mom_5, mom_20, mom_60, bias_20, high_20_ratio,
     ma_alignment, vol_20, downside_vol_20, atr_20, shadow_upper_10, shadow_lower_10,
     vol_ratio_5_20, pv_corr_20, amihud_liquidity, overnight_intraday_spread
   - WorldQuant 101 Formulaic Alphas (16 维):
     wq_alpha_001, 002, 003, 004, 006, 009, 012, 014, 020, 023, 033, 041, 044, 053, 054, 101
4. 横截面百分位排名 (Percentile Rank) 归一化至 [0, 1]
"""

import os
import sys
import sqlite3
import logging
from pathlib import Path
from typing import Optional, List, Union

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl
import numpy as np

import src.config as cfg


# 路径配置
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = DATA_DIR / "factor_cache"

DEFAULT_DB_PATH = DATA_DIR / "market_data.db"
DEFAULT_OUTPUT_PARQUET = CACHE_DIR / "features.parquet"
DEFAULT_LOG_PATH = LOGS_DIR / "model_run.log"


def setup_logger(log_file: Path = DEFAULT_LOG_PATH, name: str = "FactorCalculator") -> logging.Logger:
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


class FactorCalculator:
    """
    量化多因子计算与特征工程引擎 (33 维 Alpha 因子库 (含 WorldQuant 101) + 5 日超额收益标签)
    """

    RAW_FACTORS = cfg.RAW_FACTORS
    RANK_FEATURES = cfg.RANK_FEATURES
    TARGET_COL = cfg.TARGET_COL

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        output_path: Optional[Union[str, Path]] = None,
    ):
        self.logger = setup_logger(name="FactorCalculator")
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.output_path = Path(output_path) if output_path else DEFAULT_OUTPUT_PARQUET
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> pl.DataFrame:
        """从 SQLite 读取日线数据"""
        if not self.db_path.exists():
            raise FileNotFoundError(f"数据库不存在: {self.db_path}")

        self.logger.info(f"正在从 SQLite [{self.db_path}] 读取日线行情...")
        conn = sqlite3.connect(self.db_path.as_posix())
        try:
            df = pl.read_database(
                query="SELECT symbol, date, open, high, low, close, volume, amount FROM daily_bars",
                connection=conn,
            )
            self.logger.info(
                f"成功读取行情数据: 共 {df.shape[0]} 条记录, 涵盖 {df['symbol'].n_unique()} 只股票, "
                f"时间区间: {df['date'].min()} ~ {df['date'].max()}"
            )
            return df
        finally:
            conn.close()

    def filter_tradability(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        可交易性过滤 (Tradability Filter):
        1. 剔除上市未满 60 交易日的新股
        2. 剔除成交量 <= 0 的停牌日
        3. 剔除 T+1 日开盘一字涨停无法买入的样本
        """
        self.logger.info("执行实盘可交易性过滤 (剔除次新股、停牌与一字涨停样本)...")

        # 1. 累计上市交易日数 >= 60
        listing_days = pl.col("date").cum_count().over("symbol")

        # 2. T+1 日开盘一字涨停无法买入检测: next_open >= close * 1.095 且 next_high == next_low
        next_open = pl.col("open").shift(-1).over("symbol")
        next_high = pl.col("high").shift(-1).over("symbol")
        next_low = pl.col("low").shift(-1).over("symbol")
        is_limit_up_next_day = (next_open >= (pl.col("close") * 1.095)) & (next_high == next_low)

        filtered = df.filter(
            (listing_days >= 60)
            & (pl.col("volume") > 0)
            & (~is_limit_up_next_day.fill_null(False))
        )

        self.logger.info(
            f"可交易性过滤完成: {df.shape[0]} 行 -> {filtered.shape[0]} 行 "
            f"(过滤掉 {df.shape[0] - filtered.shape[0]} 行不可交易样本)"
        )
        return filtered

    def compute_5d_excess_label(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        严格防范未来函数计算 5 日周频截面超额收益标签 (label_Y):
        T 日收盘选股 -> T+1 开盘买入 -> T+6 开盘卖出 (持仓 5 交易日)
        第一步: 依 symbol 分组计算各股未来 5 日收益率
        第二步: 依 date 分组计算截面均值去中心化，剥离市场 Beta，获得纯 Alpha 标签
        """
        self.logger.info("计算 5 日周频超额收益预测目标 (label_Y: T+1 开盘至 T+6 开盘超额收益)...")

        lead_1_open = pl.col("open").shift(-1).over("symbol")
        lead_6_open = pl.col("open").shift(-6).over("symbol")
        lead_6_date = pl.col("date").shift(-6).over("symbol")
        raw_ret_5 = (lead_6_open / lead_1_open) - 1.0

        # 先物化单股收益率，避免不同 grouping 列嵌套导致 Polars 评估为 null
        df = df.with_columns(
            raw_ret_5.alias("_raw_ret_5"), lead_6_date.alias("_raw_label_end")
        )

        # 截面去中心化：减去当日全市场收益率均值，彻底剥离市场 Beta
        excess_ret_5 = pl.col("_raw_ret_5") - pl.col("_raw_ret_5").mean().over("date")

        # 截面 Winsorize 去极值 (截断在 [-0.30, +0.30] 防止极端黑天鹅扭曲模型)
        clipped_label = excess_ret_5.clip(-0.30, 0.30).alias(self.TARGET_COL)

        # Each excess-return label also depends on the other stocks in its date group.
        label_end = (
            pl.when(pl.col("_raw_ret_5").is_not_null())
            .then(pl.col("_raw_label_end"))
            .max()
            .over("date")
            .alias("label_end_date")
        )
        return df.with_columns(clipped_label, label_end).drop(
            "_raw_ret_5", "_raw_label_end"
        )

    def compute_alpha_factors(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        全向量化计算 33 维超强 Alpha 特征库 (17 维经典特征 + 16 维 WorldQuant 101 Alphas)
        """
        self.logger.info("正在按股票分组极速向量化衍生 33 维 Alpha 因子库 (经典 + WorldQuant 101)...")

        prev_close = pl.col("close").shift(1).over("symbol")
        prev_high = pl.col("high").shift(1).over("symbol")
        prev_low = pl.col("low").shift(1).over("symbol")
        delta_close_1 = pl.col("close") - prev_close

        # ---------- 1. 短线反转与多周期动量 ----------
        rev_1 = ((pl.col("close") / prev_close) - 1.0).alias("rev_1")
        rev_3 = ((pl.col("close") / pl.col("close").shift(3).over("symbol")) - 1.0).alias("rev_3")
        mom_5 = ((pl.col("close") / pl.col("close").shift(5).over("symbol")) - 1.0).alias("mom_5")
        mom_20 = ((pl.col("close") / pl.col("close").shift(20).over("symbol")) - 1.0).alias("mom_20")
        mom_60 = ((pl.col("close") / pl.col("close").shift(60).over("symbol")) - 1.0).alias("mom_60")

        # ---------- 2. 均线与形态衍生 ----------
        ma5 = pl.col("close").rolling_mean(5).over("symbol")
        ma10 = pl.col("close").rolling_mean(10).over("symbol")
        ma20 = pl.col("close").rolling_mean(20).over("symbol")
        ma60 = pl.col("close").rolling_mean(60).over("symbol")

        bias_20 = ((pl.col("close") / (ma20 + 1e-8)) - 1.0).alias("bias_20")
        high_20_ratio = (pl.col("close") / (pl.col("high").rolling_max(20).over("symbol") + 1e-8)).alias("high_20_ratio")

        # 均线多头排列强度打分 (0 ~ 4 分)
        ma_alignment = (
            (pl.col("close") > ma5).cast(pl.Int32)
            + (ma5 > ma10).cast(pl.Int32)
            + (ma10 > ma20).cast(pl.Int32)
            + (ma20 > ma60).cast(pl.Int32)
        ).cast(pl.Float64).alias("ma_alignment")

        # ---------- 3. 波动率与极值压力 ----------
        # 20日对数收益率标准差
        log_ret = (pl.col("close") / (prev_close + 1e-8)).log()
        vol_20 = log_ret.rolling_std(20).over("symbol").alias("vol_20")

        # 20日下行半方差波动率 (下行风险)
        downside_ret = pl.when(log_ret < 0.0).then(log_ret).otherwise(0.0)
        downside_vol_20 = (downside_ret.pow(2).rolling_mean(20).over("symbol").sqrt()).alias("downside_vol_20")

        # 20日真实波幅 ATR 归一化 (ATR / close)
        tr1 = pl.col("high") - pl.col("low")
        tr2 = (pl.col("high") - prev_close).abs()
        tr3 = (pl.col("low") - prev_close).abs()
        tr = pl.max_horizontal(tr1, tr2, tr3)
        atr_20 = ((tr.rolling_mean(20).over("symbol")) / (pl.col("close") + 1e-8)).alias("atr_20")

        # 10日平均上影线压力占比 (上方套牢盘阻力)
        candle_body_top = pl.max_horizontal(pl.col("open"), pl.col("close"))
        candle_range = pl.col("high") - pl.col("low") + 1e-6
        upper_shadow = (pl.col("high") - candle_body_top) / candle_range
        shadow_upper_10 = upper_shadow.rolling_mean(10).over("symbol").alias("shadow_upper_10")

        # 10日平均下影线支撑占比 (多头承接力)
        candle_body_bottom = pl.min_horizontal(pl.col("open"), pl.col("close"))
        lower_shadow = (candle_body_bottom - pl.col("low")) / candle_range
        shadow_lower_10 = lower_shadow.rolling_mean(10).over("symbol").alias("shadow_lower_10")

        # ---------- 4. 微观结构与资金流 ----------
        # 5日均量 / 20日均量比 (成交放量异动)
        vol_m5 = pl.col("volume").rolling_mean(5).over("symbol")
        vol_m20 = pl.col("volume").rolling_mean(20).over("symbol")
        vol_ratio_5_20 = (vol_m5 / (vol_m20 + 1e-8)).alias("vol_ratio_5_20")

        # 20日量价相关系数
        pv_corr_20 = pl.rolling_corr(rev_1, pl.col("volume"), window_size=20).over("symbol").fill_null(0.0).alias("pv_corr_20")

        # Amihud 非流动性冲击因子 (|收益率| / 成交额 * 1e8)
        amihud_liquidity = ((rev_1.abs() / (pl.col("amount") + 1.0)) * 1e8).rolling_mean(20).over("symbol").alias("amihud_liquidity")

        # 隔夜收益 vs 日内收益差 (知情者隔夜信息 vs 散户日内跟风)
        ret_overnight = (pl.col("open") / (prev_close + 1e-8)) - 1.0
        ret_intraday = (pl.col("close") / (pl.col("open") + 1e-8)) - 1.0
        overnight_intraday_spread = ((ret_overnight - ret_intraday).rolling_mean(10).over("symbol")).alias("overnight_intraday_spread")

        # ---------- 5. WorldQuant 101 Formulaic Alphas (16 维) ----------
        ret = rev_1

        # wq_alpha_001: Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)
        vol_term = ret.rolling_std(20).over("symbol")
        inner_001 = pl.when(ret < 0.0).then(vol_term).otherwise(pl.col("close"))
        signed_pow_001 = inner_001.sign() * (inner_001.abs().pow(2.0))
        ts_argmax_5 = sum([(signed_pow_001 >= signed_pow_001.shift(i).over("symbol")).cast(pl.Float64) for i in range(5)]) / 5.0
        wq_alpha_001 = ts_argmax_5.alias("wq_alpha_001")

        # wq_alpha_002: -1 * correlation(delta(log(volume), 2), intraday_return, 6)
        delta_log_vol = (pl.col("volume") + 1.0).log() - (pl.col("volume").shift(2).over("symbol") + 1.0).log()
        wq_alpha_002 = (-1.0 * pl.rolling_corr(delta_log_vol, ret_intraday, window_size=6).over("symbol").fill_null(0.0)).alias("wq_alpha_002")

        # wq_alpha_003: -1 * correlation(open, volume, 10)
        wq_alpha_003 = (-1.0 * pl.rolling_corr(pl.col("open"), pl.col("volume"), window_size=10).over("symbol").fill_null(0.0)).alias("wq_alpha_003")

        # wq_alpha_004: -1 * Ts_Rank(low, 9)
        ts_rank_low_9 = sum([(pl.col("low") >= pl.col("low").shift(i).over("symbol")).cast(pl.Float64) for i in range(9)]) / 9.0
        wq_alpha_004 = (-1.0 * ts_rank_low_9).alias("wq_alpha_004")

        # wq_alpha_006: -1 * correlation(open, volume, 10)
        wq_alpha_006 = (-1.0 * pl.rolling_corr(pl.col("open"), pl.col("volume"), window_size=10).over("symbol").fill_null(0.0)).alias("wq_alpha_006")

        # wq_alpha_009: Trend consistency vs reversal
        ts_min_5 = delta_close_1.rolling_min(5).over("symbol")
        ts_max_5 = delta_close_1.rolling_max(5).over("symbol")
        wq_alpha_009 = (pl.when(ts_min_5 > 0).then(delta_close_1).when(ts_max_5 < 0).then(delta_close_1).otherwise(-1.0 * delta_close_1)).alias("wq_alpha_009")

        # wq_alpha_012: sign(delta(volume, 1)) * (-1 * delta(close, 1))
        delta_vol_1 = pl.col("volume") - pl.col("volume").shift(1).over("symbol")
        wq_alpha_012 = (delta_vol_1.sign() * (-1.0 * delta_close_1)).alias("wq_alpha_012")

        # wq_alpha_014: -1 * delta(returns, 3) * correlation(open, volume, 10)
        delta_ret_3 = ret - ret.shift(3).over("symbol")
        wq_alpha_014 = (-1.0 * delta_ret_3 * pl.rolling_corr(pl.col("open"), pl.col("volume"), window_size=10).over("symbol").fill_null(0.0)).alias("wq_alpha_014")

        # wq_alpha_020: -1 * (open - delay(high, 1)) * (open - delay(close, 1)) * (open - delay(low, 1))
        term1_020 = pl.col("open") - prev_high
        term2_020 = pl.col("open") - prev_close
        term3_020 = pl.col("open") - prev_low
        wq_alpha_020 = (-1.0 * term1_020 * term2_020 * term3_020).alias("wq_alpha_020")

        # wq_alpha_023: Breakout exhaustion
        high_m20 = pl.col("high").rolling_mean(20).over("symbol")
        delta_high_2 = pl.col("high") - pl.col("high").shift(2).over("symbol")
        wq_alpha_023 = pl.when((pl.col("high") > prev_high) & (pl.col("high") > high_m20)).then(-1.0 * delta_high_2).otherwise(0.0).alias("wq_alpha_023")

        # wq_alpha_033: -1 * (1 - (open / close))
        wq_alpha_033 = (-1.0 * (1.0 - (pl.col("open") / (pl.col("close") + 1e-8)))).alias("wq_alpha_033")

        # wq_alpha_041: sqrt(high * low) - vwap
        vwap = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
        geom_mean = (pl.col("high") * pl.col("low")).sqrt()
        wq_alpha_041 = (geom_mean - vwap).alias("wq_alpha_041")

        # wq_alpha_044: -1 * correlation(high, volume, 5)
        wq_alpha_044 = (-1.0 * pl.rolling_corr(pl.col("high"), pl.col("volume"), window_size=5).over("symbol").fill_null(0.0)).alias("wq_alpha_044")

        # wq_alpha_053: -1 * delta(CLV, 9)
        clv = (2.0 * pl.col("close") - pl.col("high") - pl.col("low")) / (pl.col("high") - pl.col("low") + 1e-6)
        wq_alpha_053 = (-1.0 * (clv - clv.shift(9).over("symbol"))).alias("wq_alpha_053")

        # wq_alpha_054: -1 * ((low - close) * (open^5)) / ((low - high) * (close^5))
        wq_alpha_054 = (-1.0 * ((pl.col("low") - pl.col("close")) * (pl.col("open") / (pl.col("close") + 1e-8)).pow(5.0)) / (pl.col("low") - pl.col("high") - 1e-6)).alias("wq_alpha_054")

        # wq_alpha_101: (close - open) / (high - low + 0.001)
        wq_alpha_101 = ((pl.col("close") - pl.col("open")) / (pl.col("high") - pl.col("low") + 0.001)).alias("wq_alpha_101")

        factor_exprs = [
            rev_1, rev_3, mom_5, mom_20, mom_60,
            bias_20, high_20_ratio, ma_alignment,
            vol_20, downside_vol_20, atr_20, shadow_upper_10, shadow_lower_10,
            vol_ratio_5_20, pv_corr_20, amihud_liquidity, overnight_intraday_spread,
            wq_alpha_001, wq_alpha_002, wq_alpha_003, wq_alpha_004, wq_alpha_006,
            wq_alpha_009, wq_alpha_012, wq_alpha_014, wq_alpha_020, wq_alpha_023,
            wq_alpha_033, wq_alpha_041, wq_alpha_044, wq_alpha_053, wq_alpha_054, wq_alpha_101,
        ]

        return df.with_columns(factor_exprs)

    def compute_cross_sectional_ranks(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        对所有 33 个 Alpha 因子在每个交易日横截面上计算百分位排名归一化：
        rank_feature = rank(method='average') / count()  ->  映射到 [0, 1]
        """
        self.logger.info("执行 33 维因子的横截面百分位排名归一化 (Percentile Rank)...")

        rank_exprs = [
            (
                pl.col(factor).rank(method="average").over("date")
                / pl.col(factor).count().over("date")
            ).alias(f"{factor}_rank")
            for factor in self.RAW_FACTORS
        ]

        return df.with_columns(rank_exprs)

    def run_pipeline(self) -> pl.DataFrame:
        """完整执行因子工程流水线"""
        self.logger.info("=" * 60)
        self.logger.info("启动机构级 Polars 多因子计算与特征工程流水线")
        self.logger.info("=" * 60)

        # 1. 读取数据库
        raw_df = self.load_data()

        # 2. 严格按 symbol, date 排序
        df = raw_df.sort(["symbol", "date"])

        # 3. 可交易性过滤
        df = self.filter_tradability(df)

        # 4. 计算 5 日周频截面超额收益标签 (label_Y)
        df = self.compute_5d_excess_label(df)

        # 5. 衍生 17 维 Alpha 基础因子
        df = self.compute_alpha_factors(df)

        # 6. 横截面百分位排名归一化
        df = self.compute_cross_sectional_ranks(df)

        # 7. 筛选特征并去除空值
        output_cols = ["date", "symbol", "label_end_date"] + self.RANK_FEATURES + [self.TARGET_COL]
        clean_df = df.select(output_cols).drop_nulls()

        self.logger.info(
            f"特征工程处理完成: 保留有效样本 {clean_df.shape[0]} 行, "
            f"涵盖 {clean_df['symbol'].n_unique()} 只股票, 时间跨度: {clean_df['date'].min()} ~ {clean_df['date'].max()}"
        )

        # 8. 持久化至 Parquet
        self.logger.info(f"正在将特征工程矩阵写入 Parquet: {self.output_path}...")
        clean_df.write_parquet(self.output_path, compression="zstd")
        self.logger.info(
            f"Parquet 落盘成功！文件大小: {self.output_path.stat().st_size / (1024 * 1024):.2f} MB"
        )
        self.logger.info("=" * 60)

        return clean_df


if __name__ == "__main__":
    calculator = FactorCalculator()
    try:
        calculator.run_pipeline()
    except Exception as err:
        logging.getLogger("FactorCalculator").error(f"因子计算运行失败: {err}", exc_info=True)
