"""
模型训练与回测验证模块 (ModelTrainer) - 机构级评估升级版
===================================================
1. 绝对杜绝未来穿越:
   - 严格按时间序列前向切分 (Walk-Forward Split): Train (2020-2023) / Test (2024-2026)
2. LightGBM 回归模型防过拟合:
   - 针对金融微弱信噪比，限制树深 max_depth=4, num_leaves=15, 学习率 0.03
   - 特征与样本双 80% 子采样，配合 L1/L2 正则化
3. 机构级评估体系 (双重验证):
   - 信号强度: Rank IC、日度截面 Mean IC、ICIR、t 统计量、胜率
   - 真实赚钱能力: 五分组分层回测 (Quintile Backtest)、多空对冲 (Long-Short Q5-Q1) 年化收益、夏普比率、最大回撤与单调性分析
4. 生产级权重落盘: 打印特征增益重要性 (Gain Importance) 并持久化至 src/lgb_model.txt
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union, Any

# 确保根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import spearmanr

import src.config as cfg


# 路径配置
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
SRC_DIR = PROJECT_ROOT / "src"
CACHE_DIR = DATA_DIR / "factor_cache"

DEFAULT_PARQUET_PATH = CACHE_DIR / "features.parquet"
DEFAULT_MODEL_PATH = SRC_DIR / "lgb_model.txt"
DEFAULT_LOG_PATH = LOGS_DIR / "model_run.log"


def setup_logger(log_file: Path = DEFAULT_LOG_PATH, name: str = "ModelTrainer") -> logging.Logger:
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


class ModelTrainer:
    """
    量化多因子 LightGBM 回归训练器与分层回测引擎
    """

    FEATURE_COLS = cfg.RANK_FEATURES
    TARGET_COL = cfg.TARGET_COL

    def __init__(
        self,
        parquet_path: Optional[Union[str, Path]] = None,
        model_path: Optional[Union[str, Path]] = None,
        split_date: str = cfg.TRAIN_TEST_SPLIT_DATE,
        feature_cols: Optional[List[str]] = None,
        target_col: str = TARGET_COL,
        hyperparams: Optional[Dict] = None,
    ):
        self.logger = setup_logger(name="ModelTrainer")
        self.parquet_path = Path(parquet_path) if parquet_path else DEFAULT_PARQUET_PATH
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.split_date = split_date
        self.feature_cols = feature_cols or self.FEATURE_COLS
        self.target_col = target_col
        self.hyperparams = hyperparams or cfg.LIGHTGBM_PARAMS

        self.model_path.parent.mkdir(parents=True, exist_ok=True)

    def load_features(self) -> pl.DataFrame:
        """从 Parquet 读取特征工程数据集"""
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"特征文件不存在: {self.parquet_path}，请先执行 factor_calc.py！")

        self.logger.info(f"正在从 [{self.parquet_path}] 读取多因子特征矩阵...")
        df = pl.read_parquet(self.parquet_path)

        if df.is_empty():
            raise ValueError(f"特征矩阵 {self.parquet_path} 内容为空！")

        self.logger.info(
            f"成功加载特征矩阵: 共 {df.shape[0]} 条样本, {df['symbol'].n_unique()} 只股票, "
            f"特征维度: {len(self.feature_cols)}, 时间区间: {df['date'].min()} ~ {df['date'].max()}"
        )
        return df

    def split_train_test(
        self, df: pl.DataFrame
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Split chronologically and exclude training labels that reach the holdout."""
        if "label_end_date" not in df.columns:
            raise ValueError("Regenerate the feature cache with python main.py --calc.")
        self.logger.info(f"执行时序单向切分 (Walk-Forward Split), 切分点: {self.split_date}")

        train_df = df.filter(pl.col("date") < self.split_date)
        test_df = df.filter(pl.col("date") >= self.split_date)

        if train_df.is_empty() or test_df.is_empty():
            unique_dates = df["date"].unique().sort()
            split_idx = int(len(unique_dates) * 0.75)
            self.split_date = unique_dates[split_idx]
            self.logger.info(f"自适应调整切分基准至 75% 处日期: {self.split_date}")
            train_df = df.filter(pl.col("date") < self.split_date)
            test_df = df.filter(pl.col("date") >= self.split_date)

        train_df = train_df.filter(pl.col("label_end_date") < self.split_date)
        if train_df.is_empty() or test_df.is_empty():
            raise ValueError("Not enough date coverage for the training and holdout windows.")

        self.logger.info(
            f"训练集 (In-Sample):  {train_df.shape[0]} 条样本 | 区间: {train_df['date'].min()} ~ {train_df['date'].max()}"
        )
        self.logger.info(
            f"测试集 (Out-of-Sample): {test_df.shape[0]} 条样本 | 区间: {test_df['date'].min()} ~ {test_df['date'].max()}"
        )
        return train_df, test_df

    def train_model(
        self, train_df: pl.DataFrame, num_boost_round: int = cfg.NUM_BOOST_ROUND
    ) -> lgb.Booster:
        """训练 LightGBM GBDT 回归模型"""
        self.logger.info("=" * 60)
        self.logger.info("开始训练 LightGBM 梯度提升回归模型...")
        self.logger.info(f"特征列表 ({len(self.feature_cols)} 个): {self.feature_cols}")
        self.logger.info(f"超参数配置: {self.hyperparams}")

        X_train = train_df.select(self.feature_cols).to_numpy()
        y_train = train_df[self.target_col].to_numpy()

        train_data = lgb.Dataset(
            data=X_train,
            label=y_train,
            feature_name=self.feature_cols,
            free_raw_data=False,
        )

        booster: lgb.Booster = lgb.train(
            params=self.hyperparams,
            train_set=train_data,
            num_boost_round=num_boost_round,
        )

        self.logger.info("LightGBM 模型离线训练圆满完成！")
        return booster

    def evaluate_ic_and_quintiles(
        self, booster: lgb.Booster, test_df: pl.DataFrame
    ) -> Dict[str, Any]:
        """
        机构级双重评估体系：
        1. 秩相关分析 (Rank IC / ICIR / t-stat)
        2. 五分组分层回测 (Quintile Analysis: Q1~Q5 多空对冲与单调性)
        """
        self.logger.info("=" * 60)
        self.logger.info("执行样本外 (OOS) 机构级回测评估体系...")

        X_test = test_df.select(self.feature_cols).to_numpy()
        y_test = test_df[self.target_col].to_numpy()

        # 模型预测
        y_pred = booster.predict(X_test)

        # 1. 整体 Rank IC
        overall_rank_ic, p_value = spearmanr(y_pred, y_test)

        # 构造评估 DataFrame
        eval_df = test_df.select(["date", "symbol", self.target_col]).with_columns(
            pl.Series("pred_score", y_pred)
        )

        # 2. 每日截面 Rank IC 与五分组收益
        daily_ics = []
        quintile_returns = {1: [], 2: [], 3: [], 4: [], 5: []}
        long_short_returns = []

        # 按日期循环统计
        for (date_val,), group in eval_df.group_by(["date"]):
            if len(group) >= 10:  # 截面至少 10 只股票保证统计有效
                score_arr = group["pred_score"].to_numpy()
                true_arr = group[self.target_col].to_numpy()

                # 日度 IC
                ic_val, _ = spearmanr(score_arr, true_arr)
                if not np.isnan(ic_val):
                    daily_ics.append(float(ic_val))

                # 五分组分层
                group_sorted = group.sort("pred_score")
                n_stocks = len(group_sorted)
                q_size = n_stocks // 5

                q1_ret = group_sorted.slice(0, q_size)[self.target_col].mean()
                q2_ret = group_sorted.slice(q_size, q_size)[self.target_col].mean()
                q3_ret = group_sorted.slice(q_size * 2, q_size)[self.target_col].mean()
                q4_ret = group_sorted.slice(q_size * 3, q_size)[self.target_col].mean()
                q5_ret = group_sorted.slice(q_size * 4, n_stocks - q_size * 4)[self.target_col].mean()

                if q1_ret is not None and q5_ret is not None:
                    quintile_returns[1].append(q1_ret)
                    quintile_returns[2].append(q2_ret)
                    quintile_returns[3].append(q3_ret)
                    quintile_returns[4].append(q4_ret)
                    quintile_returns[5].append(q5_ret)
                    long_short_returns.append(q5_ret - q1_ret)

        daily_ics_arr = np.array(daily_ics)
        mean_ic = float(np.mean(daily_ics_arr)) if len(daily_ics_arr) > 0 else 0.0
        std_ic = float(np.std(daily_ics_arr, ddof=1)) if len(daily_ics_arr) > 1 else 1e-6
        icir = float(mean_ic / std_ic) if std_ic > 0 else 0.0
        win_rate = float(np.mean(daily_ics_arr > 0)) * 100.0
        t_stat = icir * np.sqrt(len(daily_ics_arr)) if len(daily_ics_arr) > 0 else 0.0

        # 分组年化收益与多空夏普
        q_means = [float(np.mean(quintile_returns[i])) * 50 * 100 for i in range(1, 6)]  # 周频年化约 50 期
        ls_arr = np.array(long_short_returns)
        ls_annual_ret = float(np.mean(ls_arr) * 50 * 100) if len(ls_arr) > 0 else 0.0
        ls_std = float(np.std(ls_arr, ddof=1)) if len(ls_arr) > 1 else 1e-6
        ls_sharpe = float((np.mean(ls_arr) / ls_std) * np.sqrt(50)) if ls_std > 0 else 0.0

        # 计算多空最大回撤
        cum_ls = np.cumprod(1.0 + ls_arr) if len(ls_arr) > 0 else np.array([1.0])
        peak = np.maximum.accumulate(cum_ls)
        drawdowns = (peak - cum_ls) / peak
        max_drawdown = float(np.max(drawdowns)) * 100.0 if len(drawdowns) > 0 else 0.0

        # 输出机构级评估研报
        self.logger.info("-------------------- 1. IC 信号预测能力 --------------------")
        self.logger.info(f"测试集样本量 (OOS Samples):   {len(y_test)}")
        self.logger.info(f"有效评估交易日数 (Days):       {len(daily_ics_arr)}")
        self.logger.info(f"整体 Rank IC (Overall):       {overall_rank_ic:+.4f} (p-value: {p_value:.2e})")
        self.logger.info(f"日均截面 Rank IC (Mean IC):   {mean_ic:+.4f}")
        self.logger.info(f"日度 IC 波动率 (IC Std):      {std_ic:.4f}")
        self.logger.info(f"信息比率 (Rank ICIR):         {icir:+.4f}")
        self.logger.info(f"IC 胜率 (IC > 0 占比):        {win_rate:.2f}%")
        self.logger.info(f"t 统计量 (t-statistic):       {t_stat:+.2f}")

        self.logger.info("-------------------- 2. 五分组分层回测 --------------------")
        self.logger.info(f"Q1 组 (最看空 20%) 年化超额:  {q_means[0]:+.2f}%")
        self.logger.info(f"Q2 组              年化超额:  {q_means[1]:+.2f}%")
        self.logger.info(f"Q3 组              年化超额:  {q_means[2]:+.2f}%")
        self.logger.info(f"Q4 组              年化超额:  {q_means[3]:+.2f}%")
        self.logger.info(f"Q5 组 (最看多 20%) 年化超额:  {q_means[4]:+.2f}%")
        self.logger.info(f"多空对冲 (Q5 - Q1) 年化收益:  {ls_annual_ret:+.2f}%")
        self.logger.info(f"多空对冲夏普比率 (Sharpe):    {ls_sharpe:+.2f}")
        self.logger.info(f"多空对冲最大回撤 (Max DD):    {max_drawdown:.2f}%")
        self.logger.info("-----------------------------------------------------------")

        return {
            "overall_rank_ic": overall_rank_ic,
            "mean_ic": mean_ic,
            "icir": icir,
            "win_rate": win_rate,
            "t_stat": t_stat,
            "ls_annual_ret": ls_annual_ret,
            "ls_sharpe": ls_sharpe,
            "max_drawdown": max_drawdown,
            "q_means": q_means,
        }

    def print_feature_importance(self, booster: lgb.Booster) -> None:
        """打印特征重要性评估 (按 Gain 增益降序排序)"""
        self.logger.info("=" * 60)
        self.logger.info("特征重要性评估 (Feature Importance - Sorted by Gain):")
        importance_split = booster.feature_importance(importance_type="split")
        importance_gain = booster.feature_importance(importance_type="gain")
        total_gain = float(np.sum(importance_gain)) + 1e-8

        records = []
        for name, split, gain in zip(self.feature_cols, importance_split, importance_gain):
            records.append({
                "feature": name,
                "split": int(split),
                "gain": float(gain),
                "gain_pct": float(gain / total_gain * 100.0),
            })

        records = sorted(records, key=lambda x: x["gain"], reverse=True)

        self.logger.info(f"{'特征名称':<25} | {'分裂次数 (Split)':<15} | {'增益贡献 (Gain)':<15} | {'增益占比 (%)':<10}")
        self.logger.info("-" * 75)
        for r in records:
            self.logger.info(
                f"{r['feature']:<25} | {r['split']:<18} | {r['gain']:<18.2f} | {r['gain_pct']:<10.2f}%"
            )
        self.logger.info("-" * 75)

    def save_model(self, booster: lgb.Booster) -> None:
        """持久化模型权重 (使用 Python 文件流写入，规避 Windows C++ 中文路径编码问题)"""
        self.logger.info(f"正在持久化生产模型权重至: {self.model_path}...")
        model_str = booster.model_to_string()
        with open(self.model_path, "w", encoding="utf-8") as f:
            f.write(model_str)
        self.logger.info(
            f"模型权重已成功写入: {self.model_path} (文件大小: {self.model_path.stat().st_size / 1024:.2f} KB)"
        )

    def run_pipeline(self, num_boost_round: int = cfg.NUM_BOOST_ROUND) -> lgb.Booster:
        """执行端到端训练与评估"""
        self.logger.info("=" * 60)
        self.logger.info("启动 LightGBM 机构级量化模型训练与评估流水线")
        self.logger.info("=" * 60)

        df = self.load_features()
        train_df, test_df = self.split_train_test(df)
        booster = self.train_model(train_df, num_boost_round=num_boost_round)
        self.evaluate_ic_and_quintiles(booster, test_df)
        self.print_feature_importance(booster)
        self.save_model(booster)

        self.logger.info("=" * 60)
        self.logger.info("模型训练与回测验证流水线全部圆满完成！")
        self.logger.info("=" * 60)

        return booster


if __name__ == "__main__":
    trainer = ModelTrainer()
    try:
        trainer.run_pipeline()
    except Exception as err:
        logging.getLogger("ModelTrainer").error(f"模型训练运行失败: {err}", exc_info=True)
