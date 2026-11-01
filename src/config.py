"""
全局配置模块 (config.py)
===================================================
集中管理 A 股中低频量化研究系统的核心参数配置：
- 路径配置 (数据、日志、缓存、模型)
- 股票池配置 (中证 500 / 000905)
- 调仓周期与预测目标 (5日持有周频轮动 + 截面超额收益 Alpha)
- 33 维多因子 Alpha 特征库 (包含 17 维经典因子 + 16 维 WorldQuant 101 真实实盘因子)
- 模型超参数与时间切分基准
- 组合资金配置 (10万元标准账户、Top-5等权轮动、100股一手、滑点设置)
"""

import os
from pathlib import Path

# 根目录与基础路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
SRC_DIR = PROJECT_ROOT / "src"
CACHE_DIR = DATA_DIR / "factor_cache"

# 文件路径配置
DB_PATH = DATA_DIR / "market_data.db"
FEATURES_PARQUET_PATH = CACHE_DIR / "features.parquet"
MODEL_PATH = SRC_DIR / "lgb_model.txt"
DATA_FETCH_LOG_PATH = LOGS_DIR / "data_fetch.log"
MODEL_RUN_LOG_PATH = LOGS_DIR / "model_run.log"

# 数据与基准参数
BENCHMARK_INDEX = "000905"  # 中证 500
START_DATE = "20200101"     # 历史数据起点
PRICE_ADJUST = "qfq"        # 前复权
HOLD_DAYS = 5               # 周频轮动：持仓 5 个交易日 (T+1 开盘买入至 T+6 开盘卖出)

# 33 维超强 Alpha 因子库定义 (经典特征 + WorldQuant 101 实盘公式)
RAW_FACTORS = [
    # ---------- 第一部分: 经典量价与微观结构 (17 维) ----------
    "rev_1",                    # 1日超跌反转
    "rev_3",                    # 3日反转
    "mom_5",                    # 5日动量 (周度)
    "mom_20",                   # 20日动量 (月度)
    "mom_60",                   # 60日动量 (季度)
    "bias_20",                  # 20日乖离率 (距离MA20的百分比)
    "high_20_ratio",            # 突破形态 (距离过去20日最高价距离)
    "ma_alignment",             # 均线多头排列强度 (0~4分)
    "vol_20",                   # 20日对数收益率波动率
    "downside_vol_20",          # 20日下行半方差波动率 (下跌风险度量)
    "atr_20",                   # 20日真实波幅归一化 (ATR/close)
    "shadow_upper_10",          # 10日平均上影线压力占比 (套牢盘抛压)
    "shadow_lower_10",          # 10日平均下影线支撑占比 (多头承接力)
    "vol_ratio_5_20",           # 5日均量 / 20日均量比 (成交异动放量)
    "pv_corr_20",               # 20日量价相关系数
    "amihud_liquidity",         # Amihud 非流动性冲击因子
    "overnight_intraday_spread",# 隔夜收益 vs 日内收益差 (知情者隔夜 vs 散户日内跟风)

    # ---------- 第二部分: WorldQuant 101 Formulaic Alphas (16 维) ----------
    "wq_alpha_001",             # Alpha#1: 下跌波动加权时序动量 (Ts_ArgMax SignedPower)
    "wq_alpha_002",             # Alpha#2: 成交量增速与日内收益负相关度 (-corr(delta(log(vol)), intraday))
    "wq_alpha_003",             # Alpha#3: 开盘价与成交量负相关 (-corr(open, volume, 10))
    "wq_alpha_004",             # Alpha#4: 最低价时序排位反转 (-Ts_Rank(low, 9))
    "wq_alpha_006",             # Alpha#6: 开盘量价背离 (-corr(open, volume, 10))
    "wq_alpha_009",             # Alpha#9: 连续趋势加速与反转判断
    "wq_alpha_012",             # Alpha#12: 成交量变化符号与价格变化反向
    "wq_alpha_014",             # Alpha#14: 收益率加速度与开盘成交量相关
    "wq_alpha_020",             # Alpha#20: 开盘跳空多重价差组合反转 (open-high, open-close, open-low)
    "wq_alpha_023",             # Alpha#23: 突破20日最高价后的短期枯竭回踩
    "wq_alpha_033",             # Alpha#33: 日内趋势强度反转 (-1 * (1 - open/close))
    "wq_alpha_041",             # Alpha#41: 几何极值中枢相对 VWAP 偏差 (sqrt(high*low) - vwap)
    "wq_alpha_044",             # Alpha#44: 最高价与成交量排名负相关 (-corr(high, rank(volume), 5))
    "wq_alpha_053",             # Alpha#53: 收盘区间位置指标 CLV 的 9 日变化率
    "wq_alpha_054",             # Alpha#54: 非线性开收盘高阶形态比率
    "wq_alpha_101",             # Alpha#101: 实体K线占全天振幅比例 ((close-open) / (high-low+0.001))
]

RANK_FEATURES = [f"{f}_rank" for f in RAW_FACTORS]
TARGET_COL = "label_Y"          # 5日截面超额收益率 (Alpha)

# 模型训练与时序验证配置
TRAIN_TEST_SPLIT_DATE = "2024-01-01"  # 样本内 (2020-2023) / 样本外 (2024-2026)
LIGHTGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "max_depth": 4,             # 限制树深防止噪音过拟合
    "num_leaves": 15,
    "learning_rate": 0.03,
    "feature_fraction": 0.8,    # 80% 特征子采样
    "bagging_fraction": 0.8,    # 80% 样本子采样
    "bagging_freq": 1,
    "min_child_samples": 30,    # 叶节点最小样本量
    "reg_alpha": 0.1,           # L1 正则
    "reg_lambda": 1.0,          # L2 正则
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": -1,
}
NUM_BOOST_ROUND = 300

# 组合投资与资金规划参数
TOP_N_PICKS = 5                 # 每日预测优选股票只数 (等权配置，每只占 20% 仓位)
TOTAL_BUDGET = 100000.0         # 模拟盘标准账户本金 (100,000 元)
STOP_LOSS_PCT = 0.05            # 严格止损比例 (-5%)
TAKE_PROFIT_PCT = 0.08          # 建议止盈比例 (+8%)
TRANSACTION_FEE = 0.0003        # 交易佣金 (万三)
STAMP_DUTY = 0.0005             # 印花税 (卖出千分之0.5)
SLIPPAGE = 0.001                # 双向滑点 (千一)
