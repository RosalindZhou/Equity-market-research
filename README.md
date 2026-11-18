# A股中低频选股系统

基于 **LightGBM + 33 维 Alpha 因子** 的 A 股周频轮动量化研究系统。覆盖从行情抓取、因子计算、模型训练、信号预测、回测、模拟盘到 Web 控制台与微信推送的完整链路，数据落地在本地 SQLite，不依赖任何付费行情源。

> 本项目仅用于量化方法与工程实现的个人研究，不构成任何投资建议。

## 一、系统结构

```
A股中低频选股系统/
├── main.py                  # 全流程总控入口（命令行参数分发）
├── daily_runner.py          # 每日盘后流水线执行器（供计划任务调用）
├── setup_task.py            # 一键注册 Windows 计划任务（每日自动跑）
├── run_daily.bat            # 手动触发每日流水线
├── run_web.bat              # 手动启动 Web 控制台
├── src/
│   ├── config.py            # 全局参数：股票池、因子库、模型超参、资金与滑点
│   ├── data_fetcher.py      # AkShare 行情抓取 + SQLite 增量入库（前复权）
│   ├── factor_calc.py       # 33 维因子计算与截面标准化
│   ├── model_train.py       # LightGBM 训练 + 时序切分 + IC 评估
│   ├── predictor.py         # 每日预测与 Top-N 选股信号
│   ├── backtest.py          # 区间回测、滚动调仓、止损止盈、收益归因
│   ├── paper_trader.py      # 模拟盘：持仓、成本、滑点、净值曲线（SQLite 持久化）
│   ├── report_generator.py  # 静态 HTML 日报生成
│   ├── notifier.py          # 多通道推送（PushPlus / Server酱 / 企业微信 / 钉钉 / 飞书）
│   └── web_app.py           # Flask 控制台与 API
├── templates/index.html     # Web 控制台前端
├── data/                    # 行情数据库与因子缓存（不入库）
├── logs/                    # 运行日志
└── reports/                 # 生成的 HTML 日报
```

## 二、策略逻辑

- **股票池**：中证 500（`000905`），起点 2020-01-01，前复权。
- **预测目标**：未来 5 个交易日的**截面超额收益**（相对股票池均值），即"中低频"周频轮动，而非日内高频。
- **因子库（33 维）**：
  - 经典量价与微观结构 17 维：1/3 日反转、5/20/60 日动量、20 日乖离、20 日突破距离、均线多头排列、波动率与下行半方差、ATR、上下影线压力/支撑、量比异动、量价相关、Amihud 非流动性、隔夜-日内收益差。
  - WorldQuant 101 Formulaic Alphas 16 维：Alpha#1/#2/#3 等经实盘验证的公式化因子。
- **模型**：LightGBM 回归，`TimeSeriesSplit` 滚动窗口验证，避免未来函数。
- **组合**：10 万元标准账户，Top-5 等权，100 股一手，含滑点与手续费，T+1 开盘买入、T+6 开盘卖出。

## 三、快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置推送通道（可选）
copy .env.example .env    # 填入 PUSHPLUS_TOKEN / SERVERCHAN_KEY 等

# 3. 首次全流程：抓数据 -> 算因子 -> 训练 -> 预测
python main.py --fetch
python main.py --calc
python main.py --train
python main.py --predict
```

## 四、命令行用法

| 命令 | 作用 |
|---|---|
| `python main.py` | 执行每日盘后完整流水线（更新行情 → 因子 → 预测 → 推送） |
| `--fetch` | 仅增量抓取行情并入库 |
| `--calc` | 仅计算因子并生成特征缓存 |
| `--train` | 重新训练 LightGBM 并输出 IC / 分组收益评估 |
| `--predict` | 仅用最新特征预测并推送今日选股 |
| `--backtest --start 2024-01-01 --end 2026-10-22` | 区间回测（`--no_stop` 可关闭盘中止损止盈） |
| `--paper --paper-action init\|status\|step` | 模拟盘初始化 / 查看持仓 / 推进一日 |
| `--web --port 5000` | 启动 Flask 交互式控制台 |
| `--test-notify` | 测试微信推送通道连通性 |

## 五、自动化

`setup_task.py` 会把 `daily_runner.py` 注册为 Windows 计划任务（已处理中文路径与编码问题），收盘后自动跑完"更新 → 预测 → 推送 → 生成日报"，无需人工介入。

## 六、Web 控制台

`python main.py --web` 后访问 `http://127.0.0.1:5000`：

- 实时持仓、净值曲线、当日信号
- 一键买入 / 卖出 / 推进一日 / 重置模拟盘
- 内置回测面板，可直接在页面上跑区间回测
- 历史日报浏览（`/reports/<filename>`）

## 七、说明

- 所有行情数据与模型产物均在本地，仓库内不含任何 data / db / parquet 文件。
- `.env` 存推送 Token，已被 `.gitignore` 排除。
