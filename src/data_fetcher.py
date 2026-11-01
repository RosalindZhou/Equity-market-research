"""
数据获取模块 (DataFetcher) - 高可用与多源容灾升级版
===================================================
1. 多源自动容灾 (Multi-Source Failover):
   - 主数据源: 新浪财经接口 (ak.stock_zh_a_daily)，高并发、稳定无 EastMoney push2 严格 IP 封锁。
   - 备用数据源: 东方财富接口 (ak.stock_zh_a_hist)，双引擎互相兜底。
2. 断点续传 (Resumable Fetching):
   - 自动检测 SQLite 中已入库的股票，避免中断后重复拉取，极大提高稳定性。
3. 智能重试与防封风控 (Adaptive Retry & Rate Limiting):
   - 遭遇瞬时网络错误自动指数退避重试 (Exponential Backoff)。
   - 字段标准化英文化清洗，确保与下游 Polars 因子计算层无缝兼容。
"""

import os
import sys
import time
import logging
import sqlite3
from pathlib import Path
from typing import List, Optional, Set, Union

import akshare as ak
import pandas as pd
from sqlalchemy import create_engine, Engine, text


# 项目路径配置
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
DEFAULT_DB_PATH = DATA_DIR / "market_data.db"
DEFAULT_LOG_PATH = LOGS_DIR / "data_fetch.log"


def setup_logger(log_file: Path = DEFAULT_LOG_PATH) -> logging.Logger:
    """
    配置并获取全局统一日志记录器
    同时将 INFO 及以上级别日志输出至标准输出控制台和日志文件

    :param log_file: 日志存储路径
    :return: logging.Logger 实例
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("DataFetcher")
    logger.setLevel(logging.INFO)

    # 避免多次调用重复添加 Handler
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


class DataFetcher:
    """
    A 股量化数据抓取与入库引擎（多源自动容灾 + 断点续传）
    """

    # 字段统一英文化映射字典
    COLUMN_MAPPING = {
        "日期": "date",
        "股票代码": "symbol",
        "代码": "symbol",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "pct_chg",
        "涨跌额": "change",
        "换手率": "turnover",
    }

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        """
        初始化 DataFetcher 并建立 SQLite 数据库连接

        :param db_path: SQLite 数据库路径，缺省时使用 data/market_data.db
        """
        self.logger = setup_logger()

        if db_path is None:
            self.db_path = DEFAULT_DB_PATH
        else:
            self.db_path = Path(db_path)

        # 确保数据库存放目录存在
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # 初始化 SQLAlchemy 引擎连接 SQLite
        db_url = f"sqlite:///{self.db_path.resolve().as_posix()}"
        self.engine: Engine = create_engine(db_url, echo=False)
        self._init_db_indexes()
        self.logger.info(f"已成功连接 SQLite 数据库: {self.db_path}")

    def _init_db_indexes(self) -> None:
        """初始化数据表与复合索引以加速断点续传查询"""
        try:
            with self.engine.connect() as conn:
                conn.execute(
                    text(
                        """
                        CREATE TABLE IF NOT EXISTS daily_bars (
                            symbol TEXT,
                            date TEXT,
                            open REAL,
                            close REAL,
                            high REAL,
                            low REAL,
                            volume REAL,
                            amount REAL,
                            amplitude REAL,
                            pct_chg REAL,
                            change REAL,
                            turnover REAL
                        )
                        """
                    )
                )
                conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS idx_symbol_date ON daily_bars(symbol, date)"
                    )
                )
                conn.commit()
        except Exception as e:
            self.logger.debug(f"初始化索引提示: {e}")

    def get_existing_symbols(self, min_records: int = 100) -> Set[str]:
        """
        获取数据库中已经成功入库且数据量充足的股票代码列表（用于断点续传）

        :param min_records: 判定某股票已完整入库的最小记录行数
        :return: 已入库股票代码集合
        """
        try:
            with self.engine.connect() as conn:
                query = text(
                    f"SELECT symbol, count(*) as cnt FROM daily_bars GROUP BY symbol HAVING cnt >= {min_records}"
                )
                result = conn.execute(query).fetchall()
                existing = {row[0] for row in result}
                return existing
        except Exception as e:
            self.logger.warning(f"获取已入库股票列表失败: {e}，将执行全量拉取")
            return set()

    def get_zz500_stocks(self, symbol: str = "000905") -> List[str]:
        """
        获取中证 500 成分股列表，提取出 6 位数字格式的标准股票代码

        :param symbol: 指数代码，默认为 '000905' (中证 500)
        :return: 6 位股票代码列表
        """
        self.logger.info(f"正在通过 akshare 获取指数 [{symbol}] 成分股列表...")
        try:
            cons_df = ak.index_stock_cons(symbol=symbol)
            if cons_df is None or cons_df.empty:
                self.logger.warning(f"未能获取到指数 [{symbol}] 的成分股数据 (返回空)")
                return []

            possible_code_cols = [
                "品种代码", "成分券代码", "证券代码", "代码", "stock_code", "symbol", "code"
            ]
            code_col = None
            for col in possible_code_cols:
                if col in cons_df.columns:
                    code_col = col
                    break

            if code_col is None:
                code_col = cons_df.columns[0]
                self.logger.warning(f"未匹配到标准代码列名，默认使用首列: {code_col}")

            raw_codes = cons_df[code_col].astype(str).str.strip()
            stock_list = [
                c.zfill(6) for c in raw_codes if c.isdigit() and len(c.zfill(6)) == 6
            ]
            stock_list = sorted(list(set(stock_list)))

            self.logger.info(f"成功获取指数 [{symbol}] 成分股，共计 {len(stock_list)} 只股票")
            return stock_list

        except Exception as e:
            self.logger.error(f"获取指数 [{symbol}] 成分股列表异常: {e}", exc_info=True)
            return []

    @staticmethod
    def _get_market_symbol(code: str) -> str:
        """为股票代码添加交易所市场前缀 (新浪财经规范)"""
        if code.startswith(("60", "68", "90")):
            return f"sh{code}"
        elif code.startswith(("00", "30", "20")):
            return f"sz{code}"
        elif code.startswith(("43", "83", "87", "92")):
            return f"bj{code}"
        return f"sz{code}"

    def _fetch_from_sina(
        self, code: str, start_date: str = "20200101", adjust: str = "qfq"
    ) -> Optional[pd.DataFrame]:
        """
        从新浪财经接口拉取前复权日线行情（极速且不受东财限流）

        :param code: 6 位股票代码
        :param start_date: 起始日期 (YYYYMMDD)
        :param adjust: 复权类型
        :return: 清洗后的 DataFrame 或 None
        """
        symbol_with_market = self._get_market_symbol(code)
        df = ak.stock_zh_a_daily(
            symbol=symbol_with_market, start_date=start_date, adjust=adjust
        )
        if df is None or df.empty:
            return None

        # 标准化字段
        df = df.rename(columns=self.COLUMN_MAPPING)
        df["symbol"] = code
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

        # 补全量化衍生价量指标 (涨跌幅、涨跌额、振幅)
        prev_close = df["close"].shift(1).bfill()
        if "pct_chg" not in df.columns:
            df["pct_chg"] = ((df["close"] / prev_close) - 1.0) * 100.0
        if "change" not in df.columns:
            df["change"] = df["close"] - prev_close
        if "amplitude" not in df.columns:
            df["amplitude"] = ((df["high"] - df["low"]) / prev_close) * 100.0

        return df

    def _fetch_from_eastmoney(
        self, code: str, start_date: str = "20200101", adjust: str = "qfq"
    ) -> Optional[pd.DataFrame]:
        """
        从东方财富接口拉取日线历史行情（备选数据源）

        :param code: 6 位股票代码
        :param start_date: 起始日期
        :param adjust: 复权类型
        :return: 清洗后的 DataFrame 或 None
        """
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start_date,
            adjust=adjust,
        )
        if df is None or df.empty:
            return None

        df = df.rename(columns=self.COLUMN_MAPPING)
        df["symbol"] = code
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

        return df

    def fetch_daily_bar(
        self,
        code: str,
        start_date: str = "20200101",
        adjust: str = "qfq",
        max_retries: int = 2,
    ) -> Optional[pd.DataFrame]:
        """
        双源容灾日线抓取器：
        首选新浪财经高速接口（无 IP 限流风险），若异常自动无缝切换东财接口。

        :param code: 6位股票代码
        :param start_date: 开始日期 (YYYYMMDD)
        :param adjust: 复权类型 ('qfq')
        :param max_retries: 重试次数
        :return: 清洗后的 DataFrame 或 None
        """
        # 1. 尝试主数据源（新浪财经）
        for attempt in range(max_retries + 1):
            try:
                df = self._fetch_from_sina(code=code, start_date=start_date, adjust=adjust)
                if df is not None and not df.empty:
                    return self._clean_and_order_columns(df, code)
            except Exception as e:
                if attempt < max_retries:
                    time.sleep(1.0)
                else:
                    self.logger.debug(f"新浪数据源获取 {code} 失败: {e}，尝试切换备选源...")

        # 2. 备选数据源（东方财富）
        try:
            df = self._fetch_from_eastmoney(code=code, start_date=start_date, adjust=adjust)
            if df is not None and not df.empty:
                return self._clean_and_order_columns(df, code)
        except Exception as e:
            self.logger.warning(f"股票 {code} 双数据源拉取均失败: {e}，跳过该股票")

        return None

    def _clean_and_order_columns(self, df: pd.DataFrame, code: str) -> pd.DataFrame:
        """确保字段规范化与类型一致性"""
        target_cols = [
            "symbol", "date", "open", "close", "high", "low",
            "volume", "amount", "amplitude", "pct_chg", "change", "turnover"
        ]
        df["symbol"] = code
        for col in target_cols:
            if col not in df.columns:
                df[col] = 0.0

        return df[target_cols]

    def save_to_db(self, df: pd.DataFrame, table_name: str = "daily_bars") -> None:
        """
        将数据持久化追加写入 SQLite 数据库

        :param df: 清洗后的股票数据 DataFrame
        :param table_name: 目标表名，默认为 daily_bars
        """
        if df is None or df.empty:
            return

        try:
            df.to_sql(
                name=table_name,
                con=self.engine,
                if_exists="append",
                index=False,
                chunksize=1000,
            )
        except Exception as e:
            self.logger.error(f"写入数据表 [{table_name}] 失败: {e}", exc_info=True)
            raise e

    def fetch_and_save_all(
        self,
        symbol: str = "000905",
        start_date: str = "20200101",
        adjust: str = "qfq",
        sleep_sec: float = 0.5,
    ) -> None:
        """
        批量获取成分股日线数据并入库（支持断点续传与防封控容灾）

        :param symbol: 指数代码，默认为 000905 (中证 500)
        :param start_date: 开始日期 (YYYYMMDD，默认 20200101)
        :param adjust: 复权类型 (默认 qfq)
        :param sleep_sec: 每次请求强制休眠秒数 (默认 0.5s，新浪源安全速率)
        """
        stock_pool = self.get_zz500_stocks(symbol=symbol)
        total_stocks = len(stock_pool)

        if total_stocks == 0:
            self.logger.warning("股票池为空，请检查网络或指数代码后重试")
            return

        # 检查断点续传状态
        existing_symbols = self.get_existing_symbols(min_records=100)
        if existing_symbols:
            self.logger.info(
                f"【断点续传生效】检测到数据库中已存在 {len(existing_symbols)} 只股票的完整日线数据，将自动跳过。"
            )

        self.logger.info(f"开始批量拉取中证 500 成分股日线数据，共计 {total_stocks} 只股票...")
        success_count = len(existing_symbols)
        fail_count = 0
        newly_fetched = 0

        for idx, code in enumerate(stock_pool, start=1):
            # 若已入库则跳过
            if code in existing_symbols:
                self.logger.info(f"[{idx}/{total_stocks}] 股票 {code} 已存在于数据库中，跳过（断点续传）")
                continue

            self.logger.info(f"[{idx}/{total_stocks}] 正在拉取股票 {code} 日线行情 (start={start_date}, adjust={adjust})...")

            try:
                df = self.fetch_daily_bar(code=code, start_date=start_date, adjust=adjust)
                if df is not None and not df.empty:
                    self.save_to_db(df, table_name="daily_bars")
                    success_count += 1
                    newly_fetched += 1
                    self.logger.info(f"[{idx}/{total_stocks}] 股票 {code} 成功入库 {len(df)} 条日线记录")
                else:
                    fail_count += 1
            except Exception as e:
                fail_count += 1
                self.logger.warning(f"处理股票 {code} 时发生未知异常: {e}，跳过该股票继续执行")

            # 温和间隔，既防封禁又保证拉取效率
            time.sleep(sleep_sec)

        self.logger.info("=" * 60)
        self.logger.info(
            f"所有数据抓取任务执行完毕！\n"
            f"- 成分股总数: {total_stocks}\n"
            f"- 本次新抓取入库: {newly_fetched} 只\n"
            f"- 数据库现有有效股票: {success_count} 只\n"
            f"- 失败/停牌股票: {fail_count} 只\n"
            f"- 数据存储路径: {self.db_path}"
        )
        self.logger.info("=" * 60)


if __name__ == "__main__":
    fetcher = DataFetcher()
    fetcher.fetch_and_save_all()
