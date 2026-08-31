# -*- coding: utf-8 -*-
"""
A 股股票 + 核心指数「日截面数据仓库」自动化同步脚本

核心架构：
1. 元数据层 (meta/)：全量股票列表、指数基础信息、交易日历。
2. 股票日截面层 (stock/daily/)：单交易日全市场宽表 (行情 + 复权因子 + 每日基本面指标)。
3. 指数日截面层 (index/daily/)：单交易日核心指数宽表 (指数日行情 + 指数每日估值指标)。
4. 指数成分与权重 (index/weight/)：主流宽基指数成分股与权重变动表。

设计特性：
- 按日截面拉取：单日只需 3 次请求拉完全市场 5000+ 股票，规避 API 频控。
- 自动断点补漏：比对交易日历与本地已有文件，只请求缺失的交易日。
- 原子落盘机制：临时文件 + os.replace，保证任何异常中断均不产生脏数据。
- 采用 Parquet 列式存储：体积减少 75%，读取与多因子计算速度提升 10 倍以上。
"""

import logging
import os
import time
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

try:
    import tushare as ts
except ImportError:
    raise ImportError("请先安装依赖：pip install tushare pandas pyarrow python-dotenv")


# ==============================================================================
#                                  配置区
# ==============================================================================
# 起止时间 (格式: YYYYMMDD)。END_DATE 留空或为 None 时默认同步到今天
START_DATE = os.getenv("START_DATE", "20000101")
END_DATE = os.getenv("END_DATE", None)

# 数据仓库根目录
DATA_ROOT = os.getenv("DATA_ROOT", "stock_data_warehouse")

# 目录规划
DIR_META = os.path.join(DATA_ROOT, "meta")
DIR_STOCK_DAILY = os.path.join(DATA_ROOT, "stock", "daily")
DIR_INDEX_DAILY = os.path.join(DATA_ROOT, "index", "daily")
DIR_INDEX_WEIGHT = os.path.join(DATA_ROOT, "index", "weight")
DIR_FUND_DAILY = os.path.join(DATA_ROOT, "fund", "daily")
DIR_LOGS = os.path.join(DATA_ROOT, "logs")

# API 请求间隔（秒）与重试设置
SLEEP_SECONDS = 0.15
MAX_RETRY = 3

# 纳入监控与下载的核心宽基/大盘指数列表
CORE_INDICES = [
    "000001.SH",  # 上证指数
    "399001.SZ",  # 深证成指
    "399006.SZ",  # 创业板指
    "000688.SH",  # 科创50
    "000016.SH",  # 上证50
    "000300.SH",  # 沪深300
    "000905.SH",  # 中证500
    "000906.SH",  # 中证800
    "000852.SH",  # 中证1000
    "399303.SZ",  # 国证2000
]

# 指数权重关注列表
WEIGHT_INDICES = [
    "399006.SZ",  # 创业板指
    "000688.SH",  # 科创50
    "000016.SH",  # 上证50
    "000300.SH",  # 沪深300
    "000905.SH",  # 中证500
    "000906.SH",  # 中证800
    "000852.SH",  # 中证1000
]

# 常见核心指数的最早数据年份兜底（避免从 2000 年开始无谓地遍历几百次空请求）
INDEX_MIN_START = {
    "399006.SZ": "20100601",  # 创业板指 (2010年发布)
    "000688.SH": "20200701",  # 科创50 (2020年发布)
    "000016.SH": "20040101",  # 上证50 (2004年发布)
    "000300.SH": "20050101",  # 沪深300 (2005年发布)
    "000905.SH": "20050101",  # 中证500 (2005年发布)
    "000906.SH": "20050101",  # 中证800 (2005年发布)
    "000852.SH": "20141001",  # 中证1000 (2014年发布)
}


# ==============================================================================


def setup_logging():
    """配置日志输出"""
    os.makedirs(DIR_LOGS, exist_ok=True)
    log_file = os.path.join(DIR_LOGS, "sync_daily.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def init_tushare():
    """初始化 Tushare Pro 客户端"""
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise ValueError("请在 .env 文件或环境变量中配置 TUSHARE_TOKEN")
    ts.set_token(token)
    return ts.pro_api()


def _retry(func, desc: str):
    """通用重试包装器（带指数退避）"""
    for attempt in range(1, MAX_RETRY + 1):
        try:
            return func()
        except Exception as e:
            wait_time = attempt**2
            logging.warning("%s 失败(第 %d/%d 次): %s, %d 秒后重试...", desc, attempt, MAX_RETRY, e, wait_time)
            time.sleep(wait_time)
    logging.error("%s 达到最大重试次数，跳过！", desc)
    return None


def atomic_save_parquet(df: pd.DataFrame, file_path: str):
    """原子写入 Parquet 文件，杜绝写入中断带来的文件损坏"""
    if df is None or df.empty:
        return
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    tmp_path = file_path + ".tmp"
    df.to_parquet(tmp_path, index=False, engine="pyarrow")
    os.replace(tmp_path, file_path)


# ==============================================================================
#                               1. 元数据同步模块 (Meta)
# ==============================================================================


def sync_trade_cal(pro, start_year: str = "1990", end_year: Optional[str] = None) -> pd.DataFrame:
    """同步 A 股全历史交易日历"""
    end_year = end_year or str(datetime.now().year + 1)
    file_path = os.path.join(DIR_META, "trade_cal.parquet")

    logging.info("--> 正在同步交易日历 [%s ~ %s]...", start_year, end_year)
    df = _retry(
        lambda: pro.trade_cal(exchange="SSE", start_date=f"{start_year}0101", end_date=f"{end_year}1231"),
        desc="同步交易日历",
    )
    if df is not None and not df.empty:
        df = df.sort_values("cal_date").reset_index(drop=True)
        atomic_save_parquet(df, file_path)
        logging.info("交易日历同步完成，共 %d 条记录", len(df))
        return df
    elif os.path.exists(file_path):
        return pd.read_parquet(file_path)
    else:
        raise RuntimeError("无法获取交易日历且本地无备份！")


def sync_stock_basic(pro) -> pd.DataFrame:
    """同步 A 股全量股票列表（包含上市 L、退市 D、暂停 P）"""
    file_path = os.path.join(DIR_META, "stock_basic.parquet")
    logging.info("--> 正在同步 A 股股票基础信息表 (含退市股)...")

    frames = []
    for status in ["L", "D", "P"]:
        df = _retry(
            lambda: pro.stock_basic(
                exchange="",
                list_status=status,
                fields="ts_code,symbol,name,area,industry,fullname,enname,cnspell,market,exchange,curr_type,list_status,list_date,delist_date,is_hs,act_name,act_ent_type",
            ),
            desc=f"获取状态为 {status} 的股票列表",
        )
        if df is not None and not df.empty:
            frames.append(df)
        time.sleep(SLEEP_SECONDS)

    if frames:
        all_stocks = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["ts_code"]).reset_index(drop=True)
        atomic_save_parquet(all_stocks, file_path)
        logging.info("股票基础信息同步完成，全量股票共计 %d 只", len(all_stocks))
        return all_stocks
    elif os.path.exists(file_path):
        return pd.read_parquet(file_path)
    return pd.DataFrame()


def sync_index_basic(pro) -> pd.DataFrame:
    """同步核心市场指数基础信息"""
    file_path = os.path.join(DIR_META, "index_basic.parquet")
    logging.info("--> 正在同步指数基础信息表...")

    frames = []
    for market in ["SSE", "SZSE", "CSI", "CICC"]:
        df = _retry(
            lambda: pro.index_basic(market=market, fields="ts_code,name,fullname,market,publisher,index_type,category,base_date,base_point,list_date,weight_rule,desc,exp_date"),
            desc=f"获取 {market} 指数基础信息",
        )
        if df is not None and not df.empty:
            frames.append(df)
        time.sleep(SLEEP_SECONDS)

    if frames:
        all_indices = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["ts_code"]).reset_index(drop=True)
        atomic_save_parquet(all_indices, file_path)
        logging.info("指数基础信息同步完成，共计 %d 只指数", len(all_indices))
        return all_indices
    elif os.path.exists(file_path):
        return pd.read_parquet(file_path)
    return pd.DataFrame()


# ==============================================================================
#                        2. 股票日截面宽表同步模块 (Stock Daily)
# ==============================================================================


def sync_one_day_stock(pro, trade_date: str) -> bool:
    """
    拉取单日全市场 5000+ 股票数据并横向合并为宽表：
    行情(daily) + 复权因子(adj_factor) + 每日基本面(daily_basic)
    """
    out_file = os.path.join(DIR_STOCK_DAILY, f"{trade_date}.parquet")
    if os.path.exists(out_file):
        return True

    # 1. 获取不复权行情
    df_bar = _retry(
        lambda: pro.daily(trade_date=trade_date, fields="ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"),
        desc=f"股票日线行情 {trade_date}",
    )
    time.sleep(SLEEP_SECONDS)

    # 2. 获取复权因子
    df_adj = _retry(
        lambda: pro.adj_factor(trade_date=trade_date, fields="ts_code,trade_date,adj_factor"),
        desc=f"股票复权因子 {trade_date}",
    )
    time.sleep(SLEEP_SECONDS)

    # 3. 获取每日基本面估值指标
    df_basic = _retry(
        lambda: pro.daily_basic(
            trade_date=trade_date,
            fields="ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,free_share,total_mv,circ_mv,limit_status",
        ),
        desc=f"股票每日指标 {trade_date}",
    )
    time.sleep(SLEEP_SECONDS)

    # 如果行情本身为空（例如极特殊休市或非交易数据），则不落盘
    if df_bar is None or df_bar.empty:
        logging.warning("交易日 %s 股票行情数据为空，跳过", trade_date)
        return False

    # 横向对齐合并 (以 ts_code 为主键)
    merged = df_bar.copy()

    if df_adj is not None and not df_adj.empty:
        merged = pd.merge(merged, df_adj[["ts_code", "adj_factor"]], on="ts_code", how="left")
    else:
        merged["adj_factor"] = 1.0

    if df_basic is not None and not df_basic.empty:
        # 剔除 daily_basic 中重复的 close 列，避免合并冲突
        basic_cols = [c for c in df_basic.columns if c not in ["close", "trade_date"]]
        merged = pd.merge(merged, df_basic[basic_cols], on="ts_code", how="left")

    # 类型规整与排序
    merged["trade_date"] = trade_date
    merged = merged.sort_values("ts_code").reset_index(drop=True)

    # 原子写入 Parquet
    atomic_save_parquet(merged, out_file)
    return True


# ==============================================================================
#                        3. 指数日截面宽表同步模块 (Index Daily)
# ==============================================================================


def sync_one_day_index(pro, trade_date: str) -> bool:
    """
    拉取单日核心指数数据并横向合并为宽表：
    指数日线行情(index_daily) + 指数每日指标(index_dailybasic)
    """
    out_file = os.path.join(DIR_INDEX_DAILY, f"{trade_date}.parquet")
    if os.path.exists(out_file):
        return True

    # 1. 获取指数日行情
    df_daily = _retry(
        lambda: pro.index_daily(trade_date=trade_date, fields="ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"),
        desc=f"指数日行情 {trade_date}",
    )
    time.sleep(SLEEP_SECONDS)

    # 2. 获取指数估值与流动性指标
    df_basic = _retry(
        lambda: pro.index_dailybasic(trade_date=trade_date, fields="ts_code,trade_date,total_mv,float_mv,total_share,float_share,free_share,turnover_rate,turnover_rate_f,pe,pe_ttm,pb"),
        desc=f"指数每日指标 {trade_date}",
    )
    time.sleep(SLEEP_SECONDS)

    if df_daily is None or df_daily.empty:
        return False

    # 过滤出核心关注指数
    df_daily = df_daily[df_daily["ts_code"].isin(CORE_INDICES)].copy()
    if df_daily.empty:
        return False

    merged = df_daily
    if df_basic is not None and not df_basic.empty:
        basic_cols = [c for c in df_basic.columns if c not in ["trade_date"]]
        merged = pd.merge(merged, df_basic[basic_cols], on="ts_code", how="left")

    merged["trade_date"] = trade_date
    merged = merged.sort_values("ts_code").reset_index(drop=True)

    atomic_save_parquet(merged, out_file)
    return True


# ==============================================================================
#                        4. 指数成分与权重同步模块 (Index Weight)
# ==============================================================================


def sync_index_weights(pro, start_date: str, end_date: str):
    """
    增量同步核心宽基指数的历史成分股权重
    """
    logging.info("--> 正在同步核心指数成分与历史权重...")
    os.makedirs(DIR_INDEX_WEIGHT, exist_ok=True)

    for index_code in WEIGHT_INDICES:
        file_path = os.path.join(DIR_INDEX_WEIGHT, f"{index_code}.parquet")
        local_max = None

        if os.path.exists(file_path):
            try:
                old_df = pd.read_parquet(file_path, columns=["trade_date"])
                if not old_df.empty:
                    local_max = str(old_df["trade_date"].max())
            except Exception as e:
                logging.warning("读取本地指数权重失败 %s: %s", index_code, e)

        # 结合指数实际发布日期确定需要补齐的时间段，防止从 2000 年发起几百次无效空请求触发频控
        min_valid_start = INDEX_MIN_START.get(index_code, start_date)
        req_start = max(start_date, min_valid_start)

        if local_max and local_max >= req_start:
            req_start = local_max

        if req_start > end_date:
            continue

        logging.info("同步指数权重: %s [%s ~ %s]", index_code, req_start, end_date)

        # ==================== 按月生成切片 ====================
        # 使用 pandas 生成从 req_start 到 end_date 的所有自然月区间
        month_periods = pd.period_range(start=pd.to_datetime(req_start), end=pd.to_datetime(end_date), freq="M")

        frames = []
        for p in month_periods:
            m_start = p.start_time.strftime("%Y%m%d")
            m_end = p.end_time.strftime("%Y%m%d")

            # 对首尾两个月做精确日期边界截断
            if m_start < req_start:
                m_start = req_start
            if m_end > end_date:
                m_end = end_date

            df_month = _retry(
                lambda s=m_start, e=m_end: pro.index_weight(
                    index_code=index_code,
                    start_date=s,
                    end_date=e,
                    fields="index_code,con_code,trade_date,weight",
                ),
                desc=f"获取指数权重 {index_code} ({p})",
            )

            if df_month is not None and not df_month.empty:
                frames.append(df_month)

            time.sleep(SLEEP_SECONDS)
        # =============================================================

        if frames:
            df_new = pd.concat(frames, ignore_index=True)
            if os.path.exists(file_path):
                old_df = pd.read_parquet(file_path)
                df_all = pd.concat([old_df, df_new], ignore_index=True)
            else:
                df_all = df_new

            # 确保 trade_date 为统一字符串类型后去重排序
            df_all["trade_date"] = df_all["trade_date"].astype(str)
            df_all = df_all.drop_duplicates(subset=["index_code", "con_code", "trade_date"]).sort_values(["trade_date", "con_code"]).reset_index(drop=True)
            atomic_save_parquet(df_all, file_path)


# ==============================================================================
#                        5. 场内基金/ETF 日截面同步模块 (Fund Daily)
# ==============================================================================


def sync_fund_basic(pro) -> pd.DataFrame:
    """同步全量场内基金（ETF/LOF）基础信息表（含上市 L 与 退市 D）"""
    file_path = os.path.join(DIR_META, "fund_basic.parquet")
    logging.info("--> 正在同步场内基金 (ETF/LOF) 基础信息表...")

    frames = []
    for status in ["L", "D"]:
        df = _retry(
            lambda: pro.fund_basic(
                market="E",
                status=status,
                fields="ts_code,name,management,custodian,fund_type,found_date,due_date,list_date,issue_date,delist_date,issue_amount,m_fee,c_fee,duration_year,p_value,min_amount,exp_return,benchmark,status,invest_type,type,trustee,purc_startdate,redm_startdate,market",
            ),
            desc=f"获取状态为 {status} 的场内基金列表",
        )
        if df is not None and not df.empty:
            frames.append(df)
        time.sleep(SLEEP_SECONDS)

    if frames:
        all_funds = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["ts_code"]).reset_index(drop=True)
        atomic_save_parquet(all_funds, file_path)
        logging.info("场内基金基础信息同步完成，共计 %d 只", len(all_funds))
        return all_funds
    elif os.path.exists(file_path):
        return pd.read_parquet(file_path)
    return pd.DataFrame()


def sync_one_day_fund(pro, trade_date: str) -> bool:
    """
    拉取单日全量场内 ETF/LOF 数据并横向合并为全量宽表：
    1. 不复权日行情 (fund_daily)
    2. 基金复权因子 (fund_adj)
    3. 每日基金净值与资产 (fund_nav)
    4. 场内基金份额 (fund_share)
    5. 衍生量化指标：折溢价率 (discount_rate)
    """
    out_file = os.path.join(DIR_FUND_DAILY, f"{trade_date}.parquet")
    if os.path.exists(out_file):
        return True

    # 1. 获取场内不复权行情 (OHLCV)
    df_bar = _retry(
        lambda: pro.fund_daily(
            trade_date=trade_date,
            fields="ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount",
        ),
        desc=f"ETF 日线行情 {trade_date}",
    )
    time.sleep(SLEEP_SECONDS)

    if df_bar is None or df_bar.empty:
        logging.warning("交易日 %s 场内基金行情为空，跳过", trade_date)
        return False

    # 2. 获取基金专属复权因子
    df_adj = _retry(
        lambda: pro.fund_adj(
            trade_date=trade_date,
            fields="ts_code,trade_date,adj_factor",
        ),
        desc=f"ETF 复权因子 {trade_date}",
    )
    time.sleep(SLEEP_SECONDS)

    # 3. 获取当日净值及资产规模 (按 nav_date 匹配)
    df_nav = _retry(
        lambda: pro.fund_nav(
            nav_date=trade_date,
            fields="ts_code,ann_date,nav_date,unit_nav,accum_nav,accum_div,net_asset,total_netasset,adj_nav",
        ),
        desc=f"ETF 净值数据 {trade_date}",
    )
    time.sleep(SLEEP_SECONDS)

    # 4. 获取场内基金流通份额规模 (fund_share)
    df_share = _retry(
        lambda: pro.fund_share(
            trade_date=trade_date,
            fields="ts_code,trade_date,fd_share",
        ),
        desc=f"ETF 份额数据 {trade_date}",
    )
    time.sleep(SLEEP_SECONDS)

    # 横向对齐合并宽表
    merged = df_bar.copy()

    # 合并复权因子
    if df_adj is not None and not df_adj.empty:
        merged = pd.merge(merged, df_adj[["ts_code", "adj_factor"]], on="ts_code", how="left")
    else:
        merged["adj_factor"] = 1.0

    # 合并净值与资产
    if df_nav is not None and not df_nav.empty:
        # 去重防止极端情况下同一基金多条记录
        df_nav = df_nav.drop_duplicates(subset=["ts_code"])
        merged = pd.merge(merged, df_nav, on="ts_code", how="left")

    # 合并份额
    if df_share is not None and not df_share.empty:
        df_share = df_share.drop_duplicates(subset=["ts_code"])
        merged = pd.merge(merged, df_share[["ts_code", "fd_share"]], on="ts_code", how="left")

    # 5. 衍生核心量化特征：计算折溢价率 (Discount Rate %)
    # 公式：(收盘价 - 单位净值) / 单位净值 * 100
    if "unit_nav" in merged.columns:
        merged["discount_rate"] = ((merged["close"] - merged["unit_nav"]) / merged["unit_nav"] * 100).round(4)

    # 规整与排序
    merged["trade_date"] = trade_date
    merged = merged.sort_values("ts_code").reset_index(drop=True)

    # 原子落盘
    atomic_save_parquet(merged, out_file)
    return True


# ==============================================================================
#                               6. 主调度流程
# ==============================================================================


def main():
    setup_logging()
    logging.info("=" * 60)
    logging.info("启动 A 股股票与核心指数「日截面数据仓库」自动化同步流水线")

    # 1. 确定同步时间范围
    req_start = START_DATE
    req_end = END_DATE or datetime.now().strftime("%Y%m%d")
    logging.info("目标时间范围: %s ~ %s | 存储根目录: %s", req_start, req_end, os.path.abspath(DATA_ROOT))

    # 2. 初始化 Tushare
    pro = init_tushare()

    # 3. 同步元数据 (交易日历, 股票列表, 指数列表)
    trade_cal_df = sync_trade_cal(pro, start_year=req_start[:4])
    sync_stock_basic(pro)
    sync_index_basic(pro)
    sync_fund_basic(pro)

    # 4. 获取区间内所有「真实开市交易日」
    open_days = trade_cal_df[(trade_cal_df["is_open"] == 1) & (trade_cal_df["cal_date"] >= req_start) & (trade_cal_df["cal_date"] <= req_end)]["cal_date"].tolist()

    total_days = len(open_days)
    logging.info("区间内共计 %d 个实际交易日需要检查/同步", total_days)

    # 5. 按交易日截面流水线同步 (股票 + 指数 + ETF)
    success_stock, success_index, success_fund = 0, 0, 0
    os.makedirs(DIR_STOCK_DAILY, exist_ok=True)
    os.makedirs(DIR_INDEX_DAILY, exist_ok=True)
    os.makedirs(DIR_FUND_DAILY, exist_ok=True)

    for idx, trade_date in enumerate(open_days, 1):
        stock_file_exists = os.path.exists(os.path.join(DIR_STOCK_DAILY, f"{trade_date}.parquet"))
        index_file_exists = os.path.exists(os.path.join(DIR_INDEX_DAILY, f"{trade_date}.parquet"))
        fund_file_exists = os.path.exists(os.path.join(DIR_FUND_DAILY, f"{trade_date}.parquet"))

        # 如果三者都已存在，直接跳过
        if stock_file_exists and index_file_exists and fund_file_exists:
            continue

        logging.info("[%d/%d] 正在处理交易日: %s", idx, total_days, trade_date)

        # 同步股票日截面
        if not stock_file_exists:
            if sync_one_day_stock(pro, trade_date):
                success_stock += 1

        # 同步指数日截面
        if not index_file_exists:
            if sync_one_day_index(pro, trade_date):
                success_index += 1

        # 同步 ETF 日截面 (新增)
        if not fund_file_exists:
            if sync_one_day_fund(pro, trade_date):
                success_fund += 1

    # 6. 同步指数成分与权重
    sync_index_weights(pro, req_start, req_end)

    logging.info("=" * 60)
    logging.info("✅ 数据仓库同步完成！")
    logging.info("新增/补齐股票: %d 天 | 指数: %d 天 | ETF: %d 天", success_stock, success_index, success_fund)
    logging.info("数据仓库路径: %s", os.path.abspath(DATA_ROOT))


if __name__ == "__main__":
    main()
