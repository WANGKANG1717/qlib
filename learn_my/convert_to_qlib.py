# -*- coding: utf-8 -*-
"""
数据仓库 -> Qlib 原生二进制格式 (cn_data) 极速转换脚本

功能：
1. 提取 Stock / Index / Fund 日截面 Parquet 数据，完成 Qlib 规范的字段名、日期及量纲标准化。
2. 使用 DuckDB 极速重组为单标的 CSV 临时文件。
3. 调用 Qlib dump_bin 生成 features/*.bin 二进制矩阵与 calendars/day.txt 交易日历。
4. 解析 index/weight/ 成分股权重，自动生成 Qlib 标准的成分股池 (csi300.txt, csi500.txt, csi1000.txt 等)。
"""

import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

try:
    import duckdb
except ImportError:
    raise ImportError("请先安装 duckdb 极速转换引擎：pip install duckdb")


# ==============================================================================
#                                  配置区
# ==============================================================================
# 数据仓库源根目录
DATA_ROOT = os.path.expanduser(os.getenv("DATA_ROOT", "stock_data_warehouse"))

# 目标 Qlib 数据目录 (默认存储在 ~/.qlib/qlib_data/cn_data)
QLIB_DIR = os.path.expanduser(os.getenv("QLIB_DIR", "~/.qlib/qlib_data/my_tushare_data_new"))

# 临时重组 CSV 目录 (转换后可自动清理)
TEMP_CSV_DIR = os.path.join(DATA_ROOT, "temp_csv_for_qlib")

# 包含的基础行情与基本面字段 (导出到 Qlib)
# 基础行情必须包含: open, close, high, low, volume, money, factor
STOCK_FIELDS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "money",
    "factor",
    "vwap",
    "turnover_rate",
    "turnover_rate_f",
    "pe",
    "pe_ttm",
    "pb",
    "ps",
    "ps_ttm",
    "dv_ratio",
    "dv_ttm",
    "total_mv",
    "circ_mv",
]

# 指数权重到 Qlib 股票池的映射配置 (代码 -> 文件名)
INDEX_INSTRUMENT_MAP = {
    "399006.SZ": "chinext.txt",
    "000688.SH": "star50.txt",
    "000016.SH": "csi50.txt",
    "000300.SH": "csi300.txt",
    "000905.SH": "csi500.txt",
    "000906.SH": "csi800.txt",
    "000852.SH": "csi1000.txt",
}

# 并行转换线程数
MAX_WORKERS = max(1, os.cpu_count() - 2)
# ==============================================================================


def setup_logging():
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )


def to_qlib_symbol(ts_code: str) -> str:
    """标准 A 股代码转 Qlib 命名: 600000.SH -> SH600000, 000001.SZ -> SZ000001"""
    if not ts_code or "." not in ts_code:
        return ts_code
    code, market = ts_code.split(".")
    return f"{market.upper()}{code}"


# ==============================================================================
#                       1. 数据重组模块 (DuckDB 极速引擎)
# ==============================================================================


def export_parquet_to_symbol_csv():
    """使用 DuckDB 扫描全量日截面 Parquet，按 symbol 分区极速导出为单文件 CSV"""
    logging.info("--> [步骤 1/3] 使用 DuckDB 极速重组日截面数据为单标的 CSV...")

    if os.path.exists(TEMP_CSV_DIR):
        shutil.rmtree(TEMP_CSV_DIR)
    os.makedirs(TEMP_CSV_DIR, exist_ok=True)

    con = duckdb.connect()

    # 1. 导出股票数据 (处理量纲: vol*100 为股, amount*1000 为元, 计算 vwap)
    stock_dir = os.path.join(DATA_ROOT, "stock", "daily").replace("\\", "/")
    if os.path.exists(os.path.join(DATA_ROOT, "stock", "daily")):
        logging.info("  正在处理 A 股股票日截面数据...")
        query_stock = f"""
            COPY (
                SELECT 
                    strftime(strptime(trade_date::VARCHAR, '%Y%m%d'), '%Y-%m-%d') AS date,
                    regexp_replace(ts_code, '^([0-9]+)\.([A-Za-z]+)$', '\\2\\1') AS symbol,
                    
                    -- 核心：转换为后复权价格 (Raw * Factor)
                    ROUND(open * COALESCE(adj_factor, 1.0), 4) AS open,
                    ROUND(high * COALESCE(adj_factor, 1.0), 4) AS high,
                    ROUND(low * COALESCE(adj_factor, 1.0), 4) AS low,
                    ROUND(close * COALESCE(adj_factor, 1.0), 4) AS close,
                    
                    -- 量纲换算: 手 -> 股, 千元 -> 元
                    COALESCE(vol * 100, 0.0) AS volume,
                    COALESCE(amount * 1000, 0.0) AS money,
                    COALESCE(adj_factor, 1.0) AS factor,
                    
                    -- 后复权 VWAP (成交均价 * factor)
                    CASE 
                        WHEN vol > 0 THEN ROUND(((amount * 1000) / (vol * 100)) * COALESCE(adj_factor, 1.0), 4)
                        ELSE ROUND(close * COALESCE(adj_factor, 1.0), 4)
                    END AS vwap,
                    
                    turnover_rate, turnover_rate_f, pe, pe_ttm, pb, ps, ps_ttm,
                    dv_ratio, dv_ttm, total_mv, circ_mv
                FROM '{stock_dir}/*.parquet'
                WHERE ts_code IS NOT NULL
                ORDER BY symbol, date ASC
            ) TO '{TEMP_CSV_DIR}' (
                FORMAT CSV, HEADER TRUE, PARTITION_BY (symbol), OVERWRITE_OR_IGNORE TRUE
            );
        """
        con.execute(query_stock)

    # 2. 导出核心指数数据 (指数点位天然连续，factor 统一设为 1.0)
    index_dir = os.path.join(DATA_ROOT, "index", "daily").replace("\\", "/")
    if os.path.exists(os.path.join(DATA_ROOT, "index", "daily")):
        logging.info("  正在处理核心指数数据...")
        query_index = f"""
            COPY (
                SELECT 
                    strftime(strptime(trade_date::VARCHAR, '%Y%m%d'), '%Y-%m-%d') AS date,
                    regexp_replace(ts_code, '^([0-9]+)\.([A-Za-z]+)$', '\\2\\1') AS symbol,
                    open, high, low, close,
                    COALESCE(vol * 100, 0.0) AS volume,
                    COALESCE(amount * 1000, 0.0) AS money,
                    1.0 AS factor,
                    CASE WHEN vol > 0 THEN (amount * 1000) / (vol * 100) ELSE close END AS vwap,
                    turnover_rate, turnover_rate_f, pe, pe_ttm, pb, NULL AS ps, NULL AS ps_ttm,
                    dv_ratio, dv_ttm, total_mv, float_mv AS circ_mv
                FROM '{index_dir}/*.parquet'
                WHERE ts_code IS NOT NULL
                ORDER BY symbol, date ASC
            ) TO '{TEMP_CSV_DIR}' (
                FORMAT CSV, HEADER TRUE, PARTITION_BY (symbol), OVERWRITE_OR_IGNORE TRUE
            );
        """
        con.execute(query_index)

    # 3. 导出 ETF 场内基金数据
    fund_dir = os.path.join(DATA_ROOT, "fund", "daily").replace("\\", "/")
    if os.path.exists(os.path.join(DATA_ROOT, "fund", "daily")):
        logging.info("  正在处理 ETF 场内基金数据...")
        query_fund = f"""
            COPY (
                SELECT 
                    strftime(strptime(trade_date::VARCHAR, '%Y%m%d'), '%Y-%m-%d') AS date,
                    regexp_replace(ts_code, '^([0-9]+)\.([A-Za-z]+)$', '\\2\\1') AS symbol,
                    
                    -- ETF 后复权价格 (Raw * Factor)
                    ROUND(open * COALESCE(adj_factor, 1.0), 4) AS open,
                    ROUND(high * COALESCE(adj_factor, 1.0), 4) AS high,
                    ROUND(low * COALESCE(adj_factor, 1.0), 4) AS low,
                    ROUND(close * COALESCE(adj_factor, 1.0), 4) AS close,
                    
                    COALESCE(vol * 100, 0.0) AS volume,
                    COALESCE(amount * 1000, 0.0) AS money,
                    COALESCE(adj_factor, 1.0) AS factor,
                    
                    CASE 
                        WHEN vol > 0 THEN ROUND(((amount * 1000) / (vol * 100)) * COALESCE(adj_factor, 1.0), 4)
                        ELSE ROUND(close * COALESCE(adj_factor, 1.0), 4)
                    END AS vwap,
                    
                    NULL AS turnover_rate, NULL AS turnover_rate_f, NULL AS pe, NULL AS pe_ttm, NULL AS pb, NULL AS ps, NULL AS ps_ttm,
                    NULL AS dv_ratio, NULL AS dv_ttm, total_netasset AS total_mv, net_asset AS circ_mv
                FROM '{fund_dir}/*.parquet'
                WHERE ts_code IS NOT NULL
                ORDER BY symbol, date ASC
            ) TO '{TEMP_CSV_DIR}' (
                FORMAT CSV, HEADER TRUE, PARTITION_BY (symbol), OVERWRITE_OR_IGNORE TRUE
            );
        """
        con.execute(query_fund)

    # 4. 扁平化 DuckDB 分区文件夹结构为 {symbol}.csv
    logging.info("  正在整理临时 CSV 文件名...")
    for root, dirs, files in os.walk(TEMP_CSV_DIR):
        for file in files:
            if file.endswith(".csv"):
                parent_dir = os.path.basename(root)
                if "symbol=" in parent_dir:
                    symbol = parent_dir.replace("symbol=", "")
                    target_file = os.path.join(TEMP_CSV_DIR, f"{symbol}.csv")
                    shutil.move(os.path.join(root, file), target_file)

    for d in os.listdir(TEMP_CSV_DIR):
        dir_path = os.path.join(TEMP_CSV_DIR, d)
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)

    total_symbols = len([f for f in os.listdir(TEMP_CSV_DIR) if f.endswith(".csv")])
    logging.info("✅ 临时 CSV 生成完毕，共计 %d 个标的文件", total_symbols)


# ==============================================================================
#                       2. 调用 Qlib 转二进制矩阵模块
# ==============================================================================


def run_qlib_dump_bin():
    """调用 Qlib 官方 dump_bin 模块生成高性能 .bin 二进制文件"""
    logging.info("--> [步骤 2/3] 调用 Qlib dump_bin 编译生成二进制矩阵 (features/*.bin)...")

    include_fields_str = ",".join(STOCK_FIELDS)
    cmd = [
        sys.executable,
        "-m",
        "qlib.dump_bin",
        "dump_all",
        "--csv_path",
        TEMP_CSV_DIR,
        "--qlib_dir",
        QLIB_DIR,
        "--symbol_field_name",
        "symbol",
        "--date_field_name",
        "date",
        "--include_fields",
        include_fields_str,
        "--max_workers",
        str(MAX_WORKERS),
    ]

    logging.info("执行命令: %s", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError("Qlib dump_bin 执行失败，请检查上方报错！")

    logging.info("✅ Qlib 二进制底层库编译成功！")


# ==============================================================================
#                 3. 构建成分股股票池 (instruments/*.txt)
# ==============================================================================


def build_index_instruments():
    """
    根据 index/weight/ 目录下的历史权重，生成 Qlib 标准的成分股区间池
    格式: <symbol>\t<start_date>\t<end_date>
    """
    logging.info("--> [步骤 3/3] 正在根据历史权重生成 Qlib 成分股池 (instruments/*.txt)...")

    weight_dir = os.path.join(DATA_ROOT, "index", "weight")
    instruments_dir = os.path.join(QLIB_DIR, "instruments")
    os.makedirs(instruments_dir, exist_ok=True)

    if not os.path.exists(weight_dir):
        logging.warning("未找到 %s 目录，跳过成分股池生成", weight_dir)
        return

    # 读取 Qlib 已生成的全市场交易日历
    cal_file = os.path.join(QLIB_DIR, "calendars", "day.txt")
    if not os.path.exists(cal_file):
        logging.warning("未找到交易日历 %s，跳过成分股池生成", cal_file)
        return

    with open(cal_file, "r", encoding="utf-8") as f:
        all_trading_days = sorted([line.strip() for line in f if line.strip()])

    for index_code, out_filename in INDEX_INSTRUMENT_MAP.items():
        weight_file = os.path.join(weight_dir, f"{index_code}.parquet")
        if not os.path.exists(weight_file):
            continue

        logging.info("  正在构建 %s 的成份股池 -> %s...", index_code, out_filename)
        df_weight = pd.read_parquet(weight_file)
        if df_weight.empty:
            continue

        # 格式化日期与代码
        df_weight["trade_date"] = pd.to_datetime(df_weight["trade_date"].astype(str)).dt.strftime("%Y-%m-%d")
        df_weight["symbol"] = df_weight["con_code"].apply(to_qlib_symbol)

        # 获取所有调仓点
        weight_dates = sorted(df_weight["trade_date"].unique())

        # 计算每个成份股在指数内的生效起止时间区间
        records = []
        for symbol, gp in df_weight.groupby("symbol"):
            in_dates = set(gp["trade_date"].unique())
            # 找到连续持有的区间
            start_d = None
            last_d = None

            for d in weight_dates:
                if d in in_dates:
                    if start_d is None:
                        start_d = d
                    last_d = d
                else:
                    if start_d is not None:
                        # 确定该段的结束时间 (下一期调仓日的前一天)
                        records.append(f"{symbol}\t{start_d}\t{last_d}\n")
                        start_d = None
                        last_d = None
            if start_d is not None:
                # 至今仍然在成分股内，结束日期设为最后一个交易日
                records.append(f"{symbol}\t{start_d}\t{all_trading_days[-1]}\n")

        # 写入 instruments/{name}.txt
        target_path = os.path.join(instruments_dir, out_filename)
        with open(target_path, "w", encoding="utf-8") as f:
            f.writelines(sorted(records))

    logging.info("✅ 股票池 instruments 生成完毕！")


# ==============================================================================
#                                  主入口
# ==============================================================================


def main(clean_temp_csv: bool = True):
    setup_logging()
    logging.info("=" * 60)
    logging.info("启动 数据仓库 -> Qlib 二进制格式自动化构建流水线")
    logging.info("源仓库目录: %s", DATA_ROOT)
    logging.info("目标 Qlib 目录: %s", QLIB_DIR)

    # 1. 导出 CSV
    export_parquet_to_symbol_csv()

    # 2. 编译为 Qlib 二进制
    run_qlib_dump_bin()

    # 3. 构建动态指数股票池
    build_index_instruments()

    # 4. 清理临时 CSV 文件
    if clean_temp_csv and os.path.exists(TEMP_CSV_DIR):
        logging.info("--> 正在清理临时 CSV 文件夹...")
        shutil.rmtree(TEMP_CSV_DIR)

    logging.info("=" * 60)
    logging.info("🎉 全流程构建完成！可以直接在 Qlib 中无缝初始化并读取数据。")
    logging.info("Qlib 初始化路径: qlib.init(provider_uri='%s')", QLIB_DIR)


if __name__ == "__main__":
    # clean_temp_csv=True 会在生成二进制后自动删除中间 CSV，节省磁盘空间
    main(clean_temp_csv=True)
