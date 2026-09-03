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
DATA_ROOT = os.path.expanduser(os.getenv("DATA_ROOT", "~/.qlib/qlib_data/stock_data_warehouse"))

# 目标 Qlib 数据目录 (默认存储在 ~/.qlib/qlib_data/my_tushare_data)
QLIB_DIR = os.path.expanduser(os.getenv("QLIB_DIR", "~/.qlib/qlib_data/my_tushare_data"))

# 临时重组 CSV 目录 (转换后可自动清理)
TEMP_CSV_DIR = os.path.join(DATA_ROOT, "temp_csv_for_qlib")

# 包含的基础行情与基本面字段 (导出到 Qlib)
# 基础行情必须包含: open, close, high, low, volume, amount, factor
STOCK_FIELDS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
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
    # 【新增字段】
    "industry",  # 行业整型编码
    "is_ST",  # ST 标记 (0/1)
    "list_status",  # 上市状态 (0=L, 1=D, 2=P)
]

# 指数权重到 Qlib 股票池的映射配置 (代码 -> 文件名)
INDEX_INSTRUMENT_MAP = {
    "000001.SH": "shcomp.txt",  # 上证指数
    "399001.SZ": "szcomp.txt",  # 深证成指
    "399006.SZ": "chinext.txt",  # 创业板指
    "000688.SH": "star50.txt",  # 科创50
    "000016.SH": "csi50.txt",  # 上证50
    "000300.SH": "csi300.txt",  # 沪深300
    "000905.SH": "csi500.txt",  # 中证500
    "000906.SH": "csi800.txt",  # 中证800
    "000852.SH": "csi1000.txt",  # 中证1000
    "399303.SZ": "cni2000.txt",  # 国证2000
}

SECTOR_INSTRUMENT_MAP = {
    "主板": "zb.txt",  # 主板 (沪市主板 + 深市主板)
    "创业板": "cyb.txt",  # 创业板
    "科创板": "kcb.txt",  # 科创板
    "北交所": "bjs.txt",  # 北交所
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
    logging.info("--> [步骤 1/2] 使用 DuckDB 极速重组日截面数据为单标的 CSV...")

    if os.path.exists(TEMP_CSV_DIR):
        shutil.rmtree(TEMP_CSV_DIR)
    os.makedirs(TEMP_CSV_DIR, exist_ok=True)

    con = duckdb.connect()

    # =========================================================================
    # 【新增逻辑】：读取 stock_basic.parquet，构建静态属性映射视图
    # =========================================================================
    basic_file = os.path.join(DATA_ROOT, "meta", "stock_basic.parquet")
    if os.path.exists(basic_file):
        df_meta = pd.read_parquet(basic_file, columns=["ts_code", "name", "industry", "list_status"])

        # 1. 编码 is_ST: 名称里包含 'ST' 则为 1，否则为 0
        df_meta["is_ST"] = df_meta["name"].fillna("").str.upper().str.contains("ST").astype(int)

        # 2. 编码 list_status: L=0(上市), D=1(退市), P=2(暂停)
        status_map = {"L": 0, "D": 1, "P": 2}
        df_meta["list_status_code"] = df_meta["list_status"].map(status_map).fillna(0).astype(int)

        # 3. 编码 industry: 文本转递增整数 (如: 银行=1, 电子=2)
        df_meta["industry"] = df_meta["industry"].fillna("未知")
        unique_industries =  sorted(df_meta["industry"].unique())
        ind_map = {name: i + 1 for i, name in enumerate(unique_industries)}
        df_meta["industry_code"] = df_meta["industry"].map(ind_map).astype(int)

        # 将处理好的元数据注册进 DuckDB 内存视图中
        meta_view = df_meta[["ts_code", "is_ST", "list_status_code", "industry_code"]]
        con.register("meta_view", meta_view)

        # 把行业对照表存下来，方便查数字对应哪个行业
        import json

        with open(os.path.join(DATA_ROOT, "meta", "industry_mapping.json"), "w", encoding="utf-8") as f:
            json.dump(ind_map, f, ensure_ascii=False, indent=4)
    else:
        # 兜底：如果没有 basic 文件，建一个空表防报错
        meta_view = pd.DataFrame(columns=["ts_code", "is_ST", "list_status_code", "industry_code"])
        con.register("meta_view", meta_view)
    # =========================================================================

    # 1. 导出股票数据 (LEFT JOIN 元数据)
    stock_dir = os.path.join(DATA_ROOT, "stock", "daily").replace("\\", "/")
    if os.path.exists(os.path.join(DATA_ROOT, "stock", "daily")):
        logging.info("  正在处理 A 股股票日截面数据...")
        query_stock = f"""
            COPY (
                SELECT 
                    strftime(strptime(t.trade_date::VARCHAR, '%Y%m%d'), '%Y-%m-%d') AS date,
                    regexp_replace(t.ts_code, '^([0-9]+)\.([A-Za-z]+)$', '\\2\\1') AS symbol,
                    
                    -- 核心：转换为后复权价格 (Raw * Factor)
                    ROUND(t.open * COALESCE(t.adj_factor, 1.0), 4) AS open,
                    ROUND(t.high * COALESCE(t.adj_factor, 1.0), 4) AS high,
                    ROUND(t.low * COALESCE(t.adj_factor, 1.0), 4) AS low,
                    ROUND(t.close * COALESCE(t.adj_factor, 1.0), 4) AS close,
                   
                    -- 量纲换算: 手 -> 股, 千元 -> 元
                    COALESCE(t.vol * 100, 0.0) AS volume,
                    COALESCE(t.amount * 1000, 0.0) AS amount,
                    COALESCE(t.adj_factor, 1.0) AS factor,
                    
                    -- 后复权 VWAP (成交均价 * factor)
                    CASE 
                        WHEN t.vol > 0 THEN ROUND(((t.amount * 1000) / (t.vol * 100)) * COALESCE(t.adj_factor, 1.0), 4)
                        ELSE ROUND(t.close * COALESCE(t.adj_factor, 1.0), 4)
                    END AS vwap,
                    
                    t.turnover_rate, t.turnover_rate_f, t.pe, t.pe_ttm, t.pb, t.ps, t.ps_ttm,
                    t.dv_ratio, t.dv_ttm, t.total_mv, t.circ_mv,
                    
                    -- 【新增拼接字段】
                    COALESCE(m.industry_code, 0) AS industry,
                    COALESCE(m.is_ST, 0) AS is_ST,
                    COALESCE(m.list_status_code, 0) AS list_status
                    
                FROM '{stock_dir}/*.parquet' AS t
                LEFT JOIN meta_view AS m ON t.ts_code = m.ts_code
                WHERE t.ts_code IS NOT NULL
                ORDER BY symbol, date ASC
            ) TO '{TEMP_CSV_DIR}' (
                FORMAT CSV, HEADER TRUE, PARTITION_BY (symbol), OVERWRITE_OR_IGNORE TRUE
            );
        """
        con.execute(query_stock)

    # # 2. 导出核心指数数据 (指数点位天然连续，factor 统一设为 1.0)
    # index_dir = os.path.join(DATA_ROOT, "index", "daily").replace("\\", "/")
    # if os.path.exists(os.path.join(DATA_ROOT, "index", "daily")):
    #     logging.info("  正在处理核心指数数据...")
    #     query_index = f"""
    #         COPY (
    #             SELECT
    #                 strftime(strptime(trade_date::VARCHAR, '%Y%m%d'), '%Y-%m-%d') AS date,
    #                 regexp_replace(ts_code, '^([0-9]+)\.([A-Za-z]+)$', '\\2\\1') AS symbol,
    #                 open, high, low, close,
    #                 COALESCE(vol * 100, 0.0) AS volume,
    #                 COALESCE(amount * 1000, 0.0) AS amount,
    #                 1.0 AS factor,
    #                 CASE WHEN vol > 0 THEN (amount * 1000) / (vol * 100) ELSE close END AS vwap,
    #                 turnover_rate, turnover_rate_f, pe, pe_ttm, pb, NULL AS ps, NULL AS ps_ttm,
    #                 dv_ratio, dv_ttm, total_mv, float_mv AS circ_mv,

    #                 -- 指数无此概念，统一补 0
    #                 0 AS industry,
    #                 0 AS is_ST,
    #                 0 AS list_status

    #             FROM '{index_dir}/*.parquet'
    #             WHERE ts_code IS NOT NULL
    #             ORDER BY symbol, date ASC
    #         ) TO '{TEMP_CSV_DIR}' (
    #             FORMAT CSV, HEADER TRUE, PARTITION_BY (symbol), OVERWRITE_OR_IGNORE TRUE
    #         );
    #     """
    #     con.execute(query_index)

    # # 3. 导出 ETF 场内基金数据
    # fund_dir = os.path.join(DATA_ROOT, "fund", "daily").replace("\\", "/")
    # if os.path.exists(os.path.join(DATA_ROOT, "fund", "daily")):
    #     logging.info("  正在处理 ETF 场内基金数据...")
    #     query_fund = f"""
    #         COPY (
    #             SELECT
    #                 strftime(strptime(trade_date::VARCHAR, '%Y%m%d'), '%Y-%m-%d') AS date,
    #                 regexp_replace(ts_code, '^([0-9]+)\.([A-Za-z]+)$', '\\2\\1') AS symbol,

    #                 -- ETF 后复权价格 (Raw * Factor)
    #                 ROUND(open * COALESCE(adj_factor, 1.0), 4) AS open,
    #                 ROUND(high * COALESCE(adj_factor, 1.0), 4) AS high,
    #                 ROUND(low * COALESCE(adj_factor, 1.0), 4) AS low,
    #                 ROUND(close * COALESCE(adj_factor, 1.0), 4) AS close,
    #                 COALESCE(vol * 100, 0.0) AS volume,
    #                 COALESCE(amount * 1000, 0.0) AS amount,
    #                 COALESCE(adj_factor, 1.0) AS factor,

    #                 CASE
    #                     WHEN vol > 0 THEN ROUND(((amount * 1000) / (vol * 100)) * COALESCE(adj_factor, 1.0), 4)
    #                     ELSE ROUND(close * COALESCE(adj_factor, 1.0), 4)
    #                 END AS vwap,

    #                 NULL AS turnover_rate, NULL AS turnover_rate_f, NULL AS pe, NULL AS pe_ttm, NULL AS pb, NULL AS ps, NULL AS ps_ttm,
    #                 NULL AS dv_ratio, NULL AS dv_ttm, total_netasset AS total_mv, net_asset AS circ_mv,

    #                 -- ETF 无个股行业/ST概念，统一补 0
    #                 0 AS industry,
    #                 0 AS is_ST,
    #                 0 AS list_status

    #             FROM '{fund_dir}/*.parquet'
    #             WHERE ts_code IS NOT NULL
    #             ORDER BY symbol, date ASC
    #         ) TO '{TEMP_CSV_DIR}' (
    #             FORMAT CSV, HEADER TRUE, PARTITION_BY (symbol), OVERWRITE_OR_IGNORE TRUE
    #         );
    #     """
    #     con.execute(query_fund)

    # 立即关闭 DuckDB 连接释放句柄
    con.close()
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

    # 清空 DuckDB 生成的分区文件夹 (保留 CSV 文件)
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
    logging.info("--> [步骤 2/2] 调用 Qlib dump_bin 编译生成二进制矩阵 (features/*.bin)...")

    include_fields_str = ",".join(STOCK_FIELDS)
    cmd = [
        sys.executable,
        "-m",
        "scripts.dump_bin",
        "dump_all",
        "--data_path",
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


def convert_index_weight_to_qlib_format(input_parquet, output_txt):
    """
    按截面日期比对的“状态机”算法转换指数权重
    """
    df = pd.read_parquet(input_parquet)
    if df.empty:
        logging.warning("跳过: %s (无数据)", input_parquet)
        return

    # 格式化数据
    df["symbol"] = df["con_code"].apply(to_qlib_symbol)
    df["date"] = pd.to_datetime(df["trade_date"].astype(str)).dt.strftime("%Y-%m-%d")

    MIN_DATE = "2000-01-01"
    MAX_DATE = "2099-12-31"

    # 获取所有升序排列的调仓日
    all_dates = sorted(df["date"].unique())

    active_stocks = {}  # 记录当前在指数中的股票及其起始时间 {symbol: start_date}
    results = []  # 存放最终的区间结果

    # 按照调仓日期，一天一天往后看
    for i, current_date in enumerate(all_dates):
        # 获取当前调仓日的所有股票集合
        current_stocks = set(df[df["date"] == current_date]["symbol"])

        if i == 0:
            # 第一期特例：强制把起始时间拉长到极小值 MIN_DATE
            for sym in current_stocks:
                active_stocks[sym] = MIN_DATE
        else:
            # 1. 寻找被剔除的股票 (在上期的 active 字典里，但不在本次 current_stocks 里)
            removed_stocks = set(active_stocks.keys()) - current_stocks

            # 剔除的截止时间：当前调仓日的“上一天”
            end_date = (pd.to_datetime(current_date, format="%Y-%m-%d") - pd.Timedelta(1, unit="D")).strftime("%Y-%m-%d")

            for sym in removed_stocks:
                # 记录这段区间并归档
                results.append({"code": sym, "start": active_stocks[sym], "end": end_date})
                # 从监控池中移除
                del active_stocks[sym]

            # 2. 寻找新加入的股票 (在本次 current_stocks 里，但不在上期 active 字典里)
            added_stocks = current_stocks - set(active_stocks.keys())
            for sym in added_stocks:
                # 记录新的起始时间为当前调仓日
                active_stocks[sym] = current_date

    # 循环结束后，所有还留在 active_stocks 里的股票，说明至今仍在指数内
    # 强制把结束时间拉长到极大值 MAX_DATE
    for sym, start_date in active_stocks.items():
        results.append({"code": sym, "start": start_date, "end": MAX_DATE})

    # 转为 DataFrame 并排序输出
    result_df = pd.DataFrame(results).sort_values(["code", "start"])

    # 写入 TXT
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as f:
        for _, row in result_df.iterrows():
            f.write(f"{row['code']}\t{row['start']}\t{row['end']}\n")

    logging.info("    转换完成! 共生成 %d 个持有区间 -> %s", len(result_df), output_txt)


def build_index_instruments():
    """
    根据 index/weight/ 目录下的历史权重，生成 Qlib 标准的成分股区间池
    格式: <symbol>\t<start_date>\t<end_date>
    """
    logging.info("--> [步骤 3/4] 正在根据历史权重生成 Qlib 成分股池 (instruments/*.txt)...")

    weight_dir = os.path.join(DATA_ROOT, "index", "weight")
    instruments_dir = os.path.join(QLIB_DIR, "instruments")
    os.makedirs(instruments_dir, exist_ok=True)

    if not os.path.exists(weight_dir):
        logging.warning("未找到 %s 目录，跳过成分股池生成", weight_dir)
        return

    for index_code, out_filename in INDEX_INSTRUMENT_MAP.items():
        weight_file = os.path.join(weight_dir, f"{index_code}.parquet")
        if not os.path.exists(weight_file):
            continue

        logging.info("  正在构建 %s 的成份股池 -> %s...", index_code, out_filename)

        # 写入 instruments/{name}.txt
        target_path = os.path.join(instruments_dir, out_filename)
        convert_index_weight_to_qlib_format(weight_file, target_path)

    logging.info("✅ 股票池 instruments 生成完毕！")


# ==============================================================================
#                 3. 构建板块股票池函数 (instruments/*.txt)
# ==============================================================================


def build_market_sector_instruments():
    """
    将股票基本信息按市场板块转换为 qlib instruments 格式
    注意：本函数依赖 Qlib 生成的 all.txt，因此必须在 run_qlib_dump_bin() 之后调用！
    """
    logging.info("--> [步骤 4/4] 正在基于 all.txt 生成各板块股票池 (zb.txt, cyb.txt, kcb.txt, bjs.txt)...")

    # 配置路径
    basic_file = os.path.join(DATA_ROOT, "meta", "stock_basic.parquet")
    instruments_dir = os.path.join(QLIB_DIR, "instruments")
    all_txt_path = os.path.join(instruments_dir, "all.txt")

    os.makedirs(instruments_dir, exist_ok=True)

    if not os.path.exists(basic_file):
        logging.warning("缺少 stock_basic.parquet，跳过板块股票池生成")
        return
    if not os.path.exists(all_txt_path):
        logging.warning("缺少 all.txt (可能 dump_bin 未成功)，跳过板块股票池生成")
        return

    # 1. 读取股票基础元数据
    df_market = pd.read_parquet(basic_file, columns=["ts_code", "market"])
    df_market["instrument"] = df_market["ts_code"].apply(to_qlib_symbol)
    df_market = df_market.drop(columns=["ts_code"])

    # 2. 读取 dump_bin 生成的 all.txt
    df_all = pd.read_csv(
        all_txt_path,
        sep="\t",
        header=None,
        names=["instrument", "start_date", "end_date"],
    )

    # 3. 横向合并打上 market 标签
    df_all_market = df_all.merge(df_market, on="instrument", how="left").dropna(subset=["market"])

    # 4. 分板块输出
    for market_name, out_filename in SECTOR_INSTRUMENT_MAP.items():
        # 筛选特定市场的股票
        df_filtered = df_all_market[df_all_market["market"] == market_name].copy()

        if df_filtered.empty:
            logging.warning("跳过: %s (无数据)", market_name)
            continue

        # 写入 txt 文件
        output_txt = os.path.join(instruments_dir, out_filename)
        with open(output_txt, "w", encoding="utf-8") as f:
            for _, row in df_filtered.iterrows():
                f.write(f"{row['instrument']}\t{row['start_date']}\t{row['end_date']}\n")

        logging.info("  已生成 %s 板块股票池 -> %s (共 %d 只股票)", market_name, out_filename, len(df_filtered))

    logging.info("✅ 板块股票池生成完毕！")


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

    # # 3. 构建动态指数股票池
    build_index_instruments()

    # # 4. 构建板块股票池 (主板 zb, 创业板 cyb, 科创板 kcb, 北交所 bjs)
    build_market_sector_instruments()

    # # 5. 清理临时 CSV 文件
    # if clean_temp_csv and os.path.exists(TEMP_CSV_DIR):
    #     logging.info("--> 正在清理临时 CSV 文件夹...")
    #     shutil.rmtree(TEMP_CSV_DIR)

    logging.info("=" * 60)
    logging.info("🎉 全流程构建完成！可以直接在 Qlib 中无缝初始化并读取数据。")
    logging.info("Qlib 初始化路径: qlib.init(provider_uri='%s')", QLIB_DIR)


if __name__ == "__main__":
    # clean_temp_csv=True 会在生成二进制后自动删除中间 CSV，节省磁盘空间
    main(clean_temp_csv=False)
