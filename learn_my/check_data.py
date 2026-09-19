import os

import duckdb
import pandas as pd
from tqdm import tqdm


def check_stock_data(
    stock_fields: list[str],
    data_dir: str,
    start_time: str | None = None,
    end_time: str | None = None,
    max_missing_stocks_per_field: int | None = None,
):
    """
    日期区间为闭区间。

    示例：
    check_stock_data(stock_fields, data_dir)                         # 全量
    check_stock_data(stock_fields, data_dir, "20200101")             # 2020年至今
    check_stock_data(stock_fields, data_dir, end_time="20201231")    # 截至2020年底
    check_stock_data(stock_fields, data_dir, "20200101", "20201231") # 2020年

    max_missing_stocks_per_field:
        每个字段最多展示多少只存在缺失的股票；None 表示全部展示。
    """
    start_date = pd.to_datetime(start_time).normalize() if start_time is not None else None
    end_date = pd.to_datetime(end_time).normalize() if end_time is not None else None

    if start_date is not None and end_date is not None:
        if start_date > end_date:
            raise ValueError("start_time 不能晚于 end_time")

    # 累计每个待检查字段的 NaN 总行数
    global_field_na = {f: 0 for f in stock_fields}
    # 累计所选时间范围内成功读取的数据总行数
    global_total_rows = 0
    # 累计任意待检查字段存在 NaN 的数据行数；同一行只计一次
    global_any_na_rows = 0
    # 记录每个字段在哪些文件中整列全部为 NaN
    all_null_files_by_field = {f: [] for f in stock_fields}
    # 记录每个字段在哪些文件中整列全部一样
    constant_files_by_field = {f: [] for f in stock_fields}
    # 按字段、股票累计缺失行数以及首尾缺失日期
    missing_ranges_by_field = {f: {} for f in stock_fields}
    # 记录成功读取但不包含任何数据行的空表文件
    empty_files = []
    # 记录因损坏、格式异常等原因读取失败的文件
    error_files = []
    # 统计通过日期范围和文件类型过滤的 Parquet 文件数量
    matched_files = 0

    for file in tqdm(sorted(os.listdir(data_dir))):
        if not file.endswith(".parquet"):
            continue

        file_path = os.path.join(data_dir, file)
        if not os.path.isfile(file_path):
            continue

        # 文件名格式：YYYYMMDD.parquet
        file_date = pd.to_datetime(
            os.path.splitext(file)[0],
            format="%Y%m%d",
            errors="coerce",
        )

        if pd.isna(file_date):
            print(f"[WARN] 无法从文件名解析日期，跳过: {file}")
            continue

        # 闭区间日期过滤
        if start_date is not None and file_date < start_date:
            continue
        if end_date is not None and file_date > end_date:
            continue

        matched_files += 1

        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            print(f"[WARN] 读取失败 {file}: {e}")
            error_files.append(file)
            continue

        if df.empty:
            empty_files.append(file)
            print(f"[WARN] {file} 是空表")
            continue

        missing_fields = sorted(set(stock_fields) - set(df.columns))
        if missing_fields:
            print(f"[WARN] {file} 缺少字段: {missing_fields}")

        exist_fields = [field for field in stock_fields if field in df.columns]

        if not exist_fields:
            print(f"[WARN] {file} 不存在任何待检测字段，跳过")
            continue

        sub = df[exist_fields]
        global_total_rows += len(df)

        # 文件缺少整个字段时，按该文件所有行均为 NaN 处理。这与
        # DuckDB read_parquet(..., union_by_name=true) 的行为一致。
        any_na_mask = sub.isna().any(axis=1)
        if missing_fields:
            any_na_mask = pd.Series(True, index=df.index)

        field_na_cnt = sub.isna().sum()
        for field in exist_fields:
            na_count = int(field_na_cnt[field])
            global_field_na[field] += na_count

            if na_count == len(sub):
                all_null_files_by_field[field].append(file)

            # 整列只有一个唯一值，包括对 NaN 的判断
            elif field in df.columns and df[field].nunique(dropna=False) == 1:
                constant_files_by_field[field].append(file)

        for field in missing_fields:
            global_field_na[field] += len(df)
            all_null_files_by_field[field].append(file)

        global_any_na_rows += int(any_na_mask.sum())

    # 一次性按股票聚合所有缺失字段。相比在 Python 中逐条累计数百万条
    # 缺失记录，这种方式明显更快，并且支持 union_by_name 的异构 Parquet。
    detail_fields = [field for field in stock_fields if global_field_na[field] > 0]
    if detail_fields:
        parquet_pattern = os.path.join(data_dir, "*.parquet").replace("\\", "/").replace("'", "''")
        date_expr = "CAST(strptime(trade_date::VARCHAR, '%Y%m%d') AS DATE)"
        aggregates = []

        for index, field in enumerate(detail_fields):
            quoted_field = '"' + field.replace('"', '""') + '"'
            missing_condition = f"({quoted_field} IS NULL OR isnan({quoted_field}))"
            aggregates.extend(
                [
                    f"count(*) FILTER (WHERE {missing_condition}) AS f{index}_count",
                    f"min({date_expr}) FILTER (WHERE {missing_condition}) AS f{index}_first",
                    f"max({date_expr}) FILTER (WHERE {missing_condition}) AS f{index}_last",
                ]
            )

        detail_conditions = []
        if start_date is not None:
            detail_conditions.append(f"{date_expr} >= DATE '{start_date:%Y-%m-%d}'")
        if end_date is not None:
            detail_conditions.append(f"{date_expr} <= DATE '{end_date:%Y-%m-%d}'")
        detail_where = f"WHERE {' AND '.join(detail_conditions)}" if detail_conditions else ""

        con = duckdb.connect()
        try:
            detail_df = con.execute(f"""
                SELECT
                    coalesce(ts_code, '<ts_code为空>') AS ts_code,
                    {', '.join(aggregates)}
                FROM read_parquet('{parquet_pattern}', union_by_name=true)
                {detail_where}
                GROUP BY ts_code
                ORDER BY ts_code
                """).fetchdf()
        finally:
            con.close()

        for _, row in detail_df.iterrows():
            for index, field in enumerate(detail_fields):
                missing_rows = int(row[f"f{index}_count"])
                if missing_rows == 0:
                    continue

                missing_ranges_by_field[field][str(row["ts_code"])] = {
                    "missing_rows": missing_rows,
                    "first_missing_date": pd.Timestamp(row[f"f{index}_first"]),
                    "last_missing_date": pd.Timestamp(row[f"f{index}_last"]),
                }

    print("=" * 60)
    print(
        "统计区间:",
        start_date.strftime("%Y%m%d") if start_date is not None else "最早",
        "~",
        end_date.strftime("%Y%m%d") if end_date is not None else "最新",
    )
    print(f"匹配文件数量: {matched_files:,}")
    print(f"全部数据总行数: {global_total_rows:,}")
    print(f"任意字段存在空值的总行数: {global_any_na_rows:,}")

    if global_total_rows:
        print("任意字段存在空值的总行数占比: " f"{global_any_na_rows / global_total_rows:.4%}")

    print("=" * 60)
    print("【各字段空值统计】")

    for field in stock_fields:
        na_count = global_field_na[field]
        na_ratio = na_count / global_total_rows if global_total_rows else 0
        print(f"{field:<18} | NaN行数:{na_count:<10,} " f"| 空值占比: {na_ratio:.4%}")

    print("=" * 60)
    print("【各字段缺失详情】")
    print("说明：首个~最后缺失日是总体覆盖范围，范围内的缺失日期可能不连续。")

    has_missing_detail = False
    for field in stock_fields:
        field_ranges = missing_ranges_by_field[field]
        if not field_ranges:
            continue

        has_missing_detail = True

        detail_rows = [
            {
                "ts_code": symbol,
                "missing_rows": stats["missing_rows"],
                "first_missing_date": stats["first_missing_date"].strftime("%Y%m%d"),
                "last_missing_date": stats["last_missing_date"].strftime("%Y%m%d"),
            }
            for symbol, stats in field_ranges.items()
        ]
        detail_df = pd.DataFrame(detail_rows).sort_values(
            ["missing_rows", "ts_code"],
            ascending=[False, True],
        )

        if max_missing_stocks_per_field is None:
            display_df = detail_df
        else:
            display_df = detail_df.head(max_missing_stocks_per_field)

        print("-" * 60)
        print(f"字段: {field} - 缺失数量：{global_field_na[field]}")
        print(f"  └─ 涉及股票: {len(detail_df):,} 只；" "以下为缺失总体范围（首日~末日，期间可能不连续）")
        print(display_df.to_string(index=False, header=True).replace("\n", "\n     "))

        hidden_count = len(detail_df) - len(display_df)
        if hidden_count > 0:
            print(f"     ……另有 {hidden_count:,} 只未展示")

    if not has_missing_detail:
        print("无")

    print("=" * 60)
    print("【单个文件内整列为空】")
    flag_1 = False
    for field, files in all_null_files_by_field.items():
        if files:
            flag_1 = True
            print(f"{field:<18} | 整列为空文件数:{len(files):,} | 文件: {files}")
    if not flag_1:
        print("无")

    print("=" * 60)
    print("【单个文件内整列相同】")
    flag_2 = False
    for field, files in constant_files_by_field.items():
        if files:
            flag_2 = True
            print(f"{field:<18} | 所有值相同的文件数: {len(files):,} | 文件: {files}")
    if not flag_2:
        print("无")

    if error_files:
        print("=" * 60)
        print(f"读取失败文件数量: {len(error_files)}")
    if empty_files:
        print(f"空表文件数量: {len(empty_files)}")

    return missing_ranges_by_field


STOCK_FIELDS = [
    "open",
    "high",
    "low",
    "close",
    "change",
    "vol",
    "amount",
    "adj_factor",
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
    "limit_status",
]


def check_stock_data_sql_where(data_dir):
    """检查转换为临时tmp_csv时，sql条件到底排除了多少条数据"""
    """ 
    WHERE t.ts_code IS NOT NULL
    -- 股票复权因子缺失时不能用 1.0 代替。
    AND t.adj_factor IS NOT NULL
    AND t.adj_factor > 0
    """

    cnt_ts_code_is_na = 0
    cnt_adj_factor_is_na = 0
    cnt_adj_factor_non_positive = 0
    cnt_filtered_rows = 0
    cnt_total_rows = 0

    for file in tqdm(os.listdir(data_dir)):
        if not file.endswith(".parquet"):
            continue
        # 只读取需要检查的列，速度更快
        df = pd.read_parquet(os.path.join(data_dir, file), columns=["ts_code", "adj_factor"])

        ts_code_is_na = df["ts_code"].isna()
        adj_factor_is_na = df["adj_factor"].isna()
        adj_factor_non_positive = df["adj_factor"].le(0)

        cnt_total_rows += len(df)
        cnt_ts_code_is_na += int(ts_code_is_na.sum())
        cnt_adj_factor_is_na += int(adj_factor_is_na.sum())
        cnt_adj_factor_non_positive += int(adj_factor_non_positive.sum())

        # 对应 SQL WHERE 会排除的行，使用 OR 避免重复计数
        filtered = ts_code_is_na | adj_factor_is_na | adj_factor_non_positive
        cnt_filtered_rows += int(filtered.sum())

    print("总行数:", cnt_total_rows)
    print("ts_code 为 NA:", cnt_ts_code_is_na)
    print("adj_factor 为 NA:", cnt_adj_factor_is_na)
    print("adj_factor <= 0:", cnt_adj_factor_non_positive)
    print("SQL 条件总计排除:", cnt_filtered_rows)


def check_temp_csv_with_stock_data():
    """
    用来检查生成的tmp_csv_dir中数据的行数是不是和预期的一致
    """
    stock_dir = "C:\\Users\\WANGKANG\\.qlib\\qlib_data\\stock_data_warehouse\\stock\\daily"
    index_dir = "C:\\Users\\WANGKANG\\.qlib\\qlib_data\\stock_data_warehouse\\index\\daily"
    tmp_csv_dir = "C:\\Users\\WANGKANG\\.qlib\\qlib_data\\stock_data_warehouse\\temp_csv_for_qlib"
    stock_data_num = 0
    index_data_num = 0
    tmp_csv_data_num = 0
    for file in tqdm(os.listdir(stock_dir)):
        df = pd.read_parquet(os.path.join(stock_dir, file))
        stock_data_num += len(df)

    for file in tqdm(os.listdir(index_dir)):
        df = pd.read_parquet(os.path.join(index_dir, file))
        index_data_num += len(df)

    for file in tqdm(os.listdir(tmp_csv_dir)):
        df = pd.read_csv(os.path.join(tmp_csv_dir, file))
        tmp_csv_data_num += len(df)

    print("stock_data_num: ", stock_data_num)
    print("index_data_num: ", index_data_num)
    print("tmp_csv_data_num: ", tmp_csv_data_num)

    print(f"stock_data_num + index_data_num - tmp_csv_data_num = {stock_data_num + index_data_num - tmp_csv_data_num}")

    return tmp_csv_data_num


def check_qlib_data():
    import qlib
    from qlib.data import D

    qlib_data_dir = "C:\\Users\\WANGKANG\\.qlib\\qlib_data\\my_tushare_data"
    qlib.init(
        provider_uri=qlib_data_dir,
        region=qlib.config.REG_CN,
    )
    df = D.features(
        instruments=D.instruments("all"),
        fields=["$close"],
    )
    # Qlib 会按全局交易日历把每个标的起止日期之间的数据 reindex，
    # 停牌或无交易记录的日期也会占一行，但字段值为 NaN。因此 len(df)
    # 不能直接与原始 CSV 行数比较；close 非空行数才是实际行情记录数。
    calendar_rows = len(df)
    actual_data_rows = int(df["$close"].notna().sum())
    calendar_padding_rows = int(df["$close"].isna().sum())

    print("Qlib 日历展开后总行数:", calendar_rows)
    print("Qlib $close 非空实际数据行数:", actual_data_rows)
    print("Qlib 日历补齐的 NaN 占位行数:", calendar_padding_rows)

    return actual_data_rows


if __name__ == "__main__":
    data_dir = r"C:\Users\WANGKANG\.qlib\qlib_data\stock_data_warehouse\stock\daily"
    check_stock_data(STOCK_FIELDS, data_dir, max_missing_stocks_per_field=5)
    check_stock_data_sql_where(data_dir)
    # # 检查tmp_csv数据是不是仅仅缺失了SQL中少的那么多行数据
    tmp_csv_data_num = check_temp_csv_with_stock_data()
    qlib_actual_data_num = check_qlib_data()
    print("tmp CSV 与 Qlib 实际数据行数之差:", tmp_csv_data_num - qlib_actual_data_num)
