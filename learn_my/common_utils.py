import numpy as np
import pandas as pd

""" 
行	含义	翻译
① count 4040.0	有效交易日总天数	你的数据里一共有 4040 个交易日
② mean 3690.58	均值	平均每个交易日，有约 3691 只股票
③ std 1212.09	标准差	每日股票数量波动大，标准差 1212，说明不同交易日股票数量差异很大
④ min 1694.0	最小值	最少的一个交易日，只有 1694 只股票（大概率早期老数据）
⑤ 25% 2523.0	下四分位数	25% 的交易日，股票数量 ≤2523 只
⑥ 50% 3683.5	中位数	一半交易日 ≤3683 只，一半交易日 ≥3683 只；和均值很接近，分布比较对称
⑦ 75% 4960.0	上四分位数	75% 的交易日，股票数量 ≤4960 只
⑧ max 5542.0	最大值	最多的交易日，同时有 5542 只股票
 """


def count_daily_stocks(df, datetime_col="datetime", instrument_col="instrument"):
    """统计每天的股票数量"""
    df_reset = df.reset_index()
    daily_counts = df_reset.groupby(datetime_col)[instrument_col].count()
    res = daily_counts.describe()
    return res


def convert_code(code):
    """将股票代码转换为统一格式：000001.SH -> SH000001  / 000001.SH.csv -> SH000001"""
    if code.endswith(".csv"):
        code = code[:-4]
    code_, market = code.split(".", 1)
    return f"{market}{code_}"


def is_ST(code: str):
    """判断股票是否为ST: 0不是 1是"""
    return 1 if "ST" in code.upper() else 0


def list_status_to_num(list_status: str):
    """将股票状态转成数字"""
    list_status_map = {
        "L": 0,  # 上市
        "D": 1,  # 退市
        "P": 2,  # 暂停上市
    }
    return list_status_map.get(str(list_status).strip().upper(), -1)


STOCK_BASIC_PATH = r"C:\Users\WANGKANG\Desktop\量化交易学习\stock_data\stock_basic.csv"


def get_stock_basic_info():
    """获取股票基本信息"""
    df = pd.read_csv(STOCK_BASIC_PATH, usecols=["ts_code", "name", "list_status", "list_date", "delist_date"])
    df["instrument"] = df["ts_code"].apply(convert_code)
    df["is_ST"] = df["name"].apply(is_ST)
    df["list_status"] = df["list_status"].apply(list_status_to_num)
    df = df[["instrument", "name", "is_ST", "list_status", "list_date", "delist_date"]]

    return df.set_index("instrument")


def filter_stocks(df, additional_info=False):
    """
    剔除退市、ST股
    :param df: 原始行情df，必须包含instrument
    :param additional_info: True 返回结果带上合并的基础信息列；False 仅过滤，输出和输入列保持一致
    :return: 过滤后的DataFrame
    """
    df_info = get_stock_basic_info()
    index_names = df.index.names

    # 左合并，把基础信息接入原始行情
    df_merged = df.reset_index().merge(df_info.reset_index(), on="instrument", how="left")

    list_status_map = {
        "L": 0,  # 上市
        "D": 1,  # 退市
        "P": 2,  # 暂停上市
    }

    # ========== 过滤条件 ==========
    # list_status == 'L'：正常上市；is_ST==False：剔除ST
    mask = (df_merged["list_status"] == list_status_map["L"]) & (df_merged["is_ST"] == 0)
    df_filtered = df_merged[mask].copy()
    df_filtered = df_filtered.set_index(index_names)
    df_filtered = df_filtered.sort_index()

    if not additional_info:
        # 只保留原始输入df的列
        origin_cols = df.columns.tolist()
        df_filtered = df_filtered[origin_cols]

    return df_filtered


def load_industry_dummy():
    """加载行业数据"""
    # 读取股票基本信息
    stock_basic = pd.read_csv(STOCK_BASIC_PATH, usecols=["ts_code", "industry"])
    # 将ts_code重命名为instrument
    stock_basic = stock_basic.rename(columns={"ts_code": "instrument"})
    # 去除industry中的空格和换行符
    stock_basic["industry"] = stock_basic["industry"].astype(str).str.strip()
    stock_basic.loc[stock_basic["industry"].isin(["", "nan"]), "industry"] = "other"
    # 去除重复的instrument代码
    stock_basic = stock_basic.drop_duplicates(subset=["instrument"])
    # 将ts_code转换为instrument代码
    stock_basic["instrument"] = stock_basic["instrument"].apply(convert_code)

    industry_map = stock_basic.set_index("instrument")["industry"]
    industry_dummy = pd.get_dummies(industry_map, prefix="Ind")
    industry_dummy = industry_dummy.reindex(sorted(industry_dummy.columns), axis=1)
    return industry_dummy


def load_industry_code():
    """
    加载行业数据并将行业名称编码为数字（便于写入 qlib）。

    返回：
        industry_code : pd.Series，index=instrument，value=行业整数编码(int)
        code_to_name  : dict，行业整数编码 -> 行业名称（编码字典，便于反查）
    """
    # 读取股票基本信息
    stock_basic = pd.read_csv(STOCK_BASIC_PATH, usecols=["ts_code", "industry"])
    # 将ts_code重命名为instrument
    stock_basic = stock_basic.rename(columns={"ts_code": "instrument"})
    # 去除industry中的空格和换行符
    stock_basic["industry"] = stock_basic["industry"].astype(str).str.strip()
    stock_basic.loc[stock_basic["industry"].isin(["", "nan"]), "industry"] = "other"
    # 去除重复的instrument代码
    stock_basic = stock_basic.drop_duplicates(subset=["instrument"])
    # 将ts_code转换为instrument代码
    stock_basic["instrument"] = stock_basic["instrument"].apply(convert_code)

    uniques = sorted(stock_basic["industry"].unique())
    name_to_code = {name: i for i, name in enumerate(uniques)}
    stock_basic["industry_code"] = stock_basic["industry"].map(name_to_code).astype("int32")

    industry_code = stock_basic.set_index("instrument")["industry_code"]
    return industry_code, name_to_code


def build_style_factors(df, mv_col="$total_mv", industry_col="$industry", log_mv_name="log_mv", dummy_prefix="Ind", drop_first=False):
    """
    构建行业+市值风格因子（用于中性化）。

    参数
    ----
    df           : DataFrame  行情/因子数据，MultiIndex=(instrument, datetime)，
                              必须包含市值列与行业编码列
    mv_col       : str  市值列名（默认 '$total_mv'）
    industry_col : str  行业整数编码列名（默认 '$industry'）
    log_mv_name  : str  输出的对数市值列名（默认 'log_mv'）
    dummy_prefix : str  行业哑变量列名前缀（默认 'Ind'）
    drop_first   : bool 是否丢弃第一个哑变量列以避免共线性（线性回归中性化建议 True）

    返回
    ----
    DataFrame  与 df 同 index，列 = [log_mv] + 各行业哑变量列
    """
    """
    构建行业+市值风格因子（用于中性化）。
    """
    # 1. 对数市值：防止 0 导致负无穷 (-inf)，将 0 和负数替换为 NaN
    mv = df[mv_col].copy()
    mv = mv.where(mv > 0, np.nan)
    log_mv = np.log(mv).rename(log_mv_name)

    # 2. 行业哑变量：行业编码转整数类别，避免被当连续值
    df_industry = pd.get_dummies(
        df[industry_col].astype("Int32"),  # 使用大写的 "Int32"（支持 NaN 的可空整数类型）
        prefix=dummy_prefix,
        drop_first=drop_first,
        dtype="int8",  # 节省内存
    )

    # 3. 安全联动：如果某只股票本身缺失行业代码，将其对数市值也置为 NaN
    # 这样 CSNeutralize 就会把这只股票识别为无效样本，不会参与截面中性化回归
    log_mv[df[industry_col].isna()] = np.nan

    # 4. 按 index 对齐拼接（共享同一 MultiIndex，比 merge 更安全）
    df_style = pd.concat([log_mv, df_industry], axis=1)

    assert len(df_style) == len(df), "拼接后行数与原始 df 不一致"
    return df_style


def neutralize_daily(daily_df, factor_cols, style_cols):
    from sklearn.linear_model import LinearRegression

    """
    每日横截面中性化函数
    Y (因子) = beta * X (市值 + 行业) + 残差 (纯净Alpha)
    """
    # 确保数据在进入中性化前就已经处理好了
    assert not daily_df.isna().any(axis=1).any(), "daily_df 存在缺失值"

    # 若当天有效股票过少，不具有统计回归意义
    if len(daily_df) < 50:
        return pd.DataFrame(index=daily_df.index, columns=factor_cols, dtype=float)

    X = daily_df[style_cols].values
    Y = daily_df[factor_cols].values

    # 极速多输出线性回归
    model = LinearRegression(fit_intercept=True)
    model.fit(X, Y)

    # 核心：计算残差 (Residuals)
    residuals = Y - model.predict(X)

    return pd.DataFrame(residuals, index=daily_df.index, columns=factor_cols)


def mad_zscore_df(df, n=3):
    """
    截面批量预处理：MAD 3倍中位数去极值 + Z-Score 标准化
    (矩阵向量化，含空值与常数列保护)

    参数
    ----
    df : DataFrame  单个截面（同一天）的因子矩阵，index=股票，columns=因子
    n  : float  MAD 去极值倍数，默认 3

    返回
    ----
    DataFrame  处理后的因子矩阵（NaN 已填 0）
    """
    if df.empty:
        return df

    # 1. 截面中位数与 MAD
    med = df.median(axis=0)
    mad = df.sub(med, axis=1).abs().median(axis=0)

    # 2. 上下界（1.4826 使 MAD 在正态下等价于标准差）
    upper = med + n * 1.4826 * mad
    lower = med - n * 1.4826 * mad

    # 3. 截断极值（Series 沿列方向广播）
    clipped = df.clip(lower=lower, upper=upper, axis=1)

    # 4. Z-Score 标准化，常数列/近零方差列除数置 1（结果为 0）
    mean = clipped.mean(axis=0)
    std = clipped.std(axis=0).mask(lambda s: s < 1e-8, 1.0)
    zscored = clipped.sub(mean, axis=1).div(std, axis=1)

    # 5. 残余 NaN 填 0（中性水平）
    return zscored.fillna(0.0)


def ic_report(daily_ic, alpha=0.05, fdr_method="fdr_bh", verbose=True):
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    """
    因子 IC 综合报告（含显著性检验）

    参数
    ----
    daily_ic : DataFrame
        行=日期，列=因子名，值=当日截面 Rank IC
    alpha : float
        显著性水平，默认 0.05
    fdr_method : str
        多重假设校正方法，默认 'fdr_bh'（Benjamini-Hochberg FDR）
    verbose : bool
        是否打印显著因子占比

    返回
    ----
    DataFrame  index=因子名，列包含：
        Rank_IC     : IC 均值
        ICIR        : IC 均值 / IC 标准差
        Abs_IC      : |IC| 均值
        IC_Std      : IC 标准差
        T_Days      : 有效样本天数
        t_stat      : t 统计量 = ICIR * sqrt(T)
        raw_p_value : 单因子双侧 t 检验 p 值
        adj_p_value : FDR 校正后 p 值
        is_significant : 是否通过校正后显著性检验
    """
    ic_mean = daily_ic.mean()
    ic_std = daily_ic.std()
    abs_ic = daily_ic.abs().mean()
    # 各因子有效样本天数（逐列非 NaN 计数）
    t_days = daily_ic.count()

    icir = ic_mean / ic_std.replace(0, np.nan)
    t_stat = icir * np.sqrt(t_days)
    # 双侧 t 检验，自由度 df = T - 1
    raw_p = pd.Series(
        2 * (1 - stats.t.cdf(t_stat.abs(), df=(t_days - 1).clip(lower=1))),
        index=daily_ic.columns,
    )

    # FDR 多重假设校正
    valid = raw_p.notna()
    adj_p = pd.Series(np.nan, index=daily_ic.columns)
    if valid.sum() > 0:
        _, adj_vals, _, _ = multipletests(raw_p[valid].values, alpha=alpha, method=fdr_method)
        adj_p[valid] = adj_vals

    report = pd.DataFrame(
        {
            "Rank_IC": ic_mean,
            "ICIR": icir,
            "Abs_IC": abs_ic,
            "IC_Std": ic_std,
            "T_Days": t_days,
            "t_stat": t_stat,
            "raw_p_value": raw_p,
            "adj_p_value": adj_p,
        }
    )
    report["is_significant"] = report["adj_p_value"] < alpha
    report = report.sort_values("ICIR", key=lambda s: s.abs(), ascending=False)

    if verbose:
        n_sig = int(report["is_significant"].sum())
        n_all = len(report)
        print(f"显著因子：{n_sig}/{n_all}（{n_sig / max(n_all, 1):.1%}），alpha={alpha}，校正={fdr_method}")

    return report


def factor_backtest(df_neutralized, df_label, label_col="label", ic_series=None, n_group=5, cost_rate=0.0013, periods_per_year=252, min_stocks=10, holding_days=1):
    """
    因子五分位多空回测（每个因子逐个计算年化收益 / 波动 / Sharpe）

    参数
    ----
    df_neutralized : DataFrame
        MultiIndex(datetime, instrument)，列=因子残差
    df_label : DataFrame
        MultiIndex(datetime, instrument)，含未来收益列 label_col
    label_col : str
        df_label 中的收益列名，默认 'label'
    ic_series : Series or None
        各因子 Rank_IC 均值，用于决定多空方向；
        为 None 时对每个因子按自身 IC 符号自动判定（默认全部按正向）
    n_group : int
        分组数，默认 5（Q1~Q5）
    cost_rate : float
        单边交易成本，默认 0.0013（13bps）；每次调仓多空双边扣 2*cost_rate
    periods_per_year : int
        年化期数（每年调仓次数）：日频 252 / 周频 52 / 月频 12 / 季频 4
    min_stocks : int
        每个调仓截面最少股票数，不足则跳过该期
    holding_days : int
        持有天数（调仓间隔）。标签是 N 日未来收益时，必须每隔 N 天调一次仓，
        否则逐日采样会重复计算重叠收益。日频=1 / 周频=5 / 月频=20 / 季频=60。
        采样时每隔 holding_days 个交易日取一个截面，成本每次调仓只扣一次。

    返回
    ----
    DataFrame  index=因子名，列：Long_Return / Short_Return / Net_Return / Net_Vol / Net_Sharpe
    """
    factor_cols = list(df_neutralized.columns)
    df = df_neutralized.join(df_label[[label_col]], how="inner").dropna(subset=[label_col])

    # 按 holding_days 抽取调仓日：避免 N 日标签逐日采样导致的重叠收益重复计算
    all_dates = df.index.get_level_values("datetime").unique().sort_values()
    rebalance_dates = all_dates[::holding_days]
    df = df[df.index.get_level_values("datetime").isin(rebalance_dates)]

    results = {}
    for factor in factor_cols:
        direction = 1
        if ic_series is not None and factor in ic_series.index:
            direction = 1 if ic_series[factor] >= 0 else -1

        sub = df[[factor, label_col]].dropna()
        long_id = n_group - 1 if direction == 1 else 0
        short_id = 0 if direction == 1 else n_group - 1

        def calc_legs(daily):
            if len(daily) < min_stocks:
                return (np.nan, np.nan)
            grp = pd.qcut(daily[factor].rank(method="first"), q=n_group, labels=False)
            long_ret = daily.loc[grp == long_id, label_col].mean()
            short_ret = daily.loc[grp == short_id, label_col].mean()
            return (long_ret, short_ret)

        legs_raw = sub.groupby(level="datetime").apply(calc_legs)
        legs = pd.DataFrame(legs_raw.tolist(), index=legs_raw.index, columns=["long", "short"]).dropna()
        if len(legs) < 6:
            results[factor] = {
                "Long_Return": np.nan,
                "Short_Return": np.nan,
                "Net_Return": np.nan,
                "Net_Vol": np.nan,
                "Net_Sharpe": np.nan,
            }
            continue

        long_daily = legs["long"]
        short_daily = legs["short"]
        # 多空(Top-Bottom)日度收益，扣双边成本；多头/空头单腿各扣单边成本
        ls_net = (long_daily - short_daily) - cost_rate * 2
        long_net = long_daily - cost_rate
        short_net = short_daily - cost_rate

        ann_ret = ls_net.mean() * periods_per_year
        ann_vol = ls_net.std() * np.sqrt(periods_per_year)
        sharpe = ann_ret / (ann_vol + 1e-8)
        results[factor] = {
            "Long_Return": long_net.mean() * periods_per_year,
            "Short_Return": short_net.mean() * periods_per_year,
            "Net_Return": ann_ret,
            "Net_Vol": ann_vol,
            "Net_Sharpe": sharpe,
        }

    return pd.DataFrame.from_dict(results, orient="index")


def factor_scorecard(daily_ic, df_neutralized, df_label, label_col="label", alpha=0.05, fdr_method="fdr_bh", n_group=5, cost_rate=0.0013, periods_per_year=252, holding_days=1, verbose=True):
    """
    因子综合成绩单：IC 报告 + 多空回测合并

    参数
    ----
    holding_days : int  调仓间隔（交易日）。日频=1 / 周频=5 / 月频=20 / 季频=60，
                        须与 label_col 的未来收益周期、periods_per_year 保持一致。

    返回
    ----
    DataFrame  含 Rank_IC / ICIR / Abs_IC / 显著性 / Net_Return / Net_Sharpe 等
    """
    report = ic_report(daily_ic, alpha=alpha, fdr_method=fdr_method, verbose=verbose)
    bt = factor_backtest(
        df_neutralized,
        df_label,
        label_col=label_col,
        ic_series=report["Rank_IC"],
        n_group=n_group,
        cost_rate=cost_rate,
        periods_per_year=periods_per_year,
        holding_days=holding_days,
    )
    return report.join(bt)


def calc_daily_ic(df_neutralized, df_label, label_col="label"):
    """
    计算每日截面 Rank IC（spearman）

    参数
    ----
    df_neutralized : DataFrame
        MultiIndex(datetime, instrument)，列=因子(残差)
    df_label : DataFrame
        MultiIndex(datetime, instrument)，含未来收益列 label_col
    label_col : str
        df_label 中的收益列名，默认 'label'

    返回
    ----
    DataFrame  行=日期，列=因子名，值=当日截面 Rank IC
    """
    factor_cols = list(df_neutralized.columns)
    df_ic = df_neutralized.join(df_label[[label_col]], how="inner").dropna(subset=[label_col])

    def _cross_ic(daily):
        return daily[factor_cols].corrwith(daily[label_col], method="spearman")

    return df_ic.groupby(level="datetime", group_keys=True).apply(_cross_ic)


def factor_scorecard_by_mktcap(df_neutralized, df_label, mktcap, label_col="label", n_mktcap_group=5, alpha=0.05, fdr_method="fdr_bh", n_group=5, cost_rate=0.0013, periods_per_year=252, verbose=True):
    """
    报告2：按市值分组，每组各自重复报告1（IC 报告 + 多空回测）

    做法：每个交易日截面上，按市值把股票分成 n_mktcap_group 组，
    每组取出对应 (datetime, instrument) 子样本，在组内独立计算
    daily_ic 与综合成绩单，最后汇总。

    参数
    ----
    df_neutralized : DataFrame
        MultiIndex(datetime, instrument)，列=因子(残差)
    df_label : DataFrame
        MultiIndex(datetime, instrument)，含未来收益列 label_col
    mktcap : Series
        MultiIndex(datetime, instrument)，市值(可用 log 市值)，用于分组
    n_mktcap_group : int
        市值分组数，默认 5（G0=小市值 ... G4=大市值）
    其余参数同 factor_scorecard

    返回
    ----
    (combined, group_reports)
        combined      : DataFrame，长表，含 MktCap_Group 列 + 因子索引 + 成绩单各列
        group_reports : dict{组号: 该组 scorecard DataFrame}
    """
    mktcap = mktcap.copy()
    mktcap.name = "_mktcap_"

    # 每日截面按市值分组
    def _assign_group(s):
        if len(s) < n_mktcap_group:
            return pd.Series(np.nan, index=s.index)
        return pd.Series(
            pd.qcut(s.rank(method="first"), q=n_mktcap_group, labels=False),
            index=s.index,
        )

    grp = mktcap.groupby(level="datetime", group_keys=False).apply(_assign_group)
    grp = grp.dropna().astype(int)

    group_reports = {}
    combined_list = []
    for g in range(n_mktcap_group):
        # 该市值组的 (datetime, instrument) 索引
        idx = grp[grp == g].index
        sub_factor = df_neutralized.loc[df_neutralized.index.intersection(idx)]
        if sub_factor.empty:
            continue

        if verbose:
            print(f"===== 市值组 G{g}（样本 {len(sub_factor)}）=====")

        daily_ic_g = calc_daily_ic(sub_factor, df_label, label_col=label_col)
        card = factor_scorecard(
            daily_ic_g,
            sub_factor,
            df_label,
            label_col=label_col,
            alpha=alpha,
            fdr_method=fdr_method,
            n_group=n_group,
            cost_rate=cost_rate,
            periods_per_year=periods_per_year,
            verbose=verbose,
        )
        group_reports[g] = card

        card_long = card.copy()
        card_long.insert(0, "MktCap_Group", g)
        combined_list.append(card_long)

    combined = pd.concat(combined_list) if combined_list else pd.DataFrame()
    return combined, group_reports


def summarize_scorecard(scorecard, abs_ic_thresh=0.02, sharpe_thresh=1.0, abs_ic_col="Abs_IC", sig_col="is_significant", sharpe_col="Net_Sharpe", verbose=True):
    """
    基于 scorecard 统计关键指标达标因子的数量与占比

    统计三类：
      1. |IC| >= abs_ic_thresh（默认 0.02）
      2. 显著（is_significant 为 True）
      3. Sharpe > sharpe_thresh（默认 1.0）

    参数
    ----
    scorecard : DataFrame
        因子成绩单，需含 Abs_IC / is_significant / Net_Sharpe 列
    abs_ic_thresh : float
        |IC| 阈值，默认 0.02
    sharpe_thresh : float
        Sharpe 阈值，默认 1.0
    verbose : bool
        是否打印统计结果

    返回
    ----
    DataFrame  index=指标名，列：Count（数量）/ Total（总数）/ Ratio（占比）
    """
    total = len(scorecard)

    n_abs_ic = int((scorecard[abs_ic_col] > abs_ic_thresh).sum())
    n_sig = int(scorecard[sig_col].sum())
    n_sharpe = int((scorecard[sharpe_col] > sharpe_thresh).sum())

    rows = {
        f"|IC|>{abs_ic_thresh}": n_abs_ic,
        "显著": n_sig,
        f"Sharpe>{sharpe_thresh}": n_sharpe,
    }
    summary = pd.DataFrame(
        {
            "Count": rows,
            "Total": total,
        }
    )
    summary["Ratio"] = summary["Count"] / max(total, 1)

    if verbose:
        print(f"因子总数：{total}")
        for name, cnt in rows.items():
            print(f"  {name}: {cnt} 个（占比 {cnt / max(total, 1):.1%}）")

    return summary


def factor_ic_decay(daily_ic, freq="Y", decay_thresh=0.5, verbose=True):
    """
    因子时间衰减分析（对应笔记第七节：时间衰减）

    思路：把每日截面 Rank IC 按时间段（默认按年）分组，
    计算每个因子在每个时间段的 |IC| 均值；再用【最后一段 / 第一段】
    的比值衡量保留率，比值越低说明衰减越严重。

    参数
    ----
    daily_ic : DataFrame
        calc_daily_ic 的输出，行=日期(datetime index)，列=因子名，值=当日 Rank IC
    freq : str
        时间分段频率，'Y'=按年（默认），'Q'=按季，'6M'=按半年
    decay_thresh : float
        衰减判定阈值：保留率 < decay_thresh 视为“明显衰减”，默认 0.5
    verbose : bool
        是否打印汇总统计

    返回
    ----
    (decay_by_period, decay_summary)
        decay_by_period : DataFrame  index=因子，列=各时间段的 |IC| 均值 + Retention(保留率)
        decay_summary   : DataFrame  index=指标名，列 Count/Total/Ratio
    """
    ic = daily_ic.copy()
    if not isinstance(ic.index, pd.DatetimeIndex):
        ic.index = pd.to_datetime(ic.index)

    abs_ic = ic.abs()
    # 按时间段求每因子 |IC| 均值：行=时间段，列=因子
    period_mean = abs_ic.groupby(pd.Grouper(freq=freq)).mean()
    period_mean = period_mean.dropna(how="all")

    # 转成 行=因子、列=时间段
    decay_by_period = period_mean.T
    period_labels = [str(p.year) if freq == "Y" else str(p.date()) for p in period_mean.index]
    decay_by_period.columns = period_labels

    first_col, last_col = decay_by_period.columns[0], decay_by_period.columns[-1]
    retention = decay_by_period[last_col] / (decay_by_period[first_col] + 1e-8)
    decay_by_period["Retention"] = retention

    total = len(decay_by_period)
    n_decay = int((retention < decay_thresh).sum())
    n_stable = total - n_decay
    avg_first = decay_by_period[first_col].mean()
    avg_last = decay_by_period[last_col].mean()

    decay_summary = pd.DataFrame(
        {
            "Count": {
                f"明显衰减(保留率<{decay_thresh})": n_decay,
                "基本稳定": n_stable,
            },
            "Total": total,
        }
    )
    decay_summary["Ratio"] = decay_summary["Count"] / max(total, 1)

    if verbose:
        print(f"时间衰减分析（分段频率={freq}，共 {len(period_labels)} 段：{period_labels}）")
        print(f"  首段平均 |IC|：{avg_first:.4f}")
        print(f"  末段平均 |IC|：{avg_last:.4f}")
        print(f"  整体保留率：{avg_last / (avg_first + 1e-8):.1%}")
        print(f"  明显衰减因子：{n_decay} 个（占比 {n_decay / max(total, 1):.1%}）")
        print(f"  基本稳定因子：{n_stable} 个（占比 {n_stable / max(total, 1):.1%}）")

    return decay_by_period, decay_summary


def factor_correlation(df_neutralized, method="pearson"):
    """
    计算因子两两相关系数矩阵（截面对齐后整体相关）

    参数
    ----
    df_neutralized : DataFrame
        MultiIndex(datetime, instrument)，列=因子(残差)
    method : str
        'pearson'（默认）或 'spearman'

    返回
    ----
    DataFrame  因子 x 因子 的相关系数方阵
    """
    return df_neutralized.corr(method=method)


def reduce_redundant_factors(df_neutralized, ic_series=None, corr_thresh=0.8, method="pearson", verbose=True):
    """
    相关性去冗余（对应笔记第七节：内部相关性与冗余去除）

    做法：计算因子相关矩阵，将 |corr| > corr_thresh 的因子视为冗余簇；
    贪心保留——按 |IC| 从高到低排序，优先保留信息量大的因子，
    与已保留因子高度相关的后续因子被剔除。

    参数
    ----
    df_neutralized : DataFrame
        MultiIndex(datetime, instrument)，列=因子(残差)
    ic_series : Series or None
        各因子 Rank_IC（用于决定保留优先级）；为 None 时按 |IC| 无法排序，
        退化为按列顺序保留
    corr_thresh : float
        相关系数阈值，默认 0.8，超过则认为冗余
    method : str
        相关系数类型，'pearson'（默认）或 'spearman'
    verbose : bool
        是否打印统计

    返回
    ----
    (kept_factors, dropped_map, corr_matrix)
        kept_factors : list   保留下来的因子名
        dropped_map  : dict   {被删因子: 导致其被删的代表因子}
        corr_matrix  : DataFrame  相关系数方阵
    """
    corr = factor_correlation(df_neutralized, method=method)
    abs_corr = corr.abs()
    factors = list(corr.columns)

    # 保留优先级：|IC| 高者优先
    if ic_series is not None:
        order = sorted(factors, key=lambda f: abs(ic_series.get(f, 0.0)), reverse=True)
    else:
        order = factors

    kept = []
    dropped_map = {}
    for f in order:
        redundant_with = None
        for k in kept:
            if abs_corr.loc[f, k] > corr_thresh:
                redundant_with = k
                break
        if redundant_with is None:
            kept.append(f)
        else:
            dropped_map[f] = redundant_with

    # 保持原始列顺序输出保留因子
    kept_factors = [f for f in factors if f in kept]

    if verbose:
        print(f"相关性去冗余（阈值 |corr|>{corr_thresh}，method={method}）")
        print(f"  原始因子数：{len(factors)}")
        print(f"  保留因子数：{len(kept_factors)}")
        print(f"  剔除冗余数：{len(dropped_map)}")

    return kept_factors, dropped_map, corr


# ==============================================================
# 可视化工具（对应笔记中的 image.png / image-1.png / image-2.png）
# ==============================================================
def _setup_cn_font():
    """统一设置 matplotlib 中文字体与负号显示"""
    import matplotlib

    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False


def plot_ic_ranking(scorecard, top_n=20, ic_col="Rank_IC", figsize=(8, 8)):
    """
    画 |IC| 排名前 top_n 的因子横向柱状图（对应 image.png）

    参数
    ----
    scorecard : DataFrame
        报告1成绩单，index=因子名，需含 ic_col 列
    top_n : int
        展示前多少个因子，默认 20
    ic_col : str
        用于排序和着色的 IC 列名，默认 'Rank_IC'（用其绝对值排序）

    返回
    ----
    (fig, ax)
    """
    import matplotlib.pyplot as plt

    _setup_cn_font()

    s = scorecard[ic_col].dropna()
    top = s.reindex(s.abs().sort_values(ascending=False).index)[:top_n]
    # 柱状图从上到下由高到低，先反转让最大值在顶部
    top = top[::-1]
    colors = ["#d62728" if v >= 0 else "#1f77b4" for v in top.values]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(top.index, top.values, color=colors)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel(ic_col)
    ax.set_title(f"|{ic_col}| 排名前 {top_n} 的因子（红=正向 蓝=反向）")
    for y, v in enumerate(top.values):
        ax.text(v, y, f" {v:.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=9)
    fig.tight_layout()
    return fig, ax


def plot_freq_comparison(report3, metric="Net_Sharpe", sharpe_thresh=1.0, figsize=(8, 5)):
    """
    画调仓频率对比图（对应 image-1.png）

    展示每个调仓频率下：达标因子数（metric>=阈值）+ 该频率平均 metric

    参数
    ----
    report3 : DataFrame
        报告3结果，含 'Freq' 列 + metric 列（index=因子）
    metric : str
        对比指标，默认 'Net_Sharpe'
    sharpe_thresh : float
        达标阈值，默认 1.0

    返回
    ----
    (fig, ax)
    """
    import matplotlib.pyplot as plt

    _setup_cn_font()

    freq_order = ["D", "W", "M", "Q"]
    freqs = [f for f in freq_order if f in report3["Freq"].unique()]

    n_pass, avg_metric = [], []
    for f in freqs:
        sub = report3[report3["Freq"] == f][metric].dropna()
        n_pass.append(int((sub >= sharpe_thresh).sum()))
        avg_metric.append(sub.mean())

    fig, ax1 = plt.subplots(figsize=figsize)
    bars = ax1.bar(freqs, n_pass, color="#4c72b0", alpha=0.8)
    ax1.set_ylabel(f"{metric}>={sharpe_thresh} 因子数", color="#4c72b0")
    ax1.set_xlabel("调仓频率")
    for b, v in zip(bars, n_pass):
        ax1.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=10)

    ax2 = ax1.twinx()
    ax2.plot(freqs, avg_metric, "o-", color="#dd8452", linewidth=2)
    ax2.set_ylabel(f"平均 {metric}", color="#dd8452")
    ax1.set_title(f"调仓频率对比：达标因子数 & 平均 {metric}")
    fig.tight_layout()
    return fig, ax1


def plot_corr_heatmap(corr_matrix, figsize=(10, 9), cmap="coolwarm"):
    """
    画因子相关性热力图（对应 image-2.png）

    参数
    ----
    corr_matrix : DataFrame
        reduce_redundant_factors 返回的相关方阵
    figsize : tuple
    cmap : str

    返回
    ----
    (fig, ax)
    """
    import matplotlib.pyplot as plt

    _setup_cn_font()

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr_matrix.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    cols = list(corr_matrix.columns)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    ax.set_title("因子相关性热力图")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig, ax
