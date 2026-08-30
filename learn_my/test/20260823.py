# %%
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

import qlib
from qlib.contrib.data.handler import Alpha158
from qlib.data import D

qlib.init(
    provider_uri='~/.qlib/qlib_data/my_tushare_data', 
    region=qlib.config.REG_CN
)

START_TIME = "2010-01-01"
END_TIME = "2026-01-01"


# %%
# 获取所有可用的股票代码列表
# instruments = D.instruments()
# print(instruments)

instruments_path = os.path.expanduser('~/.qlib/qlib_data/my_tushare_data/instruments/all.txt')
def get_instruments_from_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            instruments = [line.strip().split()[0] for line in f.readlines()]
        return instruments
    else:
        raise FileNotFoundError(f"未找到股票代码文件: {file_path}")
instruments = get_instruments_from_file(instruments_path)
# instruments = instruments[:10]
# instruments = ["SZ000001","SZ000002","SZ000003","SZ000004","SZ000005","SZ000006","SZ000007","SZ000008","SZ000009","SZ000010"]

# print(instruments[:10])
print(len(instruments))

# %%
print("正在加载 Alpha158 因子数据... (数据量较大，可能需要1-2分钟)")
# 2. 提取 158 个因子数据
handler = Alpha158(
    instruments=instruments,
    start_time=START_TIME,
    end_time=END_TIME,
    # 填补缺失值，防止后续算相关系数报错
    infer_processors=[], 
    learn_processors=[]
)

# fetch() 默认会返回所有股票的因子特征 DataFrame
df_features = handler.fetch()

# %%
print(df_features.shape)
# print(df_features.head(10))
print(df_features.index.names)
print ("df_features.columns = ", len(df_features.columns))

all_nan_cols = df_features.columns[df_features.isna().all()].tolist()
print('all_nan_cols =', all_nan_cols)

if 'LABEL0' in df_features.columns:
    print("Dropping 'LABEL0' column from features...")
    df_features = df_features.drop('LABEL0', axis=1)

if all_nan_cols:
    print("Columns with all NaN values found.")
    print("Dropping these columns from features...")
    df_features = df_features.drop(all_nan_cols, axis=1)

print("Final number of features:", len(df_features.columns))

# df_features = df_features.dropna() # 这里就不做处理了吧
print(df_features.shape)

# %%
# ==============================================================================
# 2. 【核心新增模块】：准备真实的行业与市值数据 (df_style)
# ==============================================================================
print("2. 正在加载真实行业与市值风格数据...")

# 2.1 读取股票基础信息，拿到每只股票的行业标签
stock_basic_path = r"C:\Users\WANGKANG\Desktop\量化交易学习\stock_data_old\stock_basic.csv"
if os.path.exists(stock_basic_path):
    stock_basic = pd.read_csv(stock_basic_path, dtype={'ts_code': str})
    stock_basic = stock_basic[['ts_code', 'industry']].dropna().copy()
    stock_basic['industry'] = stock_basic['industry'].astype(str).str.strip()
    stock_basic['industry'] = stock_basic['industry'].replace({'': 'OTHER'})
    stock_basic = stock_basic.drop_duplicates(subset=['ts_code'])
    stock_basic['instrument'] = stock_basic['ts_code'].map(
        # lambda x: x.split('.')[1].lower() + x.split('.')[0]
        lambda x: x.split('.')[1] + x.split('.')[0]
    )
    industry_map = stock_basic.set_index('instrument')['industry']
    industry_dummy = pd.get_dummies(industry_map, prefix='Ind')
    industry_dummy = industry_dummy.reindex(sorted(industry_dummy.columns), axis=1)
else:
    raise FileNotFoundError(f"未找到行业标签文件: {stock_basic_path}")

# %%
print(industry_dummy.head(1))
print(industry_dummy.shape)

print("-------")
rows_with_nan = industry_dummy[industry_dummy.isna().any(axis=1)]
print(rows_with_nan)

# %%
# 2.2 读取每日基本面数据中的流通市值（circ_mv）
# 文件结构示例：000001.SZ.csv -> ts_code, trade_date, circ_mv, ...
daily_basic_dir = r"C:\Users\WANGKANG\Desktop\量化交易学习\stock_data_old\daily_basic"
style_parts = []

for file_name in sorted(os.listdir(daily_basic_dir)):
    if not file_name.endswith('.csv'):
        continue
    file_path = os.path.join(daily_basic_dir, file_name)
    try:
        df = pd.read_csv(file_path)
    except Exception:
        continue

    needed_cols = ['ts_code', 'trade_date', 'circ_mv']
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        continue

    df = df[needed_cols].copy()
    df = df.dropna(subset=['ts_code', 'trade_date', 'circ_mv'])
    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d').dt.strftime('%Y-%m-%d')
    # df['instrument'] = df['ts_code'].map(lambda x: x.split('.')[1].lower() + x.split('.')[0])
    df['instrument'] = df['ts_code'].map(lambda x: x.split('.')[1] + x.split('.')[0])
    df['log_mv'] = np.log(df['circ_mv'].replace(0, np.nan) + 1) # 避免 log(0) 的情况
    df = df[['instrument', 'trade_date', 'log_mv']]
    style_parts.append(df)

if not style_parts:
    raise FileNotFoundError(f"未找到任何 daily_basic 数据文件: {daily_basic_dir}")

style_df = pd.concat(style_parts, ignore_index=True)

style_df = style_df.merge(
    industry_dummy.reset_index().rename(columns={'index': 'instrument'}),
    on='instrument',
    how='left'
)
style_df = style_df.fillna(0)
style_df['datetime'] = pd.to_datetime(style_df['trade_date'])

# %%
print(style_df.head(1)) 
print(style_df.shape)

print("-------")
rows_with_nan = style_df[style_df.isna().any(axis=1)]
print(rows_with_nan)

# %%
# 2.3 构造与 df_features 索引一致的 MultiIndex
# 目标索引格式： (datetime，instrument)
style_df_ = style_df[['datetime', 'instrument'] + [c for c in style_df.columns if c.startswith('Ind_') or c == 'log_mv']]
style_df_ = style_df_.set_index(['datetime', 'instrument']).sort_index()

# 对齐到 df_features 的 index
common_index = df_features.index.intersection(style_df_.index)
print(f"对齐后的样本数: {len(common_index)}")
df_style = style_df_.loc[common_index].copy()
print(f"真实市值/行业数据已生成，样本数: {df_style.shape[0]}, 列数: {df_style.shape[1]}")
print(df_style.head())
# ----------------------------------------------------

# %%
print(df_style.head(10))
print(df_style.shape)

# %%
print("df_features index names:", df_features.index.names)
print("style_df index names:", style_df_.index.names)

print("df_features index head:", df_features.index[:3])
print("style_df index head:", style_df_.index[:3])

print("交集长度:", len(df_features.index.intersection(style_df_.index)))

# %%
# ==============================================================================
# 3. 【核心新增模块】：执行每日横截面 行业与市值中性化
# ==============================================================================
print("3. 正在执行 行业与市值中性化 (截面回归提取残差)...")

# 定义自变量 (X) 与 因变量 (Y)
style_cols = df_style.columns.tolist()       # log_mv + 行业0/1列
factor_cols = df_features.columns.tolist()   # 158 个因子列

# 合并因子与风格特征
df_for_neutralize = df_features.join(df_style, how='inner')

print("df_for_neutralize shape:", df_for_neutralize.shape)
print(df_for_neutralize.head(1))
print(df_for_neutralize.isna().all().sum())

nan_cols = df_for_neutralize.columns[df_for_neutralize.isna().any()].tolist()
print(nan_cols)

# %%
# def mad_zscore_df(df_cols, n=3):
#     """
#     针对多列 DataFrame 的截面批量预处理:
#     MAD 3倍中位数去极值 + Z-Score 标准化 (矩阵向量化,含空值与常数列保护)
#     """
#     if df_cols.empty:
#         return df_cols
    
#     # 1. 计算截面中位数与 MAD
#     med = df_cols.median(axis=0)
#     mad = (df_cols.sub(med, axis=1)).abs().median(axis=0)
    
#     # 2. 计算上下界 (1.4826 是正态分布转换系数)
#     upper = med + n * 1.4826 * mad
#     lower = med - n * 1.4826 * mad
    
#     # 3. 截断极值 (Pandas 支持传入 Series 沿列方向截断)
#     clipped = df_cols.clip(lower=lower, upper=upper, axis=1)
    
#     # 4. Z-Score 标准化
#     mean = clipped.mean(axis=0)
#     std = clipped.std(axis=0)
    
#     # 保护: 若某因子全为常数 (std == 0 或 NaN), 加 1e-8 防止除以 0
#     zscored = (clipped.sub(mean, axis=1)).div(std + 1e-8, axis=1)
    
#     # 5. 填补残余 NaN 为 0 (中性水平)
#     return zscored.fillna(0.0)

# %%
# # 区分连续风格变量(市值)与离散哑变量(行业)
# mv_cols = ['log_mv']
# ind_cols = [c for c in style_cols if c != 'log_mv']

# def neutralize_daily(daily_df):
#     """
#     每日横截面中性化函数:
#     1. 对 158 个因变量 (Alpha因矩阵) 进行 MAD 去极值 + Z-score
#     2. 对市值自变量 (log_mv) 进行 MAD 去极值 + Z-score
#     3. 行业哑变量 (Ind_xxx) 保持 0/1 结构不变
#     4. 截面 OLS 多元线性回归提取纯净残差 (Alpha)
#     """
#     # 剔除风格信息不全的股票
#     valid_mask = daily_df[style_cols].notna().all(axis=1)
#     valid_df = daily_df[valid_mask].copy()
    
#     # 若当天有效样本过少，放弃回归
#     if len(valid_df) < 50:
#         return pd.DataFrame(index=daily_df.index, columns=factor_cols)
    
#     # -----------------------------------------------------------
#     # 【核心改进 1】：对 158 个因子 Y 进行批量去极值与标准化
#     # -----------------------------------------------------------
#     Y_raw = valid_df[factor_cols]
#     Y_clean = mad_zscore_df(Y_raw)
    
#     # -----------------------------------------------------------
#     # 【核心改进 2】：处理自变量 X
#     # 市值 (log_mv) 是连续变量 -> 做 MAD + Z-score
#     # 行业 (Ind_xxx) 是 0/1 哑变量 -> 必须保持原样，保留分类基准
#     # -----------------------------------------------------------
#     mv_clean = mad_zscore_df(valid_df[mv_cols])
#     ind_clean = valid_df[ind_cols]
    
#     X_clean = pd.concat([mv_clean, ind_clean], axis=1)
    
#     # -----------------------------------------------------------
#     # 3. 极速回归与残差提取
#     # -----------------------------------------------------------
#     X = X_clean.values
#     Y = Y_clean.values
    
#     model = LinearRegression(fit_intercept=True)
#     model.fit(X, Y)
#     residuals = Y - model.predict(X)
    
#     return pd.DataFrame(residuals, index=valid_df.index, columns=factor_cols)

# %%
def neutralize_daily(daily_df):
    """
    每日横截面中性化函数
    Y (因子) = beta * X (市值 + 行业) + 残差 (纯净Alpha)
    """
    # 剔除风格列有缺失值的股票
    valid_mask = ~daily_df[style_cols].isna().any(axis=1) # valid_mask = daily_df[style_cols].notna().all(axis=1)
    valid_df = daily_df[valid_mask]
    
    # 若当天有效股票过少，不具有统计回归意义
    if len(valid_df) < 50:
        return pd.DataFrame(index=daily_df.index, columns=factor_cols)
    
    X = valid_df[style_cols].values
    # 因子若有微量缺失值，用截面均值填补，防止矩阵报错
    # Y = valid_df[factor_cols].fillna(valid_df[factor_cols].mean()).values # 可能存在一整列都是0的情况

    # 因子若有缺失值，先用截面均值填补，若整列为NaN则填补为0
    Y = valid_df[factor_cols].fillna(valid_df[factor_cols].mean()).fillna(0).values
    
    # 极速多输出线性回归
    model = LinearRegression(fit_intercept=True)
    model.fit(X, Y)
    
    # 核心：计算残差 (Residuals)
    residuals = Y - model.predict(X)
    
    return pd.DataFrame(residuals, index=valid_df.index, columns=factor_cols)

# 使用 tqdm 显示每日横截面处理进度
tqdm.pandas(desc="截面中性化进度")
df_neutralized = df_for_neutralize.groupby(level='datetime', group_keys=False).progress_apply(neutralize_daily)

# 将原因子矩阵替换为中性化后的残差矩阵
df_features = df_neutralized
print("中性化提纯完成！")

# %%
print(df_features.head(10))

# %%
print("正在计算未来5日收益率标签 (Label)...")
# 3. 计算未来 5 日的真实收益率作为 Label (T+1 到 T+5 的收益)
# 逻辑：T日收盘决策 → T+1日开盘买入 → T+5日收盘卖出
# 公式：(T+5收盘价 / T+1开盘价) - 1

df_label = D.features(
    instruments=instruments,
    fields=["Ref($close, -5) / Ref($open, -1) - 1 - 0.0026"],  # 减去交易成本 0.13%
    start_time=START_TIME, 
    end_time=END_TIME
)
df_label.columns = ["label"]

# %%
print(df_label.columns)
print(df_label.tail(1))
print(df_label.shape)

# %%
print("正在执行横截面 Rank IC 计算... (每日计算)")
# 5. 定义每日计算 Rank IC 的函数
# 使用 spearman 秩相关系数，这正是 Rank IC 的本质
def calc_cross_sectional_ic(daily_df):
    # 只取因子列和 label 列做相关性计算，返回每个因子当天的 IC
    return daily_df.drop(columns=['label']).corrwith(daily_df['label'], method='spearman')

# 按日期 (datetime) 分组，每天算一次截面 IC
# 这一步 Pandas 会进行向量化运算，可能需要几十秒
# 先把中性化后的因子和 label 对齐
df_ic = df_features.join(df_label).dropna(subset=['label'])
daily_ic = df_ic.groupby(level='datetime').apply(calc_cross_sectional_ic)

# %%
print(daily_ic.shape)
print(daily_ic.head(5))

# %%
print("正在汇总 IC 与 ICIR 成绩单...")
# 6. 汇总计算最终的 因子表现
# IC 取均值，ICIR = 均值 / 标准差 (代表 IC 的稳定性，年化时可乘 sqrt(252))
factor_ic_mean = daily_ic.mean()
factor_ic_std = daily_ic.std()
factor_icir = factor_ic_mean / factor_ic_std

# 7. 组装成最终的 DataFrame 成绩单
performance_df = pd.DataFrame({
    'Rank_IC': factor_ic_mean,
    'ICIR': factor_icir,
    'Abs_IC': factor_ic_mean.abs() # 增加绝对值列，方便排序寻找强信号因子
})

# 按绝对 IC 降序排列，找出最牛的因子
performance_df = performance_df.sort_values(by='Abs_IC', ascending=False)

print("\n🎉 计算完成！这是前 20 个预测能力最强的因子：")
print("-" * 60)
print(performance_df.head(20).to_string(formatters={'Rank_IC': '{:.4f}'.format, 'ICIR': '{:.4f}'.format, 'Abs_IC': '{:.4f}'.format}))

# 建议将全量成绩单导出为 CSV 分析
performance_df.to_csv("alpha158_ic_icir_report.csv")
print("-" * 60)
print("完整报告已保存至 alpha158_ic_icir_report.csv")





# %%
# abs(ic) >= 0.02 的因子
mask = performance_df['Abs_IC'] >= 0.02
selected_factors = performance_df[mask]
print(f"\n🎯 预测能力较强的因子 (Abs_IC >= 0.02)有{len(selected_factors)}个")

# %%
# ==============================================================================
# 8. 【核心新增模块】：因子 t 统计量、p 值计算与多重假设检验校正
# ==============================================================================
print("\n8. 正在执行因子的假设检验与多重校正 (FDR / Benjamini-Hochberg)...")

# 1. 计算每个因子的基本统计量
T_days = daily_ic.count()            # 每个因子的有效交易天数
ic_mean = daily_ic.mean()            # 截面 IC 均值
ic_std = daily_ic.std()              # 截面 IC 标准差
icir = ic_mean / (ic_std + 1e-8)     # ICIR

# 2. 计算每个因子的 t 统计量 (t = ICIR * sqrt(T)) 与双尾原始 p 值
t_stats = icir * np.sqrt(T_days)
raw_p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=T_days - 1))
raw_p_series = pd.Series(raw_p_values, index=daily_ic.columns)

# 3. 多重检验校正 (FDR: Benjamini-Hochberg 法，显著性阈值 alpha=0.05)
# method='fdr_bh' 是业界与学术界检验海量 Alpha 因子的标准方法
# 如果想采用极严格的族系误差控制，也可切换为 method='holm' 或 method='bonferroni'
reject, p_adjusted, alphacSidak, alphacBonf = multipletests(
    pvals=raw_p_series.values,
    alpha=0.05,
    method='fdr_bh'
)

# 4. 整理结果并合并到 performance_df 中
stat_df = pd.DataFrame({
    't_stat': t_stats,
    'raw_p_value': raw_p_series,
    'adj_p_value': p_adjusted,
    'is_significant': reject  # 是否在多重校正后依然显著 (True/False)
})

performance_df = performance_df.join(stat_df)
performance_df = performance_df.sort_values(by='Abs_IC', ascending=False)

# 5. 输出统计报告
num_total = 158
num_raw_sig = (performance_df['raw_p_value'] < 0.05).sum()
num_adj_sig = performance_df['is_significant'].sum()

print("-" * 65)
print(f"📊 因子总数: {num_total} 个")
print(f"🔹 未经校正显著 (Raw p < 0.05) 的因子: {num_raw_sig} 个 ({num_raw_sig / num_total:.1%})")
print(f"🎯 经过多重校正显著 (FDR < 0.05) 的因子: {num_adj_sig} 个 ({num_adj_sig / num_total:.1%})")
print("-" * 65)

# 展示显著性排名前 10 的因子
display_cols = ['Rank_IC', 'ICIR', 't_stat', 'raw_p_value', 'adj_p_value', 'is_significant']
print("\n显著性排名前 10 的因子详情：")
print(performance_df[display_cols].head(10).to_string())

# %%
# ==============================================================================
# 9.2 单因子月频/周频五分位多空组合回测
# ==============================================================================
mode = "monthly"  # 可选 "monthly" 或 "weekly"
freq_name = "月频" if mode == "monthly" else "周频"
print(f"\n正在执行单因子【{freq_name}】五分位多空组合回测 ({freq_name}持仓, 扣除 13bps 成本)...")

# 1. 提取未来 1 个月(20日) 或 1 周(5日) 的前瞻收益率标签
forward_days = -20 if mode == "monthly" else -5
df_label_periodic = D.features(
    instruments=instruments,
    fields=[f"Ref($close, {forward_days}) / Ref($open, -1) - 1"], 
    start_time=START_TIME, 
    end_time=END_TIME
)
df_label_periodic.columns = [f"label_{mode}"]

print("df_label_periodic.index.names = ", df_label_periodic.index.names)

# 2. 筛选出每个周期的【最后一个交易日】
def get_rebalance_dates(trade_dates, freq="monthly"):
    """
    trade_dates: 交易日 DatetimeIndex 或 Series
    freq: "monthly" 或 "weekly"
    return: 每个周期的最后一个交易日
    """
    trade_dates = pd.to_datetime(trade_dates.unique()).sort_values()

    if freq == "monthly":
        # 按月分组，取每月最后一个交易日
        dates_df = pd.DataFrame({
            "datetime": trade_dates,
            "period": trade_dates.to_period("M")
        })
        return dates_df.groupby("period")["datetime"].last().values

    elif freq == "weekly":
        # 按周分组，取每周最后一个交易日
        # 常见以周五收尾；如果你们的交易习惯是周一/周五，可以改成 W-MON / W-FRI
        dates_df = pd.DataFrame({
            "datetime": trade_dates,
            "period": trade_dates.to_period("W-FRI")
        })
        return dates_df.groupby("period")["datetime"].last().values

    else:
        raise ValueError(f"Unsupported freq: {freq}")

all_dates = df_features.index.get_level_values('datetime').unique()
rebalance_dates = get_rebalance_dates(all_dates, freq=mode)
print(f"📅 样本期内共有 {len(rebalance_dates)} 个【{freq_name}】调仓换仓日")

# 3. 仅保留调仓日截面的因子与收益数据
date_mask = df_features.index.get_level_values('datetime').isin(rebalance_dates)
df_eval_periodic = df_features.loc[date_mask].copy()
df_eval_periodic = df_eval_periodic.join(df_label_periodic).dropna(subset=[f"label_{mode}"])

# 单边交易成本 (13 bps)
COST_RATE = 0.0013
# 年化调仓期数: 月频 12 次，周频 52 次
PERIODS_PER_YEAR = 12 if mode == "monthly" else 52

def fast_backtest_single_factor(factor_name, ic_val):
    direction = 1 if ic_val >= 0 else -1
    sub_df = df_eval_periodic[[factor_name, f"label_{mode}"]].dropna()
    
    # 1. 截面计算 Q5 均值与 Q1 均值
    def calc_ls(daily):
        if len(daily) < 10:
            return np.nan
        daily['group'] = pd.qcut(daily[factor_name].rank(method='first'), q=5, labels=False)
        long_id = 4 if direction == 1 else 0
        short_id = 0 if direction == 1 else 4
        
        # 多头收益 - 空头收益 (毛收益)
        return daily.loc[daily['group'] == long_id, f"label_{mode}"].mean() - \
               daily.loc[daily['group'] == short_id, f"label_{mode}"].mean()

    # 2. 得到多空毛收益序列
    ls_gross_series = sub_df.groupby(level='datetime').apply(calc_ls).dropna()
    
    if len(ls_gross_series) < 6:
        return np.nan, np.nan, np.nan
    
    # 3. 直接每期减去固定成本得到净收益
    ls_net_series = ls_gross_series - COST_RATE * 2
    
    # 4. 计算年化指标
    ann_net_ret = ls_net_series.mean() * PERIODS_PER_YEAR
    ann_net_vol = ls_net_series.std() * np.sqrt(PERIODS_PER_YEAR)
    net_sharpe = ann_net_ret / (ann_net_vol + 1e-8)
    
    return ann_net_ret, ann_net_vol, net_sharpe

def backtest_single_factor(factor_name, ic_val):
    """
    单个因子的多空组合回测逻辑
    """
    direction = 1 if ic_val >= 0 else -1
    
    # 提取当前因子列与收益列
    sub_df = df_eval_periodic[[factor_name, f"label_{mode}"]].dropna()
    
    # 按日期分组，截面切分为 5 等分 (Q1 ~ Q5)
    def calc_ls_return(daily):
        if len(daily) < 10:
            return pd.Series({'ls_gross_ret': np.nan, 'long_set': set(), 'short_set': set()})
        
        # 截面排名分为 5 组 (0 到 4)
        daily['group'] = pd.qcut(daily[factor_name].rank(method='first'), q=5, labels=False)
        
        # 根据 IC 正负决定多头组和空头组
        long_group_id = 4 if direction == 1 else 0
        short_group_id = 0 if direction == 1 else 4
        
        long_stocks = daily[daily['group'] == long_group_id]
        short_stocks = daily[daily['group'] == short_group_id]
        
        long_ret = long_stocks[f"label_{mode}"].mean()
        short_ret = short_stocks[f"label_{mode}"].mean()
        
        # 多空组合毛收益 (多头收益 - 空头收益)
        ls_gross = long_ret - short_ret
        
        return pd.Series({
            'ls_gross_ret': ls_gross,
            'long_set': set(long_stocks.index),
            'short_set': set(short_stocks.index)
        })
    
    # 按调仓日计算截面收益
    period_df = sub_df.groupby(level='datetime', group_keys=False).apply(calc_ls_return).dropna(subset=['ls_gross_ret'])
    
    if len(period_df) < 6:
        return np.nan, np.nan, np.nan
    
    # 计算换手率与交易成本
    long_sets = period_df['long_set'].values
    short_sets = period_df['short_set'].values
    
    turnovers = []
    for t in range(1, len(period_df)):
        # 多头单边换手: 1 - 交集/上一期持仓数
        prev_l, curr_l = long_sets[t-1], long_sets[t]
        prev_s, curr_s = short_sets[t-1], short_sets[t]
        
        to_long = 1.0 - len(prev_l & curr_l) / max(len(prev_l), 1)
        to_short = 1.0 - len(prev_s & curr_s) / max(len(prev_s), 1)
        
        turnovers.append(to_long + to_short)
    
    # 首期假定换手率为 2 (建仓双边)
    turnover_series = pd.Series([2.0] + turnovers, index=period_df.index)
    
    # 扣除换手成本: 净收益 = 毛收益 - 换手率 * 13bps
    period_df['cost'] = turnover_series * COST_RATE
    period_df['ls_net_ret'] = period_df['ls_gross_ret'] - period_df['cost']
    
    # 计算年化指标
    mean_net = period_df['ls_net_ret'].mean()
    std_net = period_df['ls_net_ret'].std()
    
    if std_net == 0 or np.isnan(std_net):
        return np.nan, np.nan, np.nan
    
    ann_net_ret = mean_net * PERIODS_PER_YEAR
    ann_net_vol = std_net * np.sqrt(PERIODS_PER_YEAR)
    net_sharpe = ann_net_ret / (ann_net_vol + 1e-8)
    
    return ann_net_ret, ann_net_vol, net_sharpe

# 遍历所有因子计算夏普比率
sharpe_results = {}
prefix = f"{mode.capitalize()}"
print(f"正在逐个回测 158 个因子 ({freq_name})...")
for factor in tqdm(performance_df.index):
    ic_val = performance_df.loc[factor, 'Rank_IC']
    # ann_ret, ann_vol, sharpe = backtest_single_factor(factor, ic_val)
    ann_ret, ann_vol, sharpe = fast_backtest_single_factor(factor, ic_val)
    sharpe_results[factor] = {
        f'{prefix}_Net_Return': ann_ret,
        f'{prefix}_Net_Vol': ann_vol,
        f'{prefix}_Net_Sharpe': sharpe
    }

# 合并到主成绩单 DataFrame
sharpe_df = pd.DataFrame.from_dict(sharpe_results, orient='index')
performance_df = performance_df.drop(columns=[c for c in sharpe_df.columns if c in performance_df.columns])
performance_df = performance_df.join(sharpe_df)

# ==============================================================================
# 汇总与统计分析
# ==============================================================================
# 按净夏普比率降序排列
target_sharpe_col = f'{prefix}_Net_Sharpe'
performance_df = performance_df.sort_values(by=target_sharpe_col, ascending=False)

num_sharpe_gt_1 = (performance_df[target_sharpe_col] > 1.0).sum()
total_factors = len(performance_df)

print("\n" + "=" * 70)
print(f"📊 【{freq_name}调仓】多空净夏普统计结果 (扣除 13bps 换手成本):")
print(f"🎯 {target_sharpe_col} > 1.0 的因子数量: {num_sharpe_gt_1} / {total_factors} ({num_sharpe_gt_1 / total_factors:.1%})")
print("=" * 70)

display_cols = ['Rank_IC', 'ICIR', f'{prefix}_Net_Return', target_sharpe_col]
print(f"\n🏆 【{freq_name}】净夏普排名前 15 的优秀因子：")
print(performance_df[display_cols].head(15).to_string(formatters={
    'Rank_IC': '{:.4f}'.format,
    'ICIR': '{:.4f}'.format,
    f'{prefix}_Net_Return': '{:.2%}'.format,
    target_sharpe_col: '{:.2f}'.format
}))

# 导出最终的综合评估报告
performance_df.to_csv("alpha158_full_evaluation.csv")
print("\n📁 完整全因子评价报告已保存至 alpha158_full_evaluation.csv")



