import json
import logging
import os
import sys
import time
import traceback

# import learn_my.common_utils as common_utils
import common_utils as common_utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

import qlib
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.report import analysis_model
from qlib.data import D
from qlib.data.dataset.handler import DataHandler, DataHandlerLP
from qlib.data.dataset.processor import (CSNeutralize, CSZScoreNorm, Fillna,
                                         Processor, RobustZScoreNorm,
                                         ZScoreNorm)
from qlib.data.filter import ExpressionDFilter, NameDFilter
from qlib.contrib.report import analysis_position

# 设置画图风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 1. 【核心】禁止自动换行折叠
pd.set_option('display.expand_frame_repr', False)

# 2. 设置终端最大显示宽度（设为一个很大的数字或 1000）
pd.set_option('display.width', 1000)

# 3. 显示所有列（防止列太多中间被省略成 ...）
pd.set_option('display.max_columns', None)

qlib.init(
    provider_uri="~/.qlib/qlib_data/my_tushare_data",
    region=qlib.config.REG_CN,
    # parallel=dict(
    #     enabled=True,
    #     num_workers=8,  # 根据CPU核心数调整
    #     mp_start_method="spawn"
    # )
)

# START_TIME = "2010-01-01"
START_TIME_PRE = "2024-06-01"  # 暖机期，多取半年数据，确保因子计算的时候有值
START_TIME = "2025-01-01"  # 实际使用开始时间
END_TIME = "2026-08-01"
END_TIME_POST = "2026-08-28"  # 计算收益的时候多向后计算一段时间，避免收益为空

WORK_DIR = "./learn_my"
IMG_DIR = './learn_my/imgs'

INSTRUMENTS_NAME = "hs300"


def get_instruments(instruments_name: str):
    """可供选择的股票池"""
    market_name_dict = {
        "all": "all",  # 主板(沪深A股)
        ############################### 板块 ###############################
        "main": "zb",  # 主板(沪深A股)
        "cyb": "cyb",  # 创业板
        "kcb": "kcb",  # 科创板
        "bjs": "bjs",  # 北交所
        ############################### 指数 ###############################
        "sse50": "sse50",  # 上证50
        "hs300": "csi300",  # 沪深300
        "csi500": "csi500",  # 中证500
        "zz800": "csi800",  # 中证800
        "zz1000": "csi1000",  # 中证1000
        "star50": "star50",  # 科创50
    }

    if instruments_name in market_name_dict.keys():
        return D.instruments(market=market_name_dict[instruments_name])
    else:
        print("instruments_name不存在！")
        return 0


# 此方法不再需要，直接使用qlib的DataHandlerLP
# def get_data(instruments):
#     """获取原始数据"""
#     fields = ["$open", "$high", "$low", "$close", "$volume", "$amount", "$vwap", "$circ_mv", "$total_mv", "$adj_factor", "$industry", "$is_ST", "$list_status"]  # 基础 K 线字段 (记得带 $)

#     # 使用 D.features 获取数据
#     df = D.features(instruments=instruments, fields=fields, start_time=START_TIME, end_time=END_TIME)  # start 和 end 设为同一天，就是查这一天的数据
#     # 设置索引并排序
#     df = df.reset_index().set_index(["datetime", "instrument"]).sort_index()
#     return df

def get_features_labels_raws(instruments):
    """计算因子"""
    # 提取 158 个因子数据
    # handler = Alpha158(
    #     instruments=instruments,
    #     start_time=START_TIME,
    #     end_time=END_TIME,
    #     # 填补缺失值，防止后续算相关系数报错
    #     infer_processors=[],
    #     learn_processors=[]
    # )

    # 自定义因子（表达式 → 名字）
    feature_fields = [
        "($close-$open)/$open",  # KMID
        "($high-$low)/$open",  # KLEN
        "($high-Greater($open,$close))/$open",  # KUP
        "Ref($close,5)/$close",  # ROC5
        "Mean($close,5)/$close",  # MA5
        "Std($close,5)/$close",  # STD5
        "Max($high,5)/$close",  # MAX5
        "Min($low,5)/$close",  # MIN5
        "Mean($volume,5)/($volume+1e-12)",  # VMA5
        "$close/Ref($close,1)-1",  # RET1
    ]
    feature_names = ["KMID", "KLEN", "KUP", "ROC5", "MA5", "STD5", "MAX5", "MIN5", "VMA5", "RET1"]

    label_fields = [
        "Ref($open,-2)/Ref($open,-1)-1",  # T+1 买入 T+2 卖出，1日
        "Ref($open,-6)/Ref($open,-1)-1",  # 1周(5日)
        "Ref($open,-21)/Ref($open,-1)-1",  # 1月(20日)
        "Ref($open,-61)/Ref($open,-1)-1",  # 1季(60日)
    ]
    label_names = ["RET_1D", "RET_1W", "RET_1M", "RET_1Q"]

    raw_fields = ["$open", "$high", "$low", "$close", "$volume", "$amount", "$vwap", "$circ_mv", "$total_mv", "$factor", "$industry", "$is_ST", "$list_status"]

    handler = DataHandlerLP(
        instruments=instruments,
        start_time=START_TIME_PRE,  # 将时间向前和向后推进一些，这样可以让因子计算更加充分，减少NAN
        end_time=END_TIME_POST,
        # ======== 【树模型 / IC 分析配置】 =========
        # infer：把学到的参数应用到所有时间段数据
        infer_processors=[
            # {"class": "StockFilterProcessor"},
            # 预测阶段：只对特征(feature)填充缺失值，通常用0填充（因为你主要算比率）
            {"class": "CSZScoreNorm", "kwargs": {"fields_group": "feature", "method": "robust"}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature", "fill_value": 0}},
        ],
        learn_processors=[
            # {"class": "StockFilterProcessor"},
            # 训练阶段：必须剔除标签(label)为空的行，否则模型无法训练
            {"class": "DropnaLabel"},
            # 训练阶段：对特征(feature)填充缺失值
            {"class": "CSZScoreNorm", "kwargs": {"fields_group": "feature", "method": "robust"}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature", "fill_value": 0}},
        ],
        # # ======== 【深度学习 / 线性模型配置】 =========
        # infer_processors=[
        #     # 1. 截面去极值 + 截面Z-Score标准化 (针对特征)
        #     {"class": "CSZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
        #     # 2. 缺失值填充 (标准化后，0代表当天的截面平均水平，用0填充最合理)
        #     {"class": "Fillna", "kwargs": {"fields_group": "feature", "fill_value": 0}},
        # ],
        # learn_processors=[
        #     # 1. 剔除标签为空的行 (仅训练期)
        #     {"class": "DropnaLabel"},
        #     # 2. 截面去极值 + 截面Z-Score标准化 (针对特征)
        #     {"class": "CSZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
        #     # 3. 缺失值填充
        #     {"class": "Fillna", "kwargs": {"fields_group": "feature", "fill_value": 0}},
        # ],
        data_loader={
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": (feature_fields, feature_names),
                    # 保留标签
                    "label": (label_fields, label_names),
                    "raw": raw_fields,
                },
            },
        },
    )
    df_features = handler.fetch(col_set="feature")
    df_labels = handler.fetch(col_set="label")
    df_raws = handler.fetch(col_set="raw")

    # 过滤指定时间段内的数据
    dt = df_features.index.get_level_values("datetime")
    df_features = df_features[(dt >= START_TIME) & (dt <= END_TIME)]
    df_labels = df_labels[(dt >= START_TIME) & (dt <= END_TIME)]
    df_raws = df_raws[(dt >= START_TIME) & (dt <= END_TIME)]

    return df_features, df_labels, df_raws

def postProcess(df_raws, df_features):
    """行业和市值中性化"""
    df_style = common_utils.build_style_factors(df_raws)

    factor_cols = df_features.columns.tolist()  # 10 个因子：KMID, KLEN...
    style_cols = df_style.columns.tolist()  # log_mv + Ind_* 行业哑变量

    df_for_neutralize = df_features.join(df_style, how="inner")

    csNeutralize = CSNeutralize(factor_cols, style_cols)
    return csNeutralize(df_for_neutralize)



def run_full_factor_analysis(df_features, df_labels, target_label="RET_1D", n_groups=5):
    """
    因子全面测试主函数
    :param df_features: 中性化后的特征 DataFrame (Index: [datetime, instrument])
    :param df_labels: 收益率标签 DataFrame
    :param target_label: 主要分析的目标收益率（如 RET_1D 或 RET_1W）
    :param n_groups: 分层回测的组数（默认 5 组）
    """
    print(f"\n{'='*25} 开始全方位因子测试 (基准标签: {target_label}) {'='*25}")
    
    # 确保索引对齐
    common_idx = df_features.index.intersection(df_labels.index)
    feats = df_features.loc[common_idx]
    labs = df_labels.loc[common_idx]
    
    factor_names = feats.columns.tolist()
    label_series = labs[target_label]

    # =========================================================================
    # 维度 1 & 3：IC / Rank IC 统计与跨期衰减分析
    # =========================================================================
    summary_records = []
    daily_ic_dict = {}

    for factor in factor_names:
        factor_series = feats[factor]

        record, daily_ic_df = analysis_position.score_ic.calc_single_factor_metrics(factor_series, labs, target_label=target_label, factor_name=factor)
        daily_ic_dict[factor] = daily_ic_df

        if abs(record['Rank IC 均值']) > 0.02:
            img_path = os.path.join(IMG_DIR, f"{factor}_{target_label}.html")
            analysis_position.score_ic.save_score_ic_graph(factor_series, label_series, save_path=img_path)
        
        summary_records.append(record)

    df_summary = pd.DataFrame(summary_records).set_index("因子名称")
    print("\n【1. 因子能力综合体检表】:")
    print(df_summary)

    # =========================================================================
    # 维度 2：分层回测 (Quantile Grouping) 与单调性检验
    # =========================================================================
    print(f"\n【2. 正在执行 {n_groups} 分层单调性回测...】")
    
    # 选出 Rank ICIR 绝对值最高的 Top 2 因子进行详细画图展示
    top_factors = df_summary["Rank ICIR"].abs().nlargest(2).index.tolist()
    
    fig, axes = plt.subplots(len(top_factors), 2, figsize=(16, 5 * len(top_factors)))
    if len(top_factors) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, factor in enumerate(top_factors):
        factor_series = feats[factor]
        
        # 1. 绘制累计 Rank IC 曲线
        daily_ic_df = daily_ic_dict[factor]
        cum_rank_ic = daily_ic_df["rank_ic"].cumsum()
        
        axes[i, 0].plot(cum_rank_ic.index.get_level_values("datetime"), cum_rank_ic.values, color="tab:blue", lw=1.5)
        axes[i, 0].set_title(f"{factor} - 累计 Rank IC 曲线 (ICIR={df_summary.loc[factor, 'Rank ICIR']})")
        axes[i, 0].grid(True, linestyle="--", alpha=0.5)
        
        # 2. 计算分层日收益率（新版 qlib 已移除 analysis_model.score_group，这里手动实现等价逻辑）
        # 参考 qlib.contrib.report.analysis_model.analysis_model_performance._group_return
        # 将每天的股票按因子值从高到低排序后等分成 n_groups 组，计算每组每天的平均收益率
        pred_label = pd.DataFrame({"score": factor_series, "label": label_series}).dropna(subset=["score"])
        pred_label = pred_label.sort_values("score", ascending=False)
        group_df = pd.DataFrame(
            {
                g: pred_label.groupby(level="datetime", group_keys=False)["label"].apply(
                    lambda x, g=g: x[len(x) // n_groups * g : len(x) // n_groups * (g + 1)].mean()
                )
                for g in range(n_groups)
            }
        )
        
        # 计算各组累计净值曲线 (组1 ~ 组N)
        cum_group_returns = (1 + group_df).cumprod()
        for g in range(n_groups):
            axes[i, 1].plot(cum_group_returns.index.get_level_values("datetime"), 
                            cum_group_returns[g], label=f"Group {g+1}")
            
        # 绘制多空对冲净值 (Top组 - Bottom组)
        long_short_ret = group_df[n_groups - 1] - group_df[0]
        # 如果因子是负向因子 (IC < 0)，则方向取反
        if df_summary.loc[factor, "Rank IC 均值"] < 0:
            long_short_ret = group_df[0] - group_df[n_groups - 1]
        cum_long_short = (1 + long_short_ret).cumprod()
        axes[i, 1].plot(cum_long_short.index.get_level_values("datetime"), 
                        cum_long_short, label="Long-Short (多空对冲)", color="black", lw=2, linestyle="--")

        axes[i, 1].set_title(f"{factor} - {n_groups} 分层累计收益率与单调性")
        axes[i, 1].legend(loc="upper left")
        axes[i, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

    # =========================================================================
    # 维度 4：因子互相关系数矩阵热力图
    # =========================================================================
    print("\n【3. 绘制因子相关性矩阵热力图...】")
    plt.figure(figsize=(10, 8))
    corr_matrix = feats.corr(method="spearman")
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("10大因子 Spearman 秩相关系数热力图 (排查多重共线性)")
    plt.tight_layout()
    plt.show()

    return df_summary


def main():
    """获取股票池"""
    instruments_name = INSTRUMENTS_NAME
    instruments = get_instruments(instruments_name)  # 获取股票池
    print(f"股票池：{instruments_name}")

    # """ 获取原始数据 """
    # df_raw = get_data(instruments)  # 获取原始数据
    # print("原始数据获取完成")

    """ 计算因子 """
    print("正在加载数据...")
    df_features, df_labels, df_raws = get_features_labels_raws(instruments)  # 计算因子
    print("数据加载完成")

    """ 行业和市值中性化 """
    print("行业和市值中性化 中...")
    df_features = postProcess(df_raws, df_features)  # 行业和市值中性化
    print("行业和市值中性化 完成")

    print(df_features.shape)
    print(df_labels.shape)
    print(df_raws.shape)

    # 3. 执行全面因子评估体系！
    df_summary = run_full_factor_analysis(
        df_features=df_features, 
        df_labels=df_labels, 
        target_label="RET_1D",  # 测试 T+1 -> T+2 收益率预测能力
        n_groups=5              # 5 分层测试
    )

    df_summary.to_csv("summary.csv")



if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
    except:
        traceback.format_exc()
    end_time = time.time()
    print(f"共耗时{end_time - start_time}s")
