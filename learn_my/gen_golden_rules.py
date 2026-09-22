import os
import warnings

# 1. 设置环境变量，强制所有新派生的子进程忽略警告（必须放在最前面）
os.environ["PYTHONWARNINGS"] = "ignore"

# 2. 忽略当前主进程的 Python 警告
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# 3. 忽略 Numpy 底层的除以 0（divide）和无效值（invalid）警告
np.seterr(divide="ignore", invalid="ignore")

import os
import qlib
from qlib.constant import REG_US
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, SigAnaRecord, PortAnaRecord
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qlib.contrib.eva.alpha import calc_ic, calc_all_ic

import numpy as np
from qlib.contrib.report import analysis_model, analysis_position

# 开启本地 MLflow 文件存储
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


def calc_total_metrics(df):
    """计算全局的 Mean, Std, ICIR"""
    mean = df.mean()
    std = df.std().replace(0, np.nan)  # 预防分母为 0
    icir = mean / std
    summary = pd.DataFrame({"Mean": mean, "Std": std, "ICIR": icir})
    # 按 Mean 的绝对值降序
    return summary.sort_values(by="Mean", ascending=False, key=abs)


def calc_period_metrics(df, freq):
    """按指定频率切片计算 (M=月, W=周)"""
    mean = df.resample(freq).mean()
    std = df.resample(freq).std().replace(0, np.nan)
    icir = mean / std
    return mean, std, icir


def plot_graph_ic_wate(pred_label: pd.DataFrame, show: bool = False):
    # ==========================================
    # 新增：使用分位数去极值 (掐头去尾各 1%)
    # ==========================================
    lower_bound = pred_label["score"].quantile(0.01)
    upper_bound = pred_label["score"].quantile(0.99)

    # 过滤掉边界外的极端样本
    pred_label = pred_label[(pred_label["score"] >= lower_bound) & (pred_label["score"] <= upper_bound)]

    # ==========================================
    # 重新计算 min, max 和 bin_width
    # ==========================================
    min_score = min(pred_label["score"])
    max_score = max(pred_label["score"])
    bin_width = abs(max_score - min_score) / 25

    # print(f"有效区间: [{min_score:.4f}, {max_score:.4f}]")
    # print(f"bin_width = {bin_width:.4f}")

    result = analysis_position.interval_win_rate.calculate_interval_win_rate(pred_label, bin_width=bin_width)
    if show:
        gf = analysis_position.interval_win_rate.plot_interval_win_rate(result, True)
    else:
        gf = None
    return result, gf


def main():
    # ==========================================
    # 1. 初始化 Qlib
    # ==========================================
    # 注意：如果你要用刚才转换的 1h 数据，路径要改为 binance_data_1h
    provider_uri = r"C:\Users\WANGKANG\.qlib\qlib_data\binance_data"
    qlib.init(provider_uri=provider_uri, region=REG_US, limit_threshold=None)

    market = "all"
    benchmark = "BTC"

    # ==========================================
    # 2. 定义数据处理器 (Data Handler)
    # ==========================================
    data_handler_config = {
        "start_time": "2020-01-01",
        "end_time": "2026-09-01",
        "fit_start_time": "2020-01-01",
        "fit_end_time": "2024-12-31",
        "instruments": market,
        # 标签表达式与名称列
        "label": (["Ref($open, -2) / Ref($open, -1) - 1"], ["LABEL0"]),
        "learn_processors": [
            {"class": "DropnaLabel"},
            {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
        ],
    }
    handler = Alpha158(**data_handler_config)

    df_features = handler.fetch(col_set="feature")
    df_labels = handler.fetch(col_set="label")

    # ==========================================
    # 3. 每日ic计算
    # ==========================================
    ic_dict = {}
    ric_dict = {}

    pred_dict_all = {}

    for feature in df_features.columns.to_list():
        pred_dict_all[feature] = pred = df_features[feature]
    all_ic_res = calc_all_ic(pred_dict_all, df_labels["LABEL0"])

    for feature in df_features.columns.to_list():
        ic_dict[feature] = all_ic_res[feature]["ic"]
        ric_dict[feature] = all_ic_res[feature]["ric"]

    # ==========================================
    # 4. ic指标计算：总/月
    # ==========================================
    # 行: datetime, 列: 158个特征
    ic_df = pd.DataFrame(ic_dict)
    ric_df = pd.DataFrame(ric_dict)

    # 计算全局排行榜 (这里以 Rank IC 为例，评价非线性相关性更准)
    total_ric_summary = calc_total_metrics(ric_df)
    print("🏆 【全局 Rank IC 排行榜 Top 20】")
    print(total_ric_summary.head(20))

    # 计算月度与周度时序指标
    ric_monthly_mean, ric_monthly_std, ric_monthly_icir = calc_period_metrics(ric_df, "ME")
    ric_weekly_mean, ric_weekly_std, ric_weekly_icir = calc_period_metrics(ric_df, "W")

    # ==========================================
    # 5. ic胜率计算
    # ==========================================
    result_list = []
    for feature in total_ric_summary.index.to_list()[:1000]:
        # print("--------------------------------------------------------------------")
        # print(f"feature = {feature}; ric = {total_ric_summary.loc[feature, 'Mean']}")
        pred_label = pd.concat([df_features[feature], df_labels["LABEL0"]], axis=1)
        pred_label.columns = ["score", "label"]
        result, gf = plot_graph_ic_wate(pred_label)
        result["feature"] = feature
        result_list.append(result)

    result_df = pd.concat(result_list)
    # 把 feature 列移到大表的最前面，方便肉眼查看
    cols = ["feature"] + [c for c in result_df.columns if c != "feature"]
    result_df = result_df[cols]
    result_df.to_csv("./result_df.csv", index=False)

    # --------------------------------------------------------------------------------------
    #                                    筛选高胜率，高收益因子
    # --------------------------------------------------------------------------------------
    mask = (result_df["win_rate"] >= 0.52) & (result_df["sample_count"] >= 1000) & (result_df["average_return"] >= 0.01)
    golden_result_df = result_df[mask]

    print("因子 区间 胜率 样本数 胜次数 平均收益 收益中位数")
    for _, r in golden_result_df.iterrows():
        print(f"{r['feature']} {r['ic_interval']} {r['win_rate']*100:.2f}% {int(r['sample_count'])} {int(r['win_count'])} {r['average_return']*100:.2f}% {r['median_return']*100:.2f}%")

    golden_result_df.to_csv("./golden_rules.csv", index=False)


if __name__ == "__main__":
    main()
