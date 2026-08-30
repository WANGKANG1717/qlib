# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from ..graph import ScatterGraph
from ..utils import guess_plotly_rangebreaks


def _concat_pred_label(factor_series, label_series):
    """
    通用对齐工具：将因子（Score/Feature）与标签（Label）按 (datetime, instrument) 索引对齐并剔除缺失值。

    参数
    ----
    factor_series : pd.Series
        因子特征序列
    label_series  : pd.Series
        收益率标签序列

    返回
    ----
    pd.DataFrame
        包含 ['score', 'label'] 两列的清洗后数据，索引为 (datetime, instrument)
    """
    # 1. 统一转换并拼接成双列 DataFrame
    concat_data = pd.DataFrame({"score": factor_series, "label": label_series})
    # 2. 剔除当天特征或标签缺失的股票（保证计算两两对齐）
    return concat_data.dropna(subset=["score", "label"], how="any")


def _get_score_ic(pred_label: pd.DataFrame):
    """

    :param pred_label:
    :return:
    """
    concat_data = pred_label.copy()
    concat_data.dropna(axis=0, how="any", inplace=True)
    _ic = concat_data.groupby(level="datetime", group_keys=False).apply(lambda x: x["label"].corr(x["score"]))
    _rank_ic = concat_data.groupby(level="datetime", group_keys=False).apply(lambda x: x["label"].corr(x["score"], method="spearman"))
    return pd.DataFrame({"ic": _ic, "rank_ic": _rank_ic})


def score_ic_graph(pred_label: pd.DataFrame, show_notebook: bool = True, **kwargs) -> list[list, tuple]:
    """score IC

        Example:


            .. code-block:: python

                from qlib.data import D
                from qlib.contrib.report import analysis_position
                pred_df_dates = pred_df.index.get_level_values(level='datetime')
                features_df = D.features(D.instruments('csi500'), ['Ref($close, -2)/Ref($close, -1)-1'], pred_df_dates.min(), pred_df_dates.max())
                features_df.columns = ['label']
                pred_label = pd.concat([features_df, pred], axis=1, sort=True).reindex(features_df.index)
                analysis_position.score_ic_graph(pred_label)


    :param pred_label: index is **pd.MultiIndex**, index name is **[instrument, datetime]**; columns names is **[score, label]**.


            .. code-block:: python

                instrument  datetime        score         label
                SH600004  2017-12-11     -0.013502       -0.013502
                            2017-12-12   -0.072367       -0.072367
                            2017-12-13   -0.068605       -0.068605
                            2017-12-14    0.012440        0.012440
                            2017-12-15   -0.102778       -0.102778


    :param show_notebook: whether to display graphics in notebook, the default is **True**.
    :return: if show_notebook is True, display in notebook; else return **plotly.graph_objs.Figure** list.
    """
    _ic_df = _get_score_ic(pred_label)

    # 1. 生成基础折线图 (包含 'ic' 和 'rank_ic' 两条曲线)
    _figure = ScatterGraph(
        _ic_df,
        layout=dict(
            title="Score IC",
            xaxis=dict(tickangle=45, rangebreaks=kwargs.get("rangebreaks", guess_plotly_rangebreaks(_ic_df.index))),
        ),
        graph_kwargs={"mode": "lines+markers"},
    ).figure

    # 2. 计算 IC 和 Rank IC 的均值
    ic_mean = _ic_df["ic"].mean()
    rank_ic_mean = _ic_df["rank_ic"].mean()

    # 3. 动态获取两条曲线的颜色 (Trace 0 是 ic, Trace 1 是 rank_ic)
    # 若 trace.line.color 为空，则采用 Plotly 默认主题的前两种颜色 (#636EFA 蓝色, #EF553B 红色)
    plotly_default_colors = ["#636EFA", "#EF553B"]
    
    ic_color = getattr(_figure.data[0].line, "color", None) or plotly_default_colors[0]
    rank_ic_color = getattr(_figure.data[1].line, "color", None) or plotly_default_colors[1]

    # 4. 添加 IC 均值虚线及数值标注
    _figure.add_hline(
        y=ic_mean,
        line_dash="dash",            # 虚线样式
        line_color=ic_color,         # 与 IC 曲线保持同色
        line_width=1.5,
        annotation_text=f"IC Mean: {ic_mean:.4f}",
        annotation_position="top left",     # 标注文字显示在左上方
    )

    # 5. 添加 Rank IC 均值虚线及数值标注
    _figure.add_hline(
        y=rank_ic_mean,
        line_dash="dash",            # 虚线样式
        line_color=rank_ic_color,    # 与 Rank IC 曲线保持同色
        line_width=1.5,
        annotation_text=f"Rank IC Mean: {rank_ic_mean:.4f}",
        annotation_position="bottom left",  # 标注文字显示在左下方（防重叠）
    )

    if show_notebook:
        ScatterGraph.show_graph_in_notebook([_figure])
    else:
        return (_figure,)

def calc_single_factor_metrics(
    factor_series: pd.Series,
    label_df: Union[pd.Series, pd.DataFrame],
    target_label: str = None,
    factor_name: str = "Factor",
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    全面评估单个因子的 IC/Rank IC 表现、统计显著性(t值/p值)与多周期衰减情况。

    参数
    ----
    factor_series : pd.Series
        待评估的单因子序列 (MultiIndex: [datetime, instrument])
    label_df      : pd.DataFrame or pd.Series
        收益率标签数据（可传入包含多周期的 DataFrame，如 RET_1D, RET_1W, RET_1M 等）
    target_label  : str, 可选
        主评估标签列名。若为 None，默认自动使用 label_df 的第一列

    返回
    ----
    record : dict
        因子的综合评估指标字典 (包含 IC均值, ICIR, 胜率, t值, p值, 各周期衰减RankIC等)
    daily_ic_df : pd.DataFrame
        针对主标签计算出的逐日 IC 与 Rank IC 时序数据 (用于后续绘制累计IC曲线)
    """
    # 1. 规范化输入格式与因子名称提取
    if isinstance(label_df, pd.Series):
        label_df = label_df.to_frame()
    elif not isinstance(label_df, pd.DataFrame):
        raise TypeError("label_df 必须是 pd.Series 或 pd.DataFrame 类型")

    if factor_name is None:
        factor_name = getattr(factor_series, "name", "Factor") or "Factor"

    # 确定基准主标签
    if target_label is None:
        target_label = label_df.columns[0]
    elif target_label not in label_df.columns:
        raise ValueError(f"目标标签 '{target_label}' 不在传入的 label_df 中: {list(label_df.columns)}")

    # 2. 计算基准主标签的逐日 IC 和 Rank IC (直接调用同文件内部函数)
    main_label_series = label_df[target_label]
    aligned_main_df = _concat_pred_label(factor_series, main_label_series)
    daily_ic_df = _get_score_ic(aligned_main_df)

    # 3. 统计核心评价指标
    rank_ic = daily_ic_df["rank_ic"].dropna()
    mean_rank_ic = rank_ic.mean()
    std_rank_ic = rank_ic.std()
    rank_icir = mean_rank_ic / (std_rank_ic + 1e-12)
    win_rate = (rank_ic > 0).mean()
    t_stat, p_val = stats.ttest_1samp(rank_ic, 0)

    # 4. 跨周期衰减测试（遍历所有周期标签）
    decay_ics = {}
    for l_col in label_df.columns:
        if l_col == target_label:
            decay_ics[f"RankIC_{l_col}"] = round(mean_rank_ic, 4)
        else:
            l_s = label_df[l_col]
            aligned_decay_df = _concat_pred_label(factor_series, l_s)
            daily_decay_ic = _get_score_ic(aligned_decay_df)["rank_ic"]
            decay_ics[f"RankIC_{l_col}"] = round(daily_decay_ic.mean(), 4) if len(daily_decay_ic.dropna()) > 0 else np.nan

    # 5. 组合生成体检报告字典
    record = {
        "因子名称": factor_name,
        "Rank IC 均值": round(mean_rank_ic, 4),
        "Rank IC 标差": round(std_rank_ic, 4),
        "Rank ICIR": round(rank_icir, 4),
        "IC 胜率": f"{win_rate * 100:.1f}%",
        "t-统计量": round(t_stat, 2),
        "p-值": round(p_val, 4),
        **decay_ics
    }

    return record, daily_ic_df

def save_graph(fig, save_path: str = "score_ic.png", width: int = 1200, height: int = 600, scale: int = 2):
    # 智能根据后缀名保存
    if save_path.lower().endswith(".html"):
        fig.write_html(save_path)
        print(f"[图表保存成功] 交互式网页: {save_path}")
    else:
        try:
            fig.write_image(save_path, width=width, height=height, scale=scale)
            print(f"[图表保存成功] 高清图片: {save_path}")
        except ValueError as e:
            if "kaleido" in str(e).lower():
                raise ImportError("保存静态图片需要安装 kaleido 库，请在终端执行: pip install -U kaleido") from e
            raise e

def save_score_ic_graph(
    pred_label: pd.DataFrame,
    save_path: str = "score_ic.png",
    width: int = 1200,
    height: int = 600,
    scale: int = 2,
    **kwargs
):
    """
    静默生成 Score IC 折线图（含均值虚线）并自动保存至指定路径。

    参数
    ----
    pred_label : pd.DataFrame
        因子特征序列 (MultiIndex: [datetime, instrument]) + 收益率标签序列
    save_path     : str, 默认 'score_ic.png'
        文件保存路径（自动根据后缀识别：支持 .png, .jpg, .jpeg, .pdf, .html）
    width         : int, 默认 1200
        输出图片宽度（像素）
    height        : int, 默认 600
        输出图片高度（像素）
    scale         : int, 默认 2
        高清缩放倍率（仅对静态图片有效，2倍即为2400x1200高清画质）
    **kwargs      : 
        透传给 score_ic_graph 的其他配置参数（例如 rangebreaks 等）

    返回
    ----
    plotly.graph_objs.Figure
        生成的 Plotly 图表对象
    """

    figure_list = score_ic_graph(pred_label=pred_label, show_notebook=False, **kwargs)
    figure = figure_list[0]
    save_graph(figure, save_path, width, height, scale)

    return figure

def _score_ic_graph(pred_label: pd.DataFrame, show_notebook: bool = True, **kwargs) -> list[list, tuple]:
    return score_ic_graph(pred_label, show_notebook=False)
