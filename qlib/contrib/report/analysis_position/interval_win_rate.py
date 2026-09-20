import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def calculate_interval_win_rate(
    pred_label: pd.DataFrame,
    ic_col: str = "score",
    return_col: str = "label",
    bin_width: float = 0.1,
) -> pd.DataFrame:
    """按左闭右开区间统计收益大于 0 的比例。"""
    frame = pred_label.copy().reset_index()
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()

    # 用整数区间编号避免 0.01 这类浮点数直接分组产生精度问题。
    frame["bin_id"] = np.floor(frame[ic_col] / bin_width + 1e-12).astype("int64")
    frame["is_win"] = frame[return_col] > 0

    result = (
        frame.groupby("bin_id", sort=True)
        .agg(
            ic_min=(ic_col, "min"),
            ic_max=(ic_col, "max"),
            sample_count=(return_col, "size"),
            win_count=("is_win", "sum"),
            # total_return=(return_col, "sum"),
            average_return=(return_col, "mean"),
            median_return=(return_col, "median"),
            instrument_count=("instrument", "nunique") # 增加标的数量统计（nunique 表示统计该区间内出现过多少个不同的加密货币）
        )
        .reset_index()
    )
    result["interval_left"] = result["bin_id"] * bin_width
    result["interval_right"] = (result["bin_id"] + 1) * bin_width
    result["ic_interval"] = result.apply(
        lambda row: f"[{row['interval_left']:.2f}, {row['interval_right']:.2f})",
        axis=1,
    )
    result["win_rate"] = result["win_count"] / result["sample_count"]
    return result

def plot_interval_win_rate(result: pd.DataFrame, show_notebook=True):
    labels = result["ic_interval"].astype(str).tolist()
    win_rates = result["win_rate"].to_numpy(dtype=float)
    sample_counts = result["sample_count"].to_numpy(dtype=int)
    average_returns = result["average_return"].to_numpy(dtype=float)
    median_returns = result["median_return"].to_numpy(dtype=float)

    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.38, 0.20, 0.21, 0.21],
        subplot_titles=("胜率", "样本数（对数轴）", "平均收益", "收益中位数"),
    )
    figure.add_trace(
        go.Bar(
            x=labels,
            y=win_rates,
            marker_color=np.where(win_rates >= 0.5, "#2E8B57", "#D95F59"),
            customdata=np.column_stack([sample_counts, result["win_count"], result["ic_min"], result["ic_max"], result["average_return"], result["median_return"], result["instrument_count"]]),
            text=win_rates,
            texttemplate="%{text:.1%}",
            textposition="outside",
            hovertemplate=(
                "区间 %{x}<br>"
                "胜率 %{y:.2%}<br>"
                "样本数 %{customdata[0]:,.0f}<br>"
                "胜次数 %{customdata[1]:,.0f}<br>"
                "实际 score %{customdata[2]:.2f} ~ %{customdata[3]:.2f}<br>"
                "平均收益 %{customdata[4]:.4%}<br>"
                "收益中位数 %{customdata[5]:.4%}<br>"
                "标的数量 %{customdata[6]:,.0f}<br>"
                "<extra></extra>"
            ),
            name="胜率",
        ),
        row=1,
        col=1,
    )
    figure.add_hline(y=0.5, line_dash="dash", line_color="#555555", row=1, col=1)
    figure.add_trace(
        go.Bar(
            x=labels,
            y=sample_counts,
            marker_color="#4C78A8",
            hovertemplate="区间 %{x}<br>样本数 %{y:,.0f}<extra></extra>",
            name="样本数",
        ),
        row=2,
        col=1,
    )

    for row, values, name, color in (
        (3, average_returns, "平均收益", "#F28E2B"),
        (4, median_returns, "收益中位数", "#17A2B8"),
    ):
        figure.add_trace(
            go.Bar(
                x=labels,
                y=values,
                marker_color=np.where(values >= 0, color, "#D95F59"),
                hovertemplate=f"区间 %{{x}}<br>{name} %{{y:.4%}}<extra></extra>",
                name=name,
            ),
            row=row,
            col=1,
        )
        figure.add_hline(y=0, line_color="#777777", line_width=1, row=row, col=1)

    figure.update_yaxes(tickformat=".1%", row=1, col=1)
    figure.update_yaxes(type="log", row=2, col=1)
    figure.update_yaxes(tickformat=".1%", row=3, col=1)
    figure.update_yaxes(tickformat=".1%", row=4, col=1)
    figure.update_xaxes(tickangle=-55, title_text="预测 Score 区间", row=4, col=1)
    figure.update_layout(
        title="预测 Score 区间统计",
        height=1050,
        showlegend=False,
        hovermode="x unified",
        bargap=0.12,
        margin=dict(t=90, b=130),
    )
    if show_notebook:
        figure.show()
    return figure


def interval_win_rate_graph(
    pred_label: pd.DataFrame,
    ic_col: str = "score",
    return_col: str = "label",
    bin_width: float = 0.1,
    show_notebook: bool = True,
):
    """生成可缩放的 Plotly 区间胜率图，供 Notebook 和 WebApp 共用。"""
    result = calculate_interval_win_rate(pred_label, ic_col, return_col, bin_width)
    return plot_interval_win_rate(result, show_notebook)

def calculate_factor_time_series_stats(
    pred_label: pd.DataFrame,
    lower_bound: float,
    upper_bound: float,
    is_short: bool = True
) -> pd.DataFrame:
    """计算单因子特定区间的时序切片统计数据"""
    df = pred_label.dropna().copy()
    
    # 截取落入区间的样本
    df_filtered = df[(df["score"] >= lower_bound) & (df["score"] < upper_bound)].copy()
    
    if df_filtered.empty:
        return pd.DataFrame()
        
    # 做多/做空 逻辑翻转
    if is_short:
        df_filtered["is_win"] = df_filtered["label"] < 0     
        df_filtered["actual_return"] = -df_filtered["label"] 
    else:
        df_filtered["is_win"] = df_filtered["label"] > 0
        df_filtered["actual_return"] = df_filtered["label"]

    # 按月重采样统计
    if isinstance(df_filtered.index, pd.MultiIndex):
        df_filtered = df_filtered.reset_index()
        
    df_filtered.set_index("datetime", inplace=True)
    
    # 按月(Month End)汇总
    monthly_stats = df_filtered.resample("ME").agg(
        sample_count=("is_win", "size"),
        win_count=("is_win", "sum"),
        avg_return=("actual_return", "mean")
    )
    
    # 计算月度胜率 (分母为 0 时填充 NaN，防止报错)
    monthly_stats["win_rate"] = (monthly_stats["win_count"] / monthly_stats["sample_count"]).replace([np.inf, -np.inf], np.nan)

    # 计算累计收益 (Cumulative Sum)
    monthly_stats["cumulative_return"] = monthly_stats["avg_return"].cumsum()
    
    return monthly_stats


def plot_factor_time_series(
    monthly_stats: pd.DataFrame,
    feature: str,
    lower_bound: float,
    upper_bound: float,
    is_short: bool = True,
    show_notebook: bool = True
):
    """根据统计数据绘制 4 面板时序体检交互图表"""
    if monthly_stats.empty:
        print(f"❌ 警告: 在区间 [{lower_bound}, {upper_bound}) 内没有找到任何样本！")
        return None

    direction = "做空" if is_short else "做多"
    title_text = f"【{feature}】区间 [{lower_bound}, {upper_bound}) 时序体检报告 ({direction})"
    
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=(
            "每月触发次数 (检验时间均匀度)", 
            f"每月{direction}胜率 (检验策略衰减)", 
            f"每月{direction}平均收益 (检验盈亏同源)",
            f"{direction}累计收益曲线 (假定等额本金)" # 新增标题
        ),
        row_heights=[0.20, 0.25, 0.25, 0.30] # 分配第四张图多一点空间
    )

    # ==========================================
    # 构建统一的悬停数据和模板
    # ==========================================
    def build_customdata(df):
        return np.column_stack([
            df["sample_count"],
            df["win_rate"].fillna(0), # 填充 0 仅为了防止前端格式化报错
            df["avg_return"],
            df["cumulative_return"]
        ])

    unified_hover = (
        "年月: %{x|%Y-%m}<br>"
        "触发次数: %{customdata[0]:,.0f}<br>"
        "胜率: %{customdata[1]:.2%}<br>"
        "平均收益: %{customdata[2]:.2%}<br>"
        "累计收益: %{customdata[3]:.2%}<extra></extra>"
    )

    # 生成包含所有维度的完整 customdata
    cd_all = build_customdata(monthly_stats)

    # 图1：触发次数 (柱状图)
    fig.add_trace(
        go.Bar(
            x=monthly_stats.index, y=monthly_stats["sample_count"],
            name="触发次数", marker_color="#4C78A8",
            customdata=cd_all, hovertemplate=unified_hover
        ), row=1, col=1
    )

    # 图2：胜率 (折线图)
    valid_win = monthly_stats.dropna(subset=["win_rate"])
    cd_valid = build_customdata(valid_win)
    fig.add_trace(
        go.Scatter(
            x=valid_win.index, y=valid_win["win_rate"],
            mode="lines+markers", name="胜率",
            line=dict(color="#E45756", width=2.5), marker=dict(size=6),
            customdata=cd_valid, hovertemplate=unified_hover
        ), row=2, col=1
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=2, col=1)

    # 图3：平均收益 (柱状图)
    colors = np.where(monthly_stats["avg_return"] >= 0, "#2E8B57", "#D95F59")
    fig.add_trace(
        go.Bar(
            x=monthly_stats.index, y=monthly_stats["avg_return"],
            name="平均收益", marker_color=colors,
            customdata=cd_all, hovertemplate=unified_hover
        ), row=3, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)

    # 【新增】图4：累计收益 (面积折线图)
    fig.add_trace(
        go.Scatter(
            x=monthly_stats.index, y=monthly_stats["cumulative_return"],
            mode="lines", name="累计收益",
            line=dict(color="#17A2B8", width=3),
            fill='tozeroy', fillcolor="rgba(23, 162, 184, 0.2)", # 增加半透明面积填充，强化视觉
            customdata=cd_all, hovertemplate=unified_hover
        ), row=4, col=1
    )
    fig.add_hline(y=0, line_dash="solid", line_color="#777777", row=4, col=1)

    # 格式化布局
    fig.update_yaxes(tickformat=".1%", row=2, col=1)
    fig.update_yaxes(tickformat=".2%", row=3, col=1)
    fig.update_yaxes(tickformat=".0%", row=4, col=1) # 累计收益通常较大，整数百分比即可
    
    fig.update_layout(
        title=title_text, 
        height=1100, # 拉高整体画布以容纳 4 张图
        showlegend=False, 
        hovermode="x unified", 
        template="plotly_white", 
        margin=dict(t=80, b=40)
    )
    
    if show_notebook:
        fig.show()
        
    return fig

def factor_time_series_graph(
    pred_label: pd.DataFrame,
    feature: str,
    lower_bound: float,
    upper_bound: float,
    is_short: bool = True,
    show_notebook: bool = True
):
    # 1. 先计算出统计数据
    stats_df = calculate_factor_time_series_stats(
        pred_label=pred_label,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        is_short=is_short
    )

    # 2. 再将统计数据传给绘图函数
    fig = plot_factor_time_series(
        monthly_stats=stats_df,
        feature=feature,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        is_short=is_short,
        show_notebook=show_notebook
    )
    return fig
