import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def calculate_interval_win_rate(
    pred_label: pd.DataFrame,
    ic_col: str = "score",
    return_col: str = "label",
    bin_width: float = 0.1,
) -> pd.DataFrame:
    """按左闭右开区间统计收益大于 0 的比例。"""
    frame = pred_label.copy()
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
            customdata=np.column_stack([sample_counts, result["win_count"], result["ic_min"], result["ic_max"]]),
            text=win_rates,
            texttemplate="%{text:.1%}",
            textposition="outside",
            hovertemplate=(
                "区间 %{x}<br>胜率 %{y:.2%}<br>样本数 %{customdata[0]:,.0f}"
                "<br>胜次数 %{customdata[1]:,.0f}<br>实际 score %{customdata[2]:.5f} ~ %{customdata[3]:.5f}<extra></extra>"
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
