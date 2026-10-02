"""
此文件用来存储自己编写的工具方法
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qlib.contrib.eva.alpha import calc_all_ic, calc_ic
from qlib.contrib.report import analysis_position


def calc_total_metrics(df):
    """计算全局的 Mean, Std, ICIR"""
    mean = df.mean()
    std = df.std().replace(0, np.nan)  # 预防分母为 0
    icir = mean / std
    return pd.DataFrame({"Mean": mean, "Std": std, "ICIR": icir})


def cal_total_ric_summary(df_features: pd.DataFrame, df_labels: pd.DataFrame):
    pred_dict_all = {}
    for feature in df_features.columns.to_list():
        pred_dict_all[feature] = df_features[feature]
    all_ic_res = calc_all_ic(pred_dict_all, df_labels["LABEL0"])

    # ic_df = pd.DataFrame({f: res["ic"] for f, res in all_ic_res.items()})
    ric_df = pd.DataFrame({f: res["ric"] for f, res in all_ic_res.items()})

    # 计算全局排行榜 (这里以 Rank IC 为例，评价非线性相关性更准)
    ric_summary = calc_total_metrics(ric_df)
    return ric_summary.sort_values(by="Mean", ascending=False, key=abs), ric_df


def risk_analysis__(portfolio_metric_dict, factor_name, freq, print=False):
    from qlib.contrib.evaluate import risk_analysis
    from qlib.utils import flatten_dict
    from qlib.utils.time import Freq

    analysis_freq = "{0}{1}".format(*Freq.parse(freq))
    # backtest info
    report_normal, positions_normal = portfolio_metric_dict.get(analysis_freq)

    # analysis
    analysis = dict()
    analysis["benchmark_return"] = risk_analysis(report_normal["bench"], freq=analysis_freq, N=365, mode="sum")
    analysis["return_without_cost"] = risk_analysis(report_normal["return"], freq=analysis_freq, N=365, mode="sum")
    analysis["return_with_cost"] = risk_analysis(report_normal["return"] - report_normal["cost"], freq=analysis_freq, N=365, mode="sum")
    analysis["excess_return_without_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"], freq=analysis_freq, N=365, mode="sum")
    analysis["excess_return_with_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"] - report_normal["cost"], freq=analysis_freq, N=365, mode="sum")

    analysis_df = pd.concat(analysis)  # type: pd.DataFrame
    # log metrics
    analysis_dict = flatten_dict(analysis_df["risk"].unstack().T.to_dict())
    analysis_dict["factor"] = factor_name
    if print:
        # print out results
        pprint(f"The following are analysis results of benchmark return({analysis_freq}).")
        pprint(analysis["benchmark_return"])
        pprint(f"The following are analysis results of the return with cost({analysis_freq}).")
        pprint(analysis["return_with_cost"])
        pprint(f"The following are analysis results of the excess return with cost({analysis_freq}).")
        pprint(analysis["excess_return_with_cost"])
    return analysis_dict, analysis_df


# ==========================================
# 封装单因子的回测 Worker 函数
# ==========================================
def run_single_backtest(factor_name, strategy_config, executor_config, backtest_config, other_config):
    """ need_qlib_init, provider_uri, region, limit_threshold, freq """
    import qlib
    from qlib.backtest import backtest
    from qlib.backtest.executor import SimulatorExecutor
    from qlib.contrib.strategy import TopkDropoutStrategy

    """
    独立进程运行的函数。
    注意：为降低进程间通信开销，不要把整个 df_features 传进来，只传单列 pred_score。
    """
    config = {
        "factor_name": factor_name,
        "strategy_config": strategy_config,
        "executor_config": executor_config,
        "backtest_config": backtest_config,
        "other_config": other_config,
    }
    portfolio_metric_dict = None
    indicator_dict = None
    try:
        if other_config['need_qlib_init']:
            qlib.init(provider_uri=other_config['provider_uri'], region=other_config['region'], limit_threshold=other_config['limit_threshold'])

        # 实例化
        strategy_obj = TopkDropoutStrategy(**strategy_config)
        executor_obj = SimulatorExecutor(**executor_config)

        # 运行回测
        portfolio_metric_dict, indicator_dict = backtest(executor=executor_obj, strategy=strategy_obj, **backtest_config)

        # 计算评估指标
        analysis_dict, analysis_df = risk_analysis__(portfolio_metric_dict, factor_name, other_config['freq'], print=False)

        return {
            # 回测配置，原样透传回去
            "config": config,
            # 输出
            "analysis_dict": analysis_dict,
            "analysis_df": analysis_df,
            "portfolio_metric_dict": portfolio_metric_dict,
            "indicator_dict": indicator_dict,
        }

    except Exception as e:
        return {
            "config": config,
            "analysis_dict": {"factor": factor_name, "error": str(e)},
            "portfolio_metric_dict": portfolio_metric_dict,
            "indicator_dict": indicator_dict,
        }


# ==========================================
# 多进程主控入口
# ==========================================
def run_all_backtest(ric_summary: pd.DataFrame, df_features: pd.DataFrame, executor_config: dict, backtest_config: dict, other_config: dict):
    """
    other_config: topk, n_drop, freq, provider_uri, region, limit_threshold, need_qlib_init
    """
    from joblib import Parallel, delayed
    from tqdm.auto import tqdm

    results = []
    factor_list = ric_summary.index.to_list()

    # 构建任务列表 (预先在主进程把数据切好，极大地降低子进程的内存拷贝开销)
    tasks = []
    for factor_name in factor_list:
        # 切片提取单列数据
        pred_score = df_features[factor_name].dropna()
        if pred_score.empty:
            continue

        # 判断是否为负向因子
        is_negative = ric_summary.loc[factor_name]["Mean"] < 0

        # 信号反转
        if is_negative:
            pred_score = -pred_score

        strategy_config = {
            "signal": pred_score,
            "topk": other_config["topk"],
            "n_drop": other_config["n_drop"],
        }

        # 把执行所需的所有参数打包
        tasks.append((factor_name, strategy_config, executor_config, backtest_config, other_config))

    results = list(
        tqdm(
            Parallel(n_jobs=-1, return_as="generator")(delayed(run_single_backtest)(*task) for task in tasks),
            total=len(tasks),
            desc="因子回测进度",
        ),
    )

    # 数据汇总与清洗
    valid_results = []
    errors = []

    for result in results:
        res = result["analysis_dict"]
        if isinstance(res, dict) and "error" in res:
            errors.append(res)
        elif res is not None:
            valid_results.append(res)

    # 打印报错详情供排查
    if errors:
        print(f"首个报错因子 [{errors[0]['factor']}]: {errors[0]['error']}")

    # 将回测成功的结果列表转化为 DataFrame
    df_risk = pd.DataFrame(valid_results)
    # 将 'factor' 列设置为索引，以便与 ric_summary 的索引（因子名）对齐
    df_risk.set_index("factor", inplace=True)
    return {
        "df_risk": df_risk,
        "results": results,
    }


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


def plot_factor_stats(factor_stats: pd.DataFrame, prefix: str = "excess_return_with_cost", show: bool = False):
    # 需要绘制的列
    plot_cols = ["Mean", "ICIR", f"{prefix}.mean", f"{prefix}.annualized_return", f"{prefix}.information_ratio", f"{prefix}.max_drawdown"]

    # 剔除无效行 (建议加上 .copy() 避免报 SettingWithCopyWarning)
    df_plot = factor_stats[plot_cols].dropna(how="all").copy()

    # 创建 5行1列 的子图，共享 X 轴，并自动使用列名作为每个子图的标题
    fig = make_subplots(rows=len(plot_cols), cols=1, shared_xaxes=True, vertical_spacing=0.04, subplot_titles=plot_cols)  # 子图之间的垂直间距

    # 遍历指标，分别添加到对应的子图中
    for i, col in enumerate(plot_cols):
        fig.add_trace(go.Bar(x=df_plot.index, y=df_plot[col], name=col, marker_color="royalblue"), row=i + 1, col=1)  # 统一设置柱子颜色

    # 设置全局布局、高度和主题
    fig.update_layout(
        title_text="因子表现与风险指标对比",
        title_font_size=20,
        height=1200,  # 增加图表总高度以容纳5个子图
        showlegend=False,  # 隐藏图例（因为每个子图已经有独立标题）
        template="plotly_white",  # 使用清爽的白色背景主题
        hovermode="x unified",  # 鼠标悬停时，统一显示同一因子在所有指标下的数值
    )

    # 调整底层 X 轴标签倾斜度，防止因子名重叠
    fig.update_xaxes(tickangle=45, row=len(plot_cols), col=1)

    # 展示图表
    if show:
        fig.show()

    return fig


def plot_ic_decay_heatmap(ric_df: pd.DataFrame, period: str = "year", top_k: int = 25, width: int = 800, height: int = 600, show: bool = False):
    """
    绘制因子 RankIC 衰减热力图，支持按 'year'（年）或 'month'（月）进行周期聚合。
    """
    # 根据传入的周期参数进行动态聚合
    if period.lower() == "year":
        # 按年聚合 (格式: 2023, 2024)
        grouped_df = ric_df.groupby(ric_df.index.year).mean()
        x_label = "Year"
        title_prefix = "Annual"
    elif period.lower() == "month":
        # 按年月聚合 (格式: 2023-01, 2023-02)
        grouped_df = ric_df.groupby(ric_df.index.to_period("M").astype(str)).mean()
        x_label = "Month"
        title_prefix = "Monthly"
    else:
        raise ValueError("period 参数必须是 'year' 或 'month'")

    # 提取 Top 因子并转置矩阵
    top_factors = ric_df.mean().abs().sort_values(ascending=False).head(top_k).index
    plot_data = grouped_df[top_factors].T

    # 绘制交互式热力图
    fig = px.imshow(
        plot_data,
        color_continuous_scale="RdBu_r",  # 红正蓝负
        color_continuous_midpoint=0,  # 强制以 0 为色带中点
        text_auto=".3f",  # 显示 3 位小数
        title=f"{title_prefix} RankIC Decay Heatmap (Top {top_k} Factors)",
        labels=dict(x=x_label, y="Factor", color=f"{title_prefix} RankIC"),
        aspect="auto",
    )

    # 根据周期动态调整 X 轴刻度显示策略
    xaxis_config = dict(tickangle=45)
    if period.lower() == "year":
        xaxis_config["tickmode"] = "linear"  # 年份较少，强制显示每一年
    else:
        xaxis_config["tickmode"] = "auto"  # 月份较多，开启自动跳级抽样防重叠
        xaxis_config["nticks"] = 20  # 建议最多显示 20 个刻度标签

    # 优化图表排版
    fig.update_layout(
        width=width,
        height=height,
        xaxis=xaxis_config,
        plot_bgcolor="white",
        # 增大底部边距 (b=100)，确保倾斜的月份文字有足够的空间显示完整
        margin=dict(l=100, r=50, t=80, b=100),
    )

    if show:
        fig.show()
    return fig


def plot_corr_matrix(corr_matrix: pd.DataFrame, show: bool = False):
    # 绘制交互式热力图
    fig = px.imshow(
        corr_matrix,
        color_continuous_scale="RdBu_r",
        zmin=-1,  # 强制色带下限为 -1
        zmax=1,  # 强制色带上限为 1
        title="因子全局相关性热力图 (Spearman)",
    )

    # 优化布局尺寸
    fig.update_layout(width=900, height=800, margin=dict(l=100, r=100, t=100, b=100))

    if show:
        fig.show()
    return fig


def plot_ordered_filtered_heatmap(corr_matrix: pd.DataFrame, threshold: float = 0.5, width: int = 900, height: int = 900, show=False):
    """
    按相关性矩阵的索引顺序严格过滤：
    - 从前向后遍历，若发现高相关，始终保留靠前的特征，剔除靠后的特征。
    - 被剔除的特征不再参与后续的剔除判定。
    """
    print(f"⏳ 正在处理相关性矩阵 (共 {corr_matrix.shape[0]} 个特征)...")
    corr_abs = corr_matrix.abs()
    # 提取按照顺序排列的所有因子名
    all_features = corr_matrix.index.tolist()
    to_drop = set()  # 用集合记录要剔除的因子，查找速度快

    # 双重循环：正向贪心遍历
    for i in range(len(all_features)):
        f1 = all_features[i]

        # 如果当前因子 f1 已经被排在前面的大哥淘汰了，那它就没有资格去淘汰别人
        if f1 in to_drop:
            continue

        # 让 f1 和排在它后面的所有小弟 (f2) 进行 PK
        for j in range(i + 1, len(all_features)):
            f2 = all_features[j]

            # 如果 f2 已经被淘汰，跳过
            if f2 in to_drop:
                continue

            # 如果相关性超标，果断淘汰排在后面的 f2
            if corr_abs.loc[f1, f2] > threshold:
                to_drop.add(f2)

    # 按照原本的顺序，挑出幸存的因子
    kept_features = [f for f in all_features if f not in to_drop]

    print(f"✅ 顺序过滤完成 (阈值: {threshold})！")
    print(f"   - 原始特征数: {len(all_features)}")
    print(f"   - 剔除特征数: {len(to_drop)}")
    print(f"   - 保留特征数: {len(kept_features)}")

    if len(kept_features) < 2:
        print("⚠️ 警告：保留下来的特征不足 2 个，无法绘制热力图，请调高阈值！")
        return kept_features

    # 截取幸存因子的数据，计算全新的相关性矩阵
    new_corr_matrix = corr_matrix.loc[kept_features, kept_features]

    # 使用 Plotly 绘制新矩阵
    fig = px.imshow(
        new_corr_matrix,
        color_continuous_scale="RdBu_r",  # 蓝(-1) - 白(0) - 红(1)
        zmin=-1,
        zmax=1,
        title=f"Filtered Feature Correlation Heatmap (Threshold: {threshold}, Keep First)",
        labels=dict(color="Spearman Corr"),
        aspect="auto",
    )

    fig.update_layout(width=width, height=height, xaxis=dict(tickangle=45), plot_bgcolor="white", margin=dict(l=100, r=50, t=80, b=100))

    # 若特征在30个以内，直接把数值印在格子上
    if len(kept_features) <= 30:
        fig.update_traces(text=new_corr_matrix.round(2), texttemplate="%{text}")

    if show:
        fig.show()

    return kept_features, fig


def pca_explained_variance(df: pd.DataFrame, show: bool = False):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # ==========================================
    # 1. 去除空数据
    # ==========================================
    df_clean = df.dropna()

    print(f"⏳ 正在对 {len(df.columns)} 个精选因子进行 PCA 分析...")

    # ==========================================
    # 2. 标准化 (Z-score Scaling)
    # ==========================================
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_clean)

    # ==========================================
    # 3. 运行 PCA 模型
    # ==========================================
    pca = PCA()
    pca.fit(scaled_data)

    # 计算各主成分的解释方差及累计方差
    exp_var = pca.explained_variance_ratio_ * 100
    cum_var = np.cumsum(exp_var)
    x_labels = [f"PC{i}" for i in range(1, len(exp_var) + 1)]

    # ==========================================
    # 4. 可视化：Plotly 帕累托图 (条形图 + 累计折线图)
    # ==========================================
    fig = go.Figure()

    # 单个主成分贡献柱状图
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=exp_var,
            name="Individual Variance",
            marker_color="royalblue",
            text=[f"{v:.1f}%" for v in exp_var],
            textposition="auto",
        )
    )

    # 累计贡献折线图
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=cum_var,
            name="Cumulative Variance",
            mode="lines+markers",
            marker=dict(color="crimson", size=8),
            yaxis="y2",
        )
    )

    fig.update_layout(
        title=f"PCA Explained Variance for {len(df.columns)} Filtered Factors",
        width=900,
        height=600,
        plot_bgcolor="white",
        yaxis=dict(title="Individual Variance (%)"),
        yaxis2=dict(title="Cumulative Variance (%)", overlaying="y", side="right", range=[0, 105]),
        legend=dict(x=0.01, y=1.05, orientation="h"),
    )

    if show:
        fig.show()

    return pca, fig


def run_topk_ndrop_grid_search(pred_score, executor_config: dict, backtest_config: dict, other_config: dict):
    from joblib import Parallel, delayed
    from tqdm.auto import tqdm

    factor_name = "LightGBM_T_N_Matrix"

    # 从 other_config 中安全提取网格搜索的边界参数，提供默认值以防未配置
    topk_start = other_config.get("topk_start", 5)
    topk_end = other_config.get("topk_end", 25)
    topk_step = other_config.get("topk_step", 1)
    
    n_drop_start = other_config.get("n_drop_start", 1)
    n_drop_step = other_config.get("n_drop_step", 1)

    strategy_configs = [
        {
            "signal": pred_score,
            "topk": topk,
            "n_drop": n_drop,
        }
        for topk in range(topk_start, topk_end + 1, topk_step)
        for n_drop in range(n_drop_start, topk + 1, n_drop_step)
    ]
    print(f"🚀 共生成 {len(strategy_configs)} 组参数，开始使用 Joblib 并行回测...")

    tasks = [(factor_name, strategy_config, executor_config, backtest_config, other_config) for strategy_config in strategy_configs]
    results = list(tqdm(
        Parallel(n_jobs=-1, return_as="generator")(
                delayed(run_single_backtest)(*task) for task in tasks
            ),
            total=len(tasks), 
            desc="参数网格搜索回测进度"
        )
    )
    return results


def plot_topk_and_n_drop_matrix(bc_results, show: bool = False):
    """ 绘制 TopK 与 N_Drop 网格搜索图 """
    # ==========================================
    # 1. 提取数据并构建 DataFrame 表
    # ==========================================
    records = []
    for res in bc_results:
        topk = res['config']['strategy_config']['topk']
        n_drop = res['config']['strategy_config']['n_drop']
        ir = res["analysis_dict"]["excess_return_with_cost.information_ratio"]
            
        records.append({
            "topk": topk,
            "n_drop": n_drop,
            "Information_Ratio": ir
        })

    df_grid = pd.DataFrame(records)

    # 打印表格前 5 行检查
    print("✅ 参数网格提取完成，数据预览：")
    print(df_grid.head())

    # ==========================================
    # 转换为透视表 (Pivot Table)
    # ==========================================
    # 因为 n_drop 的取值范围是 range(1, topk)，所以矩阵的右上角会是 NaN（自动留空）
    pivot_df = df_grid.pivot(index="n_drop", columns="topk", values="Information_Ratio")

    # ==========================================
    # 绘制 2D 热力图观察参数最佳区域
    # ==========================================
    fig_heatmap = px.imshow(
        pivot_df,
        labels=dict(x="TopK (买入阈值)", y="N_Drop (卖出阈值)", color="IR (扣费超额)"),
        x=pivot_df.columns,
        y=pivot_df.index,
        title="网格搜索: TopK 与 N_Drop 组合的信息比率 (IR) 热力图",
        color_continuous_scale="RdYlGn",  # 使用 红-黄-绿 渐变（绿色代表高 IR）
        text_auto=".2f",                  # 在方格中显示 2 位小数
        aspect="auto"
    )
    # 翻转 Y 轴，让 n_drop 较小的值排在上方，更符合阅读习惯
    fig_heatmap.update_layout(yaxis=dict(autorange="reversed"), height=600)

    if show:
        fig_heatmap.show()

    # ==========================================
    # 绘制 3D 散点/曲面图观察立体趋势
    # ==========================================
    fig_3d = px.scatter_3d(
        df_grid, 
        x="topk", 
        y="n_drop", 
        z="Information_Ratio",
        color="Information_Ratio",
        color_continuous_scale="Viridis",
        title="TopK 与 N_Drop 参数空间 3D 视图"
    )
    fig_3d.update_traces(marker=dict(size=6))
    fig_3d.update_layout(scene=dict(
        xaxis_title='TopK',
        yaxis_title='N_Drop',
        zaxis_title='Information Ratio'
    ), height=700)

    if show:
        fig_3d.show()

    return fig_heatmap, fig_3d


def plot_data_distribution(df: pd.DataFrame, col_name: str = None, show: bool = True):
    """ 展示数据分布直方图，默认使用第一列作为标签 """
    if col_name is None:
        label_col = df.columns[0]
    else:
        label_col = col_name
    label_data = df[[label_col]].dropna()

    # 使用 plotly express 绘制直方图
    fig = px.histogram(
        label_data, 
        x=label_col, 
        nbins=100, 
        title=f"数据分布 {label_col}",
        labels={label_col: "标签值"},
        opacity=0.75,
        color_discrete_sequence=['royalblue'],
        marginal="box"  # 在顶部附加箱线图，方便查看分布极值与四分位数
    )

    # 优化图表样式
    fig.update_layout(
        yaxis_title="频数",
        template="plotly_white",
        title_x=0.5,  # 标题居中
        font=dict(size=12)
    )

    # 在 notebook 中显示图表
    if show:
        fig.show()
    return fig

def interactive_signal_dashboard(pred_label: pd.DataFrame):
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    # 1. 参数设置：设置要展示的 Top K 和 Bottom K 数量
    K = 10 
    # 设置你要单独提取展示的标的名称（根据你的数据源，如果叫 BTCUSDT 请自行修改）
    BENCHMARK_SYMBOL = "BTC" 

    # 2. 提取所有交易日并排序
    dates = pred_label.index.get_level_values('datetime').unique().sort_values()
    # 预先生成所有日期的字符串列表，用于下拉菜单显示
    date_strs = [d.strftime('%Y-%m-%d') if isinstance(d, pd.Timestamp) else str(d) for d in dates]

    # state 用于记录当前索引，以及防止触发死循环的 updating 锁
    state = {'idx': 0, 'updating': False}  

    # 3. 创建 UI 组件
    btc_display = widgets.HTML()  
    btn_prev = widgets.Button(description="⬅️ 前一天", button_style='info')
    btn_next = widgets.Button(description="后一天 ➡️", button_style='info')

    date_dropdown = widgets.Dropdown(
        options=list(zip(date_strs, range(len(dates)))),
        value=0,
        description='📅 跳转日期:',
        layout=widgets.Layout(width='250px', margin='0 20px')
    )

    out_top = widgets.Output()
    out_bottom = widgets.Output()

    # =========== 核心修复：独立出底部的总收益显示组件 ===========
    top_summary_display = widgets.HTML()
    bot_summary_display = widgets.HTML()
    # ==========================================================

    # 辅助函数：安全地获取特定日期特定标的的数据
    def get_symbol_data(dt, symbol):
        try:
            row = pred_label.loc[(dt, symbol)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return row['score'], row['label']
        except KeyError:
            return None, None

    # 辅助函数：将数值格式化为带颜色的 HTML 文本
    def format_val_html(val, is_label=False):
        if pd.isna(val):
            return "<span style='color: #888;'>暂无数据</span>"
        
        if is_label:
            color = "#00ff00" if val > 0 else "#ff5252" if val < 0 else "#ffffff"
            return f"<span style='color: {color}; font-weight: bold;'>{val:+.4%}</span>"
        else:
            return f"<span style='color: #64b5f6; font-weight: bold;'>{val:.6f}</span>"

    # 4. 核心更新逻辑
    def update_dashboard(index):
        # 【同步 UI】: 确保点按钮时，下拉框的显示也能跟着变，加锁防止重复触发
        state['updating'] = True
        date_dropdown.value = index
        state['updating'] = False
        
        current_date = dates[index]
        date_str = date_strs[index]
        
        # 获取并渲染 BTC 顶部数据
        curr_score, curr_label = get_symbol_data(current_date, BENCHMARK_SYMBOL)
        if index > 0:
            prev_date = dates[index - 1]
            prev_score, prev_label = get_symbol_data(prev_date, BENCHMARK_SYMBOL)
            prev_date_str = date_strs[index - 1]
        else:
            prev_score, prev_label = None, None
            prev_date_str = "无前日记录"
            
        btc_html = f"""
            <div style="background-color: #1e1e1e; padding: 15px 20px; border-radius: 8px; margin-bottom: 15px; border: 1px solid #444; display: flex; justify-content: space-around; align-items: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
                <div style="text-align: center; flex: 1;">
                    <h4 style="margin: 0 0 10px 0; color: #aaa;">⏪ 前日 ({BENCHMARK_SYMBOL}) - <span style="font-weight:normal;">{prev_date_str}</span></h4>
                    <div style="font-size: 15px; color: #ccc;">
                        得分: {format_val_html(prev_score, False)} &nbsp;&nbsp;|&nbsp;&nbsp; 收益: {format_val_html(prev_label, True)}
                    </div>
                </div>
                <div style="border-left: 2px dashed #555; height: 50px; margin: 0 20px;"></div>
                <div style="text-align: center; flex: 1;">
                    <h4 style="margin: 0 0 10px 0; color: #fff;">▶️ 当日 ({BENCHMARK_SYMBOL}) - <span style="font-weight:normal;">{date_str}</span></h4>
                    <div style="font-size: 17px; color: #fff;">
                        得分: {format_val_html(curr_score, False)} &nbsp;&nbsp;|&nbsp;&nbsp; 收益: {format_val_html(curr_label, True)}
                    </div>
                </div>
            </div>
        """
        btc_display.value = btc_html
        
        # 截取当天数据并按 score 降序排列
        day_df = pred_label.xs(current_date, level='datetime').sort_values(by='score', ascending=False)
        
        max_abs_label = day_df['label'].abs().max()
        if pd.isna(max_abs_label) or max_abs_label == 0:
            max_abs_label = 1  
        
        def style_label(val):
            if pd.isna(val) or val == 0:
                return ''
            ratio = min(abs(val) / max_abs_label, 1.0)
            alpha = 0.15 + 0.7 * ratio
            if val > 0:
                return f'background-color: rgba(0, 200, 83, {alpha}); '
            else:
                return f'background-color: rgba(213, 0, 0, {alpha});'

        top_k_df = day_df.head(K)
        bottom_k_df = day_df.tail(K)
        
        # 计算统计汇总数据
        top_total_ret = top_k_df['label'].sum(skipna=True)
        top_mean_ret = top_k_df['label'].mean(skipna=True)
        
        bot_total_ret = bottom_k_df['label'].sum(skipna=True)
        bot_mean_ret = bottom_k_df['label'].mean(skipna=True)
        
        style_method = 'map' if hasattr(top_k_df.style, 'map') else 'applymap'
        
        styled_top = top_k_df.style\
            .background_gradient(cmap='Greens', subset=['score'])\
            .format({'score': '{:.6f}', 'label': '{:.6f}'})
        getattr(styled_top, style_method)(style_label, subset=['label'])
            
        styled_bottom = bottom_k_df.style\
            .background_gradient(cmap='Reds', subset=['score'])\
            .format({'score': '{:.6f}', 'label': '{:.6f}'})
        getattr(styled_bottom, style_method)(style_label, subset=['label'])
        
        # 表格渲染区域：仅更新表格内容
        with out_top:
            clear_output(wait=True)
            print(f"🟢 Top {K} 标的 (得分最高)")
            display(styled_top)
            
        with out_bottom:
            clear_output(wait=True)
            print(f"🔴 Bottom {K} 标的 (得分最低)")
            display(styled_bottom)

        t_color = "#00ff00" if top_total_ret > 0 else "#ff5252" if top_total_ret < 0 else "#ffffff"
        top_summary_display.value = (
            f"<div style='margin-top: 8px; padding: 6px; border: 1px solid #444; border-radius: 4px; text-align: center; background-color: #1e1e1e; font-size: 14px; color: #eeeeee;'>"
            f"总收益: <span style='color: {t_color}; font-weight: bold;'>{top_total_ret:+.4%}</span> &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"均值: <span style='color: {t_color};'>{top_mean_ret:+.4%}</span></div>"
        )
        
        b_color = "#00ff00" if bot_total_ret > 0 else "#ff5252" if bot_total_ret < 0 else "#ffffff"
        bot_summary_display.value = (
            f"<div style='margin-top: 8px; padding: 6px; border: 1px solid #444; border-radius: 4px; text-align: center; background-color: #1e1e1e; font-size: 14px; color: #eeeeee;'>"
            f"总收益: <span style='color: {b_color}; font-weight: bold;'>{bot_total_ret:+.4%}</span> &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"均值: <span style='color: {b_color};'>{bot_mean_ret:+.4%}</span></div>"
        )
        # =================================================================

    # 5. 按钮与下拉框事件回调
    def on_prev_clicked(b):
        if state['idx'] > 0:
            state['idx'] -= 1
            update_dashboard(state['idx'])

    def on_next_clicked(b):
        if state['idx'] < len(dates) - 1:
            state['idx'] += 1
            update_dashboard(state['idx'])

    def on_dropdown_change(change):
        if change['name'] == 'value' and not state['updating']:
            state['idx'] = change['new']
            update_dashboard(state['idx'])

    btn_prev.on_click(on_prev_clicked)
    btn_next.on_click(on_next_clicked)
    date_dropdown.observe(on_dropdown_change)

    # 6. 页面布局与展示
    controls = widgets.HBox(
        [btn_prev, date_dropdown, btn_next], 
        layout=widgets.Layout(justify_content='center', align_items='center', margin='10px 0 20px 0')
    )

    # =========== 核心修复：将单独的表格 Output 和独立的汇总 HTML 组合打包起来 ===========
    tables = widgets.HBox(
        [
            widgets.VBox([out_top, top_summary_display]), 
            widgets.VBox([out_bottom, bot_summary_display])
        ],
        layout=widgets.Layout(justify_content='space-around')
    )
    # ===========================================================================

    # 初始化
    update_dashboard(state['idx'])

    # 组装完整的 UI
    display(widgets.VBox([btc_display, controls, tables]))
