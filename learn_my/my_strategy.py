import pandas as pd
import qlib
from qlib.constant import REG_US
from qlib.contrib.data.handler import Alpha158
import time

# ==========================================
# 基础配置与动态时间计算
# ==========================================
# TARGET_DATE = "2026-09-01"  

# 动态往前推 1 年作为起始时间，给时序特征（如均线）留足预热数据
# target_dt = pd.to_datetime(TARGET_DATE)
# start_dt = target_dt - pd.DateOffset(years=1)
# START_DATE = start_dt.strftime("%Y-%m-%d")

START_DATE = "2020-01-01"
TARGET_DATE = "2026-09-01"

PROVIDER_URI = r"C:\Users\WANGKANG\.qlib\qlib_data\binance_data"
MARKET = "all"

if __name__ == "__main__":
    start_time = time.time()
    print("初始化 Qlib...")
    qlib.init(provider_uri=PROVIDER_URI, region=REG_US, limit_threshold=None)

    # ==========================================
    # 获取特征数据
    # ==========================================
    print(f"⏳ 正在获取特征数据 (窗口: {START_DATE} 至 {TARGET_DATE})...")
    data_handler_config = {
        "start_time": START_DATE,
        "end_time": TARGET_DATE, 
        "instruments": MARKET,
        "label": (
            ["Ref($open, -2) / Ref($open, -1) - 1"], 
            ["LABEL0"]
        ),
    }

    handler = Alpha158(**data_handler_config)
    df_all = handler.fetch()


    # ==========================================
    # 加载黄金规则并匹配买入信号
    # ==========================================
    print("📖 正在加载黄金规则...")
    golden_rules = pd.read_csv("golden_rules.csv")

    factor_list =  ["LABEL0"] + list({row.feature for row in golden_rules.itertuples()})
    print(f"factor_list = {factor_list}, length = {len(factor_list)}")

    # 初始化当天全 False 的信号列
    print("⚙️ 正在匹配买入条件 (OR 逻辑)...")
    final_buy_signal = pd.Series(False, index=df_all.index)
    for row in golden_rules.itertuples():
        factor = row.feature
        lower = row.interval_left
        upper = row.interval_right
        
        if factor in df_all.columns:
            # 这里的条件比对会自动在几百天的所有标的上同时完成
            condition = (df_all[factor] >= lower) & (df_all[factor] < upper)
            final_buy_signal = final_buy_signal | condition

    # 提取触发了买入的记录
    buy_records = df_all.loc[final_buy_signal, factor_list].reset_index()

    output_path = "qlib_golden_signals.csv"
    buy_records.to_csv(output_path, index=False)

    print(f"✅ 信号生成完毕！日期: {TARGET_DATE}")
    print(f"📂 JSON 已保存至: {output_path}")
    print(f"📈 共生成了 {len(buy_records)} 条买入指令。")
    end_time = time.time()
    print(f"共用时： {(end_time - start_time):.0f}秒")
