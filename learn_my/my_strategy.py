import time

import pandas as pd
import redis

import qlib
from qlib.constant import REG_US
from qlib.contrib.data.handler import Alpha158

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
STRATEGY_NAME = "MyFirstStrategy"

def save_to_redis(redis_client: redis.Redis, buy_records: pd.DataFrame, target_date: str):
    """
    将生成的买入信号写入 Redis，供 Freqtrade 读取
    """
    print("💾 开始将信号写入 Redis...")
    
    # 1. 如果没有任何信号，必须为 target_date 写入 _COMPLETED_ 标记，防止 Freqtrade 阻塞死等
    if buy_records.empty:
        today_str = pd.to_datetime(target_date).strftime('%Y-%m-%d')
        redis_key = f"qlib:signals:{STRATEGY_NAME}:{today_str}"
        redis_client.sadd(redis_key, "_COMPLETED_")
        redis_client.expire(redis_key, 86,400 * 3) # 三天
        print(f"⚠️ 目标日期 {today_str} 无买入信号，已写入完成标记。")
        return

    # 2. 确保 datetime 列是时间格式，方便后续按天分组
    buy_records = buy_records.copy()
    buy_records['datetime'] = pd.to_datetime(buy_records['datetime']) # 冗余操作
    
    # 3. 按日期分组，循环写入 Redis
    # 将 Timestamp 转换为 YYYY-MM-DD 字符串
    grouped = buy_records.groupby(buy_records['datetime'].dt.strftime('%Y-%m-%d')) # .dt.strftime('%Y-%m-%d'))冗余操作
    
    for date_str, group in grouped:
        redis_key = f"qlib:signals:{STRATEGY_NAME}:{date_str}"
        
        # 获取该日期下所有要买入的币种
        # 注意：这里 /USDT:USDT 是 U本位合约写法。如果是现货，请改为 f"{inst}/USDT"
        buy_symbols = [f"{inst}/USDT:USDT" for inst in group['instrument']]
        
        if buy_symbols:
            # 批量写入买入币种
            redis_client.sadd(redis_key, *buy_symbols)
        
        # ⚠️ 关键动作：打上当日计算完成的标记
        redis_client.sadd(redis_key, "_COMPLETED_")
        
        # 设置过期时间：3天 (3 * 24 * 60 * 60 = 259200秒)
        # 防止历史信号一直堆积在 Redis 占用内存
        redis_client.expire(redis_key, 86400 * 3)
        
    print(f"✅ 成功将 {len(grouped)} 个交易日的信号存入 Redis (过期时间: 3天)。")


if __name__ == "__main__":
    start_time = time.time()
    print("初始化 Qlib...")
    qlib.init(provider_uri=PROVIDER_URI, region=REG_US, limit_threshold=None)
    
    # 初始化 Redis 连接
    try:
        redis_client = redis.Redis(host='127.0.0.1', port=6379, db=0, decode_responses=True)
        # 测试一下连接是否通畅
        redis_client.ping()
        print(f"✅ Redis 连接成功！")
    except Exception as e:
        print(f"❌ Redis 连接失败，请检查服务是否开启: {e}")
        exit(1)

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

    # ==========================================
    # 保存结果 (CSV + Redis)
    # ==========================================
    # 1. 写入 Redis (将 redis_client 和 TARGET_DATE 传进去)
    save_to_redis(redis_client, buy_records, TARGET_DATE)

    # 2. 存入 CSV (保留此步可用于回测或问题排查)
    output_path = "qlib_golden_signals.csv"
    buy_records.to_csv(output_path, index=False)

    print(f"✅ 信号生成完毕！日期: {TARGET_DATE}")
    print(f"📂 CSV 已保存至: {output_path}")
    print(f"📈 共生成了 {len(buy_records)} 条买入指令。")
    end_time = time.time()
    print(f"共用时： {(end_time - start_time):.0f}秒")
