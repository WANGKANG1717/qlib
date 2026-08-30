
""" 
将格式转为 qlib 所需的格式
"""
import os
from multiprocessing import Pool
from common_utils import load_industry_code, convert_code, get_stock_basic_info
import json

import pandas as pd
from tqdm import tqdm

output_dir = r"C:\Users\WANGKANG\Desktop\qlib\learn_my\stock_data"
data_dir = r'C:\Users\WANGKANG\Desktop\量化交易学习\stock_data\hfq'
daily_basic_dir = r"C:\Users\WANGKANG\Desktop\量化交易学习\stock_data\daily_basic"
adj_factor_dir = r"C:\Users\WANGKANG\Desktop\量化交易学习\stock_data\adj_factor"
name_to_code_path = r"C:\Users\WANGKANG\Desktop\qlib\learn_my\stock_data\name_to_code.json"

os.makedirs(output_dir, exist_ok=True)

industry_code, name_to_code = load_industry_code()
with open(name_to_code_path, "w") as f:
    json.dump(name_to_code, f, ensure_ascii=False, indent=4)

stock_basic = get_stock_basic_info()

def process_single_file(file):
    qlib_code = convert_code(file)  # 转换为 qlib 格式的代码
    
    try:
        # 1. 读取 K线数据和基本面数据
        data_path = os.path.join(data_dir, file)
        df = pd.read_csv(data_path, usecols=['trade_date', 'open', 'close', 'high', 'low', 'vol', 'amount'])
        
        daily_basic_path = os.path.join(daily_basic_dir, file)
        df_daily_basic = pd.read_csv(daily_basic_path, usecols=['trade_date', 'circ_mv', 'total_mv'])

        adj_factor_path = os.path.join(adj_factor_dir, file)
        df_factor = pd.read_csv(adj_factor_path, usecols=['trade_date', 'adj_factor'])  # 读取复权因子数据
        df_factor = df_factor.rename(columns={'adj_factor': 'factor'}) # 将adj_factor重命名为factor
        
        
        # 2. 统一转换日期格式
        df['date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
        df_daily_basic['date'] = pd.to_datetime(df_daily_basic['trade_date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
        df_factor['date'] = pd.to_datetime(df_factor['trade_date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
        
        # 3. 重命名列
        df = df.rename(columns={'vol': 'volume'})
        
        # 4. 合并市值数据（只保留 date 和市值列）
        df = df[['date', 'open', 'close', 'high', 'low', 'volume', 'amount']].merge(
            df_daily_basic[['date', 'circ_mv', 'total_mv']],
            on='date',
            how='left'
        ).merge(
            df_factor[['date', 'factor']],
            on='date',
            how='left'
        )
        df['vwap'] = (df['amount'] * df["factor"] / df['volume'] * 10).round(4) # 计算 VWAP（成交量加权平均价）

        # 4.1 设置行业
        assert industry_code.get(qlib_code) != None, f'不存在{qlib_code}的行业！'
        df['industry'] = industry_code[qlib_code]

        # 4.2 设置是否是ST：0不是 1是
        df['is_ST'] = stock_basic.loc[qlib_code]['is_ST']
        # 4.3 设置股票状态：L/0=上市, D/1=退市, P/2=暂停上市
        df['list_status'] = stock_basic.loc[qlib_code]['list_status']
        
        # 5. 排序并重置索引（按日期升序）
        df = df.sort_values('date').reset_index(drop=True)
        
        # 6. 检查数据有效性
        if df.empty:
            print(f"⚠ 警告: {file} 转换后数据为空")
        
        # 7. 保存为 CSV
        csv_path = os.path.join(output_dir, f"{qlib_code}.csv")
        df.to_csv(csv_path, index=False)
        
    except FileNotFoundError:
        print(f"✗ 文件缺失: {file} (可能 daily_basic 目录没有对应文件)")
    except ValueError as e:
        print(f"✗ 数据格式错误: {file} - {str(e)}")
    except Exception as e:
        print(f"✗ 处理失败: {file} - {str(e)}")

if __name__ == '__main__':
    # 获取所有CSV文件
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    # 使用多进程池并行处理
    with Pool(processes=10) as pool:
        list(tqdm(pool.imap(process_single_file, files), total=len(files)))

    print("\n数据转换完成！下一步运行 dump_bin 命令行工具转为 qlib 格式数据")
    print(r"运行.\learn_my\transfer_data_format.bat")


""" 
# 验证有没有添加成功
import qlib
from qlib.data import D

# 1. 初始化
qlib.init(
    provider_uri='~/.qlib/qlib_data/my_tushare_data', 
    region=qlib.config.REG_CN
)

# 2. 设定查询参数
code = "sz000001"           # 股票代码 (注意带上前缀，和你的 csv 命名保持一致)
start_date = "2000-01-01"   # 你想要查询的具体日期
end_date = "2010-01-01"     # 你想要查询的具体日期
fields = ['$open', '$high', '$low', '$close', '$volume', '$amount', '$vwap', '$circ_mv', '$total_mv', '$factor', '$industry', '$is_ST', '$list_status'] # 基础 K 线字段 (记得带 $)

# 3. 使用 D.features 获取数据
df = D.features(
    instruments=[code], 
    fields=fields, 
    start_time=start_date, 
    end_time=end_date    # start 和 end 设为同一天，就是查这一天的数据
)

# 4. 打印查看结果
print(df)
"""
