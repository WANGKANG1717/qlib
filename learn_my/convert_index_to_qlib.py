"""
将tushare获取的指数成分股权重数据转换为qlib格式的instruments文件

输入格式 (tushare index_weight):
index_code,con_code,trade_date,weight
000016.SH,600036.SH,20090430,7.15

输出格式 (qlib instruments):
SH600036	2009-04-30	2099-12-31
"""
import os
from pathlib import Path
from common_utils import convert_code

import pandas as pd

def convert_index_weight_to_qlib_format_(input_csv, output_txt):
    """
    将tushare指数权重CSV转换为qlib instruments格式
    
    Args:
        input_csv: tushare指数权重CSV文件路径
        output_txt: 输出的qlib instruments txt文件路径
    """
    df = pd.read_csv(input_csv)
    df['qlib_code'] = df['con_code'].apply(convert_code)

    # 转换日期格式: 20090430 -> 2009-04-30
    df['date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')

    MIN_DATE = '2000-01-01'
    MAX_DATE = '2099-12-31'

    cur_min_date = df['date'].min()
    cur_max_date = df['date'].max()

    df.loc[df['date'] == cur_min_date, 'date'] = MIN_DATE
    df.loc[df['date'] == cur_max_date, 'date'] = MAX_DATE

    stock_groups = df.groupby('qlib_code')
    results = []
    for qlib_code, group in stock_groups:
        start_date = group['date'].min()
        end_date = group['date'].max()
        
        results.append({
            'code': qlib_code,
            'start': start_date,
            'end': end_date
        })

    result_df = pd.DataFrame(results)

    # 按股票代码排序
    result_df = result_df.sort_values('code')
    
    # 写入txt文件 (制表符分隔)
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    with open(output_txt, 'w', encoding='utf-8') as f:
        for _, row in result_df.iterrows():
            f.write(f"{row['code']}\t{row['start']}\t{row['end']}\n")
    
    print(f"转换完成! 共 {len(result_df)} 只股票")
    print(f"输出文件: {output_txt}")


def convert_index_weight_to_qlib_format(index_weight_dir, qlib_instruments_dir):
    """
    批量转换所有指数权重文件
    
    Args:
        index_weight_dir: tushare指数权重CSV目录
        qlib_instruments_dir: qlib instruments输出目录
    """
    # 指数代码映射
    index_mapping = {
        '000016.SH': 'sse50',      # 上证50
        '000300.SH': 'csi300',     # 沪深300
        '000905.SH': 'csi500',     # 中证500
        '000906.SH': 'csi800',     # 中证800
        '000852.SH': 'csi1000',    # 中证1000
        '000688.SH': 'star50',     # 科创50
        "399006.SZ": "chinext",    # 创业板指
    }
    
    os.makedirs(qlib_instruments_dir, exist_ok=True)
    
    for csv_file in Path(index_weight_dir).glob('*.csv'):
        index_code = csv_file.stem  # 例如: 000016.SH
        
        if index_code in index_mapping:
            qlib_name = index_mapping[index_code]
            output_txt = os.path.join(qlib_instruments_dir, f'{qlib_name}.txt')
            
            print(f"\n处理: {index_code} -> {qlib_name}.txt")
            try:
                convert_index_weight_to_qlib_format_(str(csv_file), output_txt)
            except Exception as e:
                print(f"错误: {e}")
        else:
            print(f"跳过未知指数: {index_code}")


def convert_market_to_qlib_format(stock_basic_csv_path, qlib_all_market_path, qlib_instruments_dir):
    """
    将股票基本信息按市场板块转换为qlib instruments格式
    
    Args:
        stock_basic_csv_path: stock_basic.csv文件路径
        qlib_all_market_path: all.txt文件路径
        qlib_instruments_dir: qlib instruments输出目录
    """
    os.makedirs(qlib_instruments_dir, exist_ok=True)

    # 市场板块映射
    market_mapping = {
        '主板': 'zb',      # zhu ban
        '创业板': 'cyb',   # chuang ye ban
        '科创板': 'kcb',   # ke chuang ban
        '北交所': 'bjs',   # bei jiao suo
    }

    df_market = pd.read_csv(stock_basic_csv_path, usecols=["ts_code", "market"])
    df_market['instrument'] = df_market['ts_code'].apply(convert_code)
    df_market = df_market.drop(columns=['ts_code'])

    df_all = pd.read_csv(qlib_all_market_path, sep='\t',header=None, names=['instrument', 'start_date', 'end_date'])
    df_all_market = df_all.merge(df_market, on='instrument', how='left').dropna()
    
    
    for market_name, file_prefix in market_mapping.items():
        # 筛选特定市场的股票
        df_filtered = df_all_market[df_all_market['market'] == market_name].copy()
        
        if df_filtered.empty:
            print(f"跳过: {market_name} (无数据)")
            continue
        
        # 写入txt文件
        output_txt = os.path.join(qlib_instruments_dir, f'{file_prefix}.txt')
        with open(output_txt, 'w', encoding='utf-8') as f:
            for _, row in df_filtered.iterrows():
                f.write(f"{row['instrument']}\t{row['start_date']}\t{row['end_date']}\n")
        
        print(f"转换完成: {market_name} -> {file_prefix}.txt (共 {len(df_filtered)} 只股票)")

if __name__ == '__main__':
    # 配置路径
    INDEX_WEIGHT_DIR = r"C:\Users\WANGKANG\Desktop\量化交易学习\stock_data\index_weight"
    QLIB_INSTRUMENTS_DIR = r"C:\Users\WANGKANG\.qlib\qlib_data\my_tushare_data\instruments"

    STOCK_BASIC_PATH =  r'C:\Users\WANGKANG\Desktop\量化交易学习\stock_data\stock_basic.csv'
    QLIB_ALL_MARKET_PATH = r"C:\Users\WANGKANG\.qlib\qlib_data\my_tushare_data\instruments\all.txt"
    
    # 批量转换所有指数
    convert_index_weight_to_qlib_format(INDEX_WEIGHT_DIR, QLIB_INSTRUMENTS_DIR)
    # 批量转换板块
    convert_market_to_qlib_format(STOCK_BASIC_PATH, QLIB_ALL_MARKET_PATH, QLIB_INSTRUMENTS_DIR)
