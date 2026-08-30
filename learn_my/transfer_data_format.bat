@echo off
chcp 65001

python scripts/dump_bin.py dump_all ^
    --data_path C:/Users/WANGKANG/Desktop/qlib/learn_my/stock_data ^
    --qlib_dir ~/.qlib/qlib_data/my_tushare_data ^
    @REM --symbol_field_name ts_code ^
    --date_field_name date ^
    --include_fields open,close,high,low,volume,amount,vwap,circ_mv,total_mv,factor,industry,is_ST,list_status
