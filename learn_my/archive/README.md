# 🗄️ Archive (历史代码归档)

本目录用于存放已被新方案取代、不再参与实际运行的历史脚本，仅作参考与追溯，请勿在生产流程中调用。

## 📁 归档文件说明

| 文件名 | 历史功能与定位 | 废弃原因 |
| :-- | :-- | :-- |
| `transfer_data.py` | 早期数据转换脚本，通过多进程 `Pool` 逐个读取本地 hfq / daily_basic / adj_factor 目录下的 CSV，拼接后生成 Qlib 所需的单标的 CSV。 | 依赖硬编码的本地绝对路径，且逐文件读取效率低。已由 `convert_to_qlib.py`（基于 DuckDB 扫描 Parquet + 分区导出）整体替代。 |
| `transfer_data_format.bat` | 配合 `transfer_data.py` 的批处理脚本，手动调用 `scripts/dump_bin.py` 将 CSV 编译为 Qlib `.bin` 二进制矩阵。 | 参数与路径均为手写硬编码，易出错。`dump_bin` 调用已内聚到 `convert_to_qlib.py` 的 `run_qlib_dump_bin()` 中自动执行。 |
| `convert_index_to_qlib.py` | 将 Tushare `index_weight` 指数成分权重 CSV 转换为 Qlib instruments 文件（`code  start  end` 格式）。 | 基于逐日 CSV 处理的旧实现。已由 `convert_to_qlib.py` 中的状态机算法 `convert_index_weight_to_qlib_format()` 替代，直接读取 Parquet 并按截面比对生成区间。 |

## 🔄 现行替代方案

- **数据下载**：`learn_my/data_download.py`
- **格式转换 + dump_bin**：`learn_my/convert_to_qlib.py`
