import json
import os
from typing import Optional

import fire
import pandas as pd
from loguru import logger
from tqdm import tqdm

import qlib
from qlib.data import D


class DataHealthChecker:
    """Checks a dataset for data completeness and correctness. The data will be converted to a pd.DataFrame and checked for the following problems:
    - any of the columns ["open", "high", "low", "close", "volume"] are missing
    - any data is missing
    - any step change in the OHLCV columns is above a threshold (default: 0.5 for price, 3 for volume)
    - any factor is missing
    """

    def __init__(
        self,
        csv_path=None,
        qlib_dir=None,
        freq="day",
        large_step_threshold_price=0.5,
        large_step_threshold_volume=3,
        missing_data_num=0,
        report_dir=None,
    ):
        assert csv_path or qlib_dir, "One of csv_path or qlib_dir should be provided."
        assert not (csv_path and qlib_dir), "Only one of csv_path or qlib_dir should be provided."

        self.data = {}
        self.problems = {}
        self.freq = freq
        self.large_step_threshold_price = large_step_threshold_price
        self.large_step_threshold_volume = large_step_threshold_volume
        self.missing_data_num = missing_data_num
        self.qlib_dir = os.path.abspath(os.path.expanduser(qlib_dir)) if qlib_dir else None
        self.report_dir = os.path.abspath(os.path.expanduser(report_dir)) if report_dir else None

        if csv_path:
            assert os.path.isdir(csv_path), f"{csv_path} should be a directory."
            files = [f for f in os.listdir(csv_path) if f.endswith(".csv")]
            for filename in tqdm(files, desc="Loading data"):
                df = pd.read_csv(os.path.join(csv_path, filename))
                if "date" in df.columns:
                    df = df.sort_values("date").reset_index(drop=True)
                self.data[filename] = df

        elif qlib_dir:
            qlib.init(provider_uri=qlib_dir)
            self.load_qlib_data()

    def load_qlib_data(self):
        instruments = D.instruments(market="all")
        instrument_list = D.list_instruments(instruments=instruments, as_list=True, freq=self.freq)
        required_fields = ["$open", "$close", "$low", "$high", "$volume", "$factor"]
        for instrument in instrument_list:
            df = D.features([instrument], required_fields, freq=self.freq)
            df.rename(
                columns={
                    "$open": "open",
                    "$close": "close",
                    "$low": "low",
                    "$high": "high",
                    "$volume": "volume",
                    "$factor": "factor",
                },
                inplace=True,
            )
            self.data[instrument] = df
        logger.info(f"Loaded {len(self.data)} instruments from Qlib.")

    # NOTE:
    # This check is added due to a known issue in Qlib where feature paths
    # are constructed using lowercased instrument names. On case-sensitive
    # file systems (e.g. Linux), uppercase directory names under `features/`
    # will cause data loading failures.
    #
    # See: https://github.com/microsoft/qlib/issues/2053
    def check_features_dir_lowercase(self) -> Optional[pd.DataFrame]:
        """
        Check whether all subdirectories under `<qlib_dir>/features` are named in lowercase.

        This validation helps prevent data loading issues on case-sensitive
        file systems caused by uppercase instrument directory names.
        """
        if not self.qlib_dir:
            return None

        features_dir = os.path.join(self.qlib_dir, "features")
        if not os.path.isdir(features_dir):
            logger.warning(f"`features` directory not found under {self.qlib_dir}")
            return None

        bad_dirs = []
        for name in os.listdir(features_dir):
            full_path = os.path.join(features_dir, name)
            if os.path.isdir(full_path) and name != name.lower():
                bad_dirs.append(name)

        if bad_dirs:
            result_df = pd.DataFrame({"non_lowercase_dir": bad_dirs})
            return result_df
        else:
            logger.info(
                f"✅ All subdirectories under `{os.path.join(self.qlib_dir, 'features')}` are named in lowercase."
            )
            return None

    def check_missing_data(self) -> Optional[pd.DataFrame]:
        """Check missing values in required OHLCV fields only.

        Optional valuation/fundamental fields are allowed to be null and must not
        make an otherwise usable market-data file fail this check.
        """
        result_dict = {
            "instruments": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        }
        for filename, df in self.data.items():
            missing_counts = {
                col: int(df[col].isnull().sum()) if col in df.columns else len(df)
                for col in ["open", "high", "low", "close", "volume"]
            }
            if any(count > self.missing_data_num for count in missing_counts.values()):
                result_dict["instruments"].append(filename)
                for col, count in missing_counts.items():
                    result_dict[col].append(count)

        result_df = pd.DataFrame(result_dict).set_index("instruments")
        if not result_df.empty:
            return result_df
        else:
            logger.info(f"✅ There are no missing data.")
            return None

    def check_large_step_changes(self) -> Optional[pd.DataFrame]:
        """Check if there are any large step changes above the threshold in the OHLCV columns."""
        result_dict = {
            "instruments": [],
            "col_name": [],
            "date": [],
            "pct_change": [],
        }
        for filename, df in self.data.items():
            affected_columns = []
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    pct_change = df[col].pct_change(fill_method=None).abs()
                    threshold = self.large_step_threshold_volume if col == "volume" else self.large_step_threshold_price
                    if pct_change.max() > threshold:
                        max_position = int(pct_change.fillna(float("-inf")).to_numpy().argmax())
                        if "date" in df.columns:
                            change_date = pd.to_datetime(df.iloc[max_position]["date"])
                        else:
                            index_value = df.index[max_position]
                            change_date = pd.to_datetime(index_value[-1] if isinstance(index_value, tuple) else index_value)
                        result_dict["instruments"].append(filename)
                        result_dict["col_name"].append(col)
                        result_dict["date"].append(change_date.strftime("%Y-%m-%d"))
                        result_dict["pct_change"].append(float(pct_change.iloc[max_position]))
                        affected_columns.append(col)

        result_df = pd.DataFrame(result_dict).set_index("instruments")
        if not result_df.empty:
            return result_df
        else:
            logger.info(f"✅ There are no large step changes in the OHLCV column above the threshold.")
            return None

    def check_required_columns(self) -> Optional[pd.DataFrame]:
        """Check if any of the required columns (OLHCV) are missing in the DataFrame."""
        required_columns = ["open", "high", "low", "close", "volume"]
        result_dict = {
            "instruments": [],
            "missing_col": [],
        }
        for filename, df in self.data.items():
            if not all(column in df.columns for column in required_columns):
                missing_required_columns = [column for column in required_columns if column not in df.columns]
                result_dict["instruments"].append(filename)
                result_dict["missing_col"].append(",".join(missing_required_columns))

        result_df = pd.DataFrame(result_dict).set_index("instruments")
        if not result_df.empty:
            return result_df
        else:
            logger.info(f"✅ The columns (OLHCV) are complete and not missing.")
            return None

    def check_ohlc_consistency(self) -> Optional[pd.DataFrame]:
        """Check low <= open/close <= high and low <= high."""
        records = []
        required = {"open", "high", "low", "close"}
        for filename, df in self.data.items():
            if not required.issubset(df.columns):
                continue
            invalid = df[
                (df["high"] < df[["open", "close", "low"]].max(axis=1))
                | (df["low"] > df[["open", "close", "high"]].min(axis=1))
            ]
            for position in df.index.get_indexer(invalid.index):
                row = df.iloc[position]
                if "date" in df.columns:
                    date = pd.to_datetime(row["date"])
                else:
                    index_value = df.index[position]
                    date = pd.to_datetime(index_value[-1] if isinstance(index_value, tuple) else index_value)
                records.append(
                    {
                        "instruments": filename,
                        "date": date.strftime("%Y-%m-%d"),
                        "open": row["open"],
                        "high": row["high"],
                        "low": row["low"],
                        "close": row["close"],
                    }
                )
        result_df = pd.DataFrame(records)
        if not result_df.empty:
            return result_df.set_index("instruments")
        logger.info("✅ All OHLC rows satisfy low <= open/close <= high.")
        return None

    def check_negative_values(self) -> Optional[pd.DataFrame]:
        """Check that volume and amount/money are non-negative."""
        records = []
        for filename, df in self.data.items():
            counts = {}
            for col in ["volume", "amount", "money"]:
                if col in df.columns:
                    count = int((df[col].dropna() < 0).sum())
                    if count:
                        counts[f"negative_{col}_count"] = count
            if counts:
                records.append({"instruments": filename, **counts})
        result_df = pd.DataFrame(records)
        if not result_df.empty:
            return result_df.set_index("instruments").fillna(0).astype(int)
        logger.info("✅ Volume and amount/money contain no negative values.")
        return None

    def check_duplicate_dates(self) -> Optional[pd.DataFrame]:
        """Check duplicate dates within each instrument."""
        records = []
        for filename, df in self.data.items():
            dates = df["date"] if "date" in df.columns else df.index.get_level_values(-1)
            duplicate_count = int(pd.Index(dates).duplicated().sum())
            if duplicate_count:
                records.append({"instruments": filename, "duplicate_date_count": duplicate_count})
        result_df = pd.DataFrame(records)
        if not result_df.empty:
            return result_df.set_index("instruments")
        logger.info("✅ There are no duplicate dates within instruments.")
        return None

    def check_missing_factor(self) -> Optional[pd.DataFrame]:
        """Check whether factor exists and contains missing/non-positive values."""
        result_dict = {
            "instruments": [],
            "missing_factor_col": [],
            "missing_factor_count": [],
            "nonpositive_factor_count": [],
        }
        for filename, df in self.data.items():
            if "factor" not in df.columns:
                result_dict["instruments"].append(filename)
                result_dict["missing_factor_col"].append(True)
                result_dict["missing_factor_count"].append(len(df))
                result_dict["nonpositive_factor_count"].append(0)
                continue

            market_columns = [col for col in ["open", "high", "low", "close", "volume"] if col in df.columns]
            # Qlib aligns instruments to the shared calendar. Suspension/non-trading
            # rows are all-NaN by design, so factor is required only on rows that
            # contain at least one market-data value.
            active_rows = df[market_columns].notna().any(axis=1) if market_columns else pd.Series(True, index=df.index)
            missing_count = int((active_rows & df["factor"].isnull()).sum())
            nonpositive_count = int((df["factor"].dropna() <= 0).sum())
            if missing_count or nonpositive_count:
                result_dict["instruments"].append(filename)
                result_dict["missing_factor_col"].append(False)
                result_dict["missing_factor_count"].append(missing_count)
                result_dict["nonpositive_factor_count"].append(nonpositive_count)

        result_df = pd.DataFrame(result_dict).set_index("instruments")
        if not result_df.empty:
            return result_df
        else:
            logger.info(f"✅ The `factor` column already exists and is not empty.")
            return None

    def _save_reports(self, results):
        if not self.report_dir:
            return
        os.makedirs(self.report_dir, exist_ok=True)
        summary = {"files_checked": len(self.data), "checks": {}}
        for name, result in results.items():
            issue_count = 0 if result is None else len(result)
            summary["checks"][name] = {"issue_count": issue_count}
            report_path = os.path.join(self.report_dir, f"{name}.csv")
            if isinstance(result, pd.DataFrame) and not result.empty:
                result.to_csv(report_path)
            elif os.path.exists(report_path):
                os.remove(report_path)
        with open(os.path.join(self.report_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"Health-check reports saved to {self.report_dir}")

    def check_data(self):
        check_missing_data_result = self.check_missing_data()
        check_large_step_changes_result = self.check_large_step_changes()
        check_required_columns_result = self.check_required_columns()
        check_missing_factor_result = self.check_missing_factor()
        check_ohlc_consistency_result = self.check_ohlc_consistency()
        check_negative_values_result = self.check_negative_values()
        check_duplicate_dates_result = self.check_duplicate_dates()
        check_features_dir_case_result = self.check_features_dir_lowercase()
        results = {
            "missing_data": check_missing_data_result,
            "large_step_changes": check_large_step_changes_result,
            "required_columns": check_required_columns_result,
            "missing_factor": check_missing_factor_result,
            "ohlc_consistency": check_ohlc_consistency_result,
            "negative_values": check_negative_values_result,
            "duplicate_dates": check_duplicate_dates_result,
            "features_dir_case": check_features_dir_case_result,
        }
        self._save_reports(results)
        if (
            check_missing_data_result is not None
            or check_large_step_changes_result is not None
            or check_required_columns_result is not None
            or check_missing_factor_result is not None
            or check_ohlc_consistency_result is not None
            or check_negative_values_result is not None
            or check_duplicate_dates_result is not None
            or check_features_dir_case_result is not None
        ):
            print(f"\nSummary of data health check ({len(self.data)} files checked):")
            print("-------------------------------------------------")
            if isinstance(check_missing_data_result, pd.DataFrame):
                logger.warning(f"There is missing data.")
                print(check_missing_data_result)
            if isinstance(check_large_step_changes_result, pd.DataFrame):
                logger.warning(f"The OHLCV column has large step changes.")
                print(check_large_step_changes_result)
            if isinstance(check_required_columns_result, pd.DataFrame):
                logger.warning(f"Columns (OLHCV) are missing.")
                print(check_required_columns_result)
            if isinstance(check_missing_factor_result, pd.DataFrame):
                logger.warning("The factor column is missing or has missing/non-positive values on active rows.")
                print(check_missing_factor_result)
            if isinstance(check_ohlc_consistency_result, pd.DataFrame):
                logger.warning("Some rows violate OHLC consistency constraints.")
                print(check_ohlc_consistency_result)
            if isinstance(check_negative_values_result, pd.DataFrame):
                logger.warning("Volume or amount/money contains negative values.")
                print(check_negative_values_result)
            if isinstance(check_duplicate_dates_result, pd.DataFrame):
                logger.warning("Some instruments contain duplicate dates.")
                print(check_duplicate_dates_result)
            if isinstance(check_features_dir_case_result, pd.DataFrame):
                logger.warning(
                    f"Some subdirectories under `{os.path.join(self.qlib_dir, 'features')}` contain uppercase letters, please rename them to lowercase manually."
                )
                print(check_features_dir_case_result)


if __name__ == "__main__":
    fire.Fire(DataHealthChecker)
