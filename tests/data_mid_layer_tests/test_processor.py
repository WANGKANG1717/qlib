# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
import warnings

import numpy as np
import pandas as pd
from qlib.data import D
from qlib.tests import TestAutoData
from qlib.data.dataset.processor import CSNeutralize, CSZFillna, CSZScoreNorm, MinMaxNorm, ZScoreNorm


class TestCSNeutralize(unittest.TestCase):
    def test_preserves_column_groups_and_missing_values(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        instruments = [f"S{i}" for i in range(6)]
        index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
        market_value = np.tile([10, 20, 30, 40, 50, 60], 2).astype(float)
        industry = np.tile([1, 1, 2, 2, 3, 3], 2).astype(float)
        log_market_value = np.log(market_value)

        feature = pd.DataFrame(
            {
                "factor_1": 2 * log_market_value + industry,
                "factor_2": -log_market_value + 0.5 * industry,
            },
            index=index,
        )
        feature.loc[(dates[0], "S0"), "factor_2"] = np.nan
        raw = pd.DataFrame({"$total_mv": market_value, "$industry": industry}, index=index)
        raw.loc[(dates[1], "S5"), "$industry"] = np.nan
        label = pd.DataFrame({"LABEL0": np.arange(len(index), dtype=float)}, index=index)
        data = pd.concat({"feature": feature, "label": label, "raw": raw}, axis=1)

        result = CSNeutralize(min_stocks=3)(data)

        pd.testing.assert_index_equal(result.index, data.index)
        pd.testing.assert_index_equal(result.columns, data.columns)
        pd.testing.assert_frame_equal(result["label"], data["label"])
        pd.testing.assert_frame_equal(result["raw"], data["raw"])
        self.assertTrue(np.nanmax(np.abs(result[("feature", "factor_1")].to_numpy())) < 1e-10)
        self.assertTrue(pd.isna(result.loc[(dates[0], "S0"), ("feature", "factor_2")]))
        self.assertTrue(result.loc[(dates[1], "S5"), "feature"].isna().all())

    def test_preserves_float32_feature_dtype_without_future_warning(self):
        date = pd.Timestamp("2024-01-02")
        instruments = [f"S{i}" for i in range(4)]
        index = pd.MultiIndex.from_product([[date], instruments], names=["datetime", "instrument"])
        feature = pd.DataFrame(
            {
                "factor_1": np.array([1, 2, 4, 8], dtype=np.float32),
                "factor_2": np.array([8, 4, 2, 1], dtype=np.float32),
            },
            index=index,
        )
        raw = pd.DataFrame(
            {
                "$total_mv": np.array([10, 20, 30, 40], dtype=np.float32),
                "$industry": np.array([1, 1, 2, 2], dtype=np.float32),
            },
            index=index,
        )
        data = pd.concat({"feature": feature, "raw": raw}, axis=1)

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            result = CSNeutralize(min_stocks=3)(data)

        pd.testing.assert_series_equal(result["feature"].dtypes, feature.dtypes)


class TestProcessor(TestAutoData):
    TEST_INST = "SH600519"

    def test_MinMaxNorm(self):
        def normalize(df):
            min_val = np.nanmin(df.values, axis=0)
            max_val = np.nanmax(df.values, axis=0)
            ignore = min_val == max_val
            for _i, _con in enumerate(ignore):
                if _con:
                    max_val[_i] = 1
                    min_val[_i] = 0
            df.loc(axis=1)[df.columns] = (df.values - min_val) / (max_val - min_val)
            return df

        origin_df = D.features([self.TEST_INST], ["$high", "$open", "$low", "$close"]).tail(10)
        origin_df["test"] = 0
        df = origin_df.copy()
        mmn = MinMaxNorm(fields_group=None, fit_start_time="2021-05-31", fit_end_time="2021-06-11")
        mmn.fit(df)
        mmn.__call__(df)
        origin_df = normalize(origin_df)
        assert (df == origin_df).all().all()

    def test_ZScoreNorm(self):
        def normalize(df):
            mean_train = np.nanmean(df.values, axis=0)
            std_train = np.nanstd(df.values, axis=0)
            ignore = std_train == 0
            for _i, _con in enumerate(ignore):
                if _con:
                    std_train[_i] = 1
                    mean_train[_i] = 0
            df.loc(axis=1)[df.columns] = (df.values - mean_train) / std_train
            return df

        origin_df = D.features([self.TEST_INST], ["$high", "$open", "$low", "$close"]).tail(10)
        origin_df["test"] = 0
        df = origin_df.copy()
        zsn = ZScoreNorm(fields_group=None, fit_start_time="2021-05-31", fit_end_time="2021-06-11")
        zsn.fit(df)
        zsn.__call__(df)
        origin_df = normalize(origin_df)
        assert (df == origin_df).all().all()

    def test_CSZFillna(self):
        origin_df = D.features(D.instruments(market="csi300"), fields=["$high", "$open", "$low", "$close"])
        origin_df = origin_df.groupby("datetime", group_keys=False).apply(lambda x: x[97:99])[228:238]
        df = origin_df.copy()
        CSZFillna(fields_group=None).__call__(df)
        assert ~df[1:2].isna().all().all() and origin_df[1:2].isna().all().all()

    def test_CSZScoreNorm(self):
        origin_df = D.features(D.instruments(market="csi300"), fields=["$high", "$open", "$low", "$close"])
        origin_df = origin_df.groupby("datetime", group_keys=False).apply(lambda x: x[10:12])[50:60]
        df = origin_df.copy()
        CSZScoreNorm(fields_group=None).__call__(df)
        # If we use the formula directly on the original data, we cannot get the correct result,
        # because the original data is processed by `groupby`, so we use the method of slicing,
        # taking the 2nd group of data from the original data, to calculate and compare.
        assert (df[2:4] == ((origin_df[2:4] - origin_df[2:4].mean()).div(origin_df[2:4].std()))).all().all()


if __name__ == "__main__":
    unittest.main()
