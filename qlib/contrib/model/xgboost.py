# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Text, Union
from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...model.interpret.base import FeatureInt
from ...data.dataset.weight import Reweighter


class XGBModel(Model, FeatureInt):
    """XGBModel Model"""

    def __init__(self, **kwargs):
        self._params = {}
        self._params.update(kwargs)
        self.model = None

    def fit(
        self,
        dataset: DatasetH,
        num_boost_round=1000,
        early_stopping_rounds=50,
        verbose_eval=20,
        evals_result=dict(),
        reweighter=None,
        **kwargs,
    ):
        if evals_result is None:
            evals_result = {}
            
        evals = []
        dtrain = None
        
        # 确保 train segment 存在
        assert "train" in dataset.segments, "The 'train' segment is required."

        # 动态处理 train 和 optional 的 valid 数据集
        for key in ["train", "valid"]:
            if key in dataset.segments:
                df = dataset.prepare(
                    key,
                    col_set=["feature", "label"],
                    data_key=DataHandlerLP.DK_L,
                )
                if df.empty:
                    raise ValueError(f"Empty data from dataset segment '{key}', please check your dataset config.")

                x, y = df["feature"], df["label"]

                # XGBoost 需要 1D array 作为 label
                if y.values.ndim == 2 and y.values.shape[1] == 1:
                    y_1d = np.squeeze(y.values)
                else:
                    raise ValueError("XGBoost doesn't support multi-label training")

                if reweighter is None:
                    w = None
                elif isinstance(reweighter, Reweighter):
                    w = reweighter.reweight(df)
                else:
                    raise ValueError("Unsupported reweighter type.")

                dmatrix = xgb.DMatrix(x.values, label=y_1d, weight=w)
                evals.append((dmatrix, key))
                
                if key == "train":
                    dtrain = dmatrix

        self.model = xgb.train(
            self._params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
            evals_result=evals_result,
            **kwargs,
        )
        
        # 仅对存在于 evals_result 中的 key 进行后处理
        for key in ["train", "valid"]:
            if key in evals_result:
                evals_result[key] = list(evals_result[key].values())[0]

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        return pd.Series(self.model.predict(xgb.DMatrix(x_test)), index=x_test.index)

    def get_feature_importance(self, *args, **kwargs) -> pd.Series:
        """get feature importance

        Notes
        -------
            parameters reference:
                https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.get_score
        """
        return pd.Series(self.model.get_score(*args, **kwargs)).sort_values(ascending=False)
