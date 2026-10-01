# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import List, Text, Tuple, Union
from ...model.base import ModelFT
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...model.interpret.base import LightGBMFInt
from ...data.dataset.weight import Reweighter
from qlib.workflow import R
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class LGBModel(ModelFT, LightGBMFInt):
    """LightGBM Model"""

    def __init__(self, loss="mse", early_stopping_rounds=50, num_boost_round=1000, **kwargs):
        if loss not in {"mse", "binary"}:
            raise NotImplementedError
        self.params = {"objective": loss, "verbosity": -1}
        self.params.update(kwargs)
        self.early_stopping_rounds = early_stopping_rounds
        self.num_boost_round = num_boost_round
        self.model = None

    def _prepare_data(self, dataset: DatasetH, reweighter=None) -> List[Tuple[lgb.Dataset, str]]:
        """
        The motivation of current version is to make validation optional
        - train segment is necessary;
        """
        ds_l = []
        assert "train" in dataset.segments
        for key in ["train", "valid"]:
            if key in dataset.segments:
                df = dataset.prepare(key, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
                if df.empty:
                    raise ValueError("Empty data from dataset, please check your dataset config.")
                x, y = df["feature"], df["label"]

                # Lightgbm need 1D array as its label
                if y.values.ndim == 2 and y.values.shape[1] == 1:
                    y = np.squeeze(y.values)
                else:
                    raise ValueError("LightGBM doesn't support multi-label training")

                if reweighter is None:
                    w = None
                elif isinstance(reweighter, Reweighter):
                    w = reweighter.reweight(df)
                else:
                    raise ValueError("Unsupported reweighter type.")
                ds_l.append((lgb.Dataset(x.values, label=y, weight=w, free_raw_data=False), key))
        return ds_l

    def fit(
        self,
        dataset: DatasetH,
        num_boost_round=None,
        early_stopping_rounds=None,
        verbose_eval=20,
        evals_result=None,
        reweighter=None,
        **kwargs,
    ):
        if evals_result is None:
            evals_result = {}  # in case of unsafety of Python default values
        ds_l = self._prepare_data(dataset, reweighter)
        ds, names = list(zip(*ds_l))
        early_stopping_callback = lgb.early_stopping(
            self.early_stopping_rounds if early_stopping_rounds is None else early_stopping_rounds
        )
        # NOTE: if you encounter error here. Please upgrade your lightgbm
        verbose_eval_callback = lgb.log_evaluation(period=verbose_eval)
        evals_result_callback = lgb.record_evaluation(evals_result)
        self.model = lgb.train(
            self.params,
            ds[0],  # training dataset
            num_boost_round=self.num_boost_round if num_boost_round is None else num_boost_round,
            valid_sets=ds,
            valid_names=names,
            callbacks=[early_stopping_callback, verbose_eval_callback, evals_result_callback],
            **kwargs,
        )
        for k in names:
            for key, val in evals_result[k].items():
                name = f"{key}.{k}"
                for epoch, m in enumerate(val):
                    R.log_metrics(**{name.replace("@", "_"): m}, step=epoch)

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        return pd.Series(self.model.predict(x_test.values), index=x_test.index)

    def finetune(self, dataset: DatasetH, num_boost_round=10, verbose_eval=20, reweighter=None):
        """
        finetune model

        Parameters
        ----------
        dataset : DatasetH
            dataset for finetuning
        num_boost_round : int
            number of round to finetune model
        verbose_eval : int
            verbose level
        """
        # Based on existing model and finetune by train more rounds
        ds_l = self._prepare_data(dataset, reweighter)
        dtrain, _ = ds_l[0]

        if dtrain.construct().num_data() == 0:
            raise ValueError("Empty data from dataset, please check your dataset config.")
        verbose_eval_callback = lgb.log_evaluation(period=verbose_eval)
        self.model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            init_model=self.model,
            valid_sets=[dtrain],
            valid_names=["train"],
            callbacks=[verbose_eval_callback],
        )


class PCALGBModel(LGBModel):
    """自带 PCA 降维工作流的 LightGBM 模型"""

    def __init__(self, loss="mse", early_stopping_rounds=50, num_boost_round=1000, n_components=10, **kwargs):
        super().__init__(
            loss=loss, 
            early_stopping_rounds=early_stopping_rounds, 
            num_boost_round=num_boost_round, 
            **kwargs
        )
        # 2. 挂载额外的 PCA 组件
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components)
        self._is_pca_fitted = False

    def _prepare_data(self, dataset: DatasetH, reweighter=None) -> List[Tuple[lgb.Dataset, str]]:
        """重写父类的数据准备方法，在灌入 Dataset 之前拦截并进行 PCA 降维"""
        ds_l = []
        assert "train" in dataset.segments
        
        for key in ["train", "valid"]:
            if key in dataset.segments:
                df = dataset.prepare(key, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
                if df.empty:
                    raise ValueError("Empty data from dataset, please check your dataset config.")
                
                x, y = df["feature"], df["label"]

                # ==========================================
                # 注入 PCA 机制
                # ==========================================
                x_vals = x.fillna(0).values
                if key == "train":
                    x_vals = self.scaler.fit_transform(x_vals)
                    x_vals = self.pca.fit_transform(x_vals)
                    self._is_pca_fitted = True
                else: # "valid"
                    x_vals = self.scaler.transform(x_vals)
                    x_vals = self.pca.transform(x_vals)
                
                # 定义降维后的列名，保证 lgb.Dataset 知道特征名 (PC1, PC2...)
                pca_feature_names = [f"PC{i+1}" for i in range(self.n_components)]
                # ==========================================

                # 复用父类的标签和权重处理逻辑
                if y.values.ndim == 2 and y.values.shape[1] == 1:
                    y = np.squeeze(y.values)
                else:
                    raise ValueError("LightGBM doesn't support multi-label training")

                if reweighter is None:
                    w = None
                elif isinstance(reweighter, Reweighter):
                    w = reweighter.reweight(df)
                else:
                    raise ValueError("Unsupported reweighter type.")
                
                # 注意：传入的是降维后的 x_vals 以及自定义的 feature_name
                ds_l.append((lgb.Dataset(x_vals, label=y, weight=w, free_raw_data=False, feature_name=pca_feature_names), key))
                
        return ds_l

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        """重写父类的预测方法，截获推理视图并应用已拟合的 PCA"""
        if self.model is None or not getattr(self, "_is_pca_fitted", False):
            raise ValueError("model or PCA is not fitted yet!")
            
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        
        # ==========================================
        # 注入 PCA 机制
        # ==========================================
        x_vals = x_test.fillna(0).values
        x_vals = self.scaler.transform(x_vals)
        x_vals = self.pca.transform(x_vals)
        # ==========================================
        
        # 输出的 MultiIndex 与父类保持一致
        return pd.Series(self.model.predict(x_vals), index=x_test.index), x_test, x_vals
