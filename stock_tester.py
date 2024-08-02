import os
from factor_processors.data_loader import DataLoader
import rqdatac
import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetRegressorStock
from sklearn.model_selection import train_test_split
from pytorch_tabnet.utils import StockDataLoaderCS, StockDatasetCS
from pytorch_tabnet.metrics import (
    PearsonCorrelationAbsMetric,
    PearsonCorrelationSqrMetric,
    pearsonCorrelationLossAbs,
    pearsonCorrelationLossSqr,
)


def get_feature():
    data = pd.read_pickle("./sample_data.pkl")
    data["label"] = data.iloc[:, 0].groupby(level=1).shift(-1)
    data.dropna(inplace=True)
    return data


if __name__ == "__main__":
    rqdatac.init()
    start_date = pd.to_datetime("20210101")
    end_date = pd.to_datetime("20210301")

    data = get_feature()
    features = data.drop(columns=["label"])
    label = data["label"]
    dates = data.index.get_level_values(0).unique()
    train_dates, valid_dates = train_test_split(dates, test_size=0.2, random_state=42)

    X_train = features.loc[train_dates]
    y_train = label.loc[train_dates]
    X_valid = features.loc[valid_dates]
    y_valid = label.loc[valid_dates]

    clf = TabNetRegressorStock()

    clf.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_name=["train", "valid"],
        eval_metric=["rmse", PearsonCorrelationAbsMetric],
        max_epochs=10,
        patience=0,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
        loss_fn=pearsonCorrelationLossAbs,
    )
