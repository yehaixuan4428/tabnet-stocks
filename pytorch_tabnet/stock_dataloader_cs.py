import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from joblib import Parallel, delayed
from tqdm import tqdm
import numba


# Custom Dataset and DataLoader
class StockDatasetCS(Dataset):
    """
    截面特征
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.dates = x.index.get_level_values(0).unique()

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        date = self.dates[idx]
        x, y = self.x.loc[date].values, self.y.loc[date].values.reshape(-1, 1)
        return x, y


class StockDataLoaderCS(DataLoader):
    """
    custom stock market dataloader, 每个batch只包含单日全部股票
    """

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, batch_size=1, collate_fn=self.collate_fn, **kwargs)

    def collate_fn(self, batch):
        features, labels = zip(*batch)
        features = torch.tensor(features[0], dtype=torch.float)
        labels = torch.tensor(labels[0], dtype=torch.float)
        return features, labels


class StockPredictDatasetCS(Dataset):
    """
    截面特征
    """

    def __init__(self, x):
        self.x = x
        self.dates = x.index.get_level_values(0).unique()

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        date = self.dates[idx]
        x = self.x.loc[date].values
        return x


class StockPredictDataLoaderCS(DataLoader):
    """
    custom stock market dataloader, 每个batch只包含单日全部股票
    """

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, batch_size=1, collate_fn=self.collate_fn, **kwargs)

    def collate_fn(self, batch):
        features = torch.tensor(batch[0], dtype=torch.float)
        return features


def validate_eval_set_stocks(eval_set, eval_name, X_train, y_train):
    """Check if the shapes of eval_set are compatible with (X_train, y_train).

    Parameters
    ----------
    eval_set : list of tuple
        List of eval tuple set (X, y).
        The last one is used for early stopping
    eval_name : list of str
        List of eval set names.
    X_train : np.ndarray
        Train owned products
    y_train : np.array
        Train targeted products

    Returns
    -------
    eval_names : list of str
        Validated list of eval_names.
    eval_set : list of tuple
        Validated list of eval_set.

    """
    eval_name = eval_name or [f"val_{i}" for i in range(len(eval_set))]

    assert len(eval_set) == len(
        eval_name
    ), "eval_set and eval_name have not the same length"
    if len(eval_set) > 0:
        assert all(
            len(elem) == 2 for elem in eval_set
        ), "Each tuple of eval_set need to have two elements"
    for name, (X, y) in zip(eval_name, eval_set):
        msg = (
            f"Dimension mismatch between X_{name} "
            + f"{X.shape} and X_train {X_train.shape}"
        )
        assert len(X.shape) == len(X_train.shape), msg

        msg = (
            f"Dimension mismatch between y_{name} "
            + f"{y.shape} and y_train {y_train.shape}"
        )
        assert len(y.shape) == len(y_train.shape), msg

        msg = (
            f"Number of columns is different between X_{name} "
            + f"({X.shape[1]}) and X_train ({X_train.shape[1]})"
        )
        assert X.shape[1] == X_train.shape[1], msg

        if len(y_train.shape) == 2:
            msg = (
                f"Number of columns is different between y_{name} "
                + f"({y.shape[1]}) and y_train ({y_train.shape[1]})"
            )
            assert y.shape[1] == y_train.shape[1], msg
        msg = (
            f"You need the same number of rows between X_{name} "
            + f"({X.shape[0]}) and y_{name} ({y.shape[0]})"
        )
        assert X.shape[0] == y.shape[0], msg

    return eval_name, eval_set


if __name__ == "__main__":
    pass
