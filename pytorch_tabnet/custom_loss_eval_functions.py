import torch
import torch.nn as nn
import numpy as np
from pytorch_tabnet.metrics import Metric


def pearsonCorrelationLossSqr(y_pred, y_true):
    """
    Dummy example similar to using default torch.nn.functional.cross_entropy
    """
    data = torch.cat((y_pred, y_true), dim=1)
    corr_matrix = torch.corrcoef(data.T)
    return -torch.square(corr_matrix[0, 1])


class PearsonCorrelationSqrMetric(Metric):

    def __init__(self):
        self._name = "IC"  # write an understandable name here
        self._maximize = True

    def __call__(self, y_true, y_score):
        """
        Compute AUC of predictions.

        Parameters
        ----------
        y_true: np.ndarray
            Target matrix or vector
        y_score: np.ndarray
            Score matrix or vector

        Returns
        -------
            float
            AUC of predictions vs targets.
        """
        return (np.corrcoef(y_true.squeeze(), y_score.squeeze())[0, 1]) ** 2


def pearsonCorrelationLossAbs(y_pred, y_true):
    """
    Dummy example similar to using default torch.nn.functional.cross_entropy
    """
    data = torch.cat((y_pred, y_true), dim=1)
    corr_matrix = torch.corrcoef(data.T)
    return -torch.abs(corr_matrix[0, 1])


class PearsonCorrelationAbsMetric(Metric):

    def __init__(self):
        self._name = "IC"  # write an understandable name here
        self._maximize = True

    def __call__(self, y_true, y_score):
        """
        Compute AUC of predictions.

        Parameters
        ----------
        y_true: np.ndarray
            Target matrix or vector
        y_score: np.ndarray
            Score matrix or vector

        Returns
        -------
            float
            AUC of predictions vs targets.
        """
        return np.abs(np.corrcoef(y_true.squeeze(), y_score.squeeze())[0, 1])
