import sys
import os
import pandas as pd
import numpy as np
from functools import partial

from sklearn import metrics
from scipy.optimize import fmin


def print_score(model, x_valid, y_valid):
    preds = model.predict_proba(x_valid)[:, 1]
    score = metrics.roc_auc_score(y_valid, preds)
    print(f"Model {model.__class__.__name__}, AUC score: {score:.6f}")


def split_fold(df, fold, features):
    """
    Utility function that splits the dataset into folds at every step.
    """
    return (
        df.loc[df.kfold != fold, features],
        df.loc[df.kfold != fold, "target"],
        df.loc[df.kfold == fold, features],
        df.loc[df.kfold == fold, "target"],
    )


class OptimizeAUC:
    def __init__(self):
        self.coef_ = 0

    def _auc(self, coef, X, y):
        x_coef = X * coef
        predictions = np.sum(x_coef, axis=1)
        auc_score = metrics.roc_auc_score(y, predictions)

        return -1.0 * auc_score

    def fit(self, X, y):
        loss_partial = partial(self._auc, X=X, y=y)
        initial_coef = np.random.dirichlet(np.ones(X.shape[1]), size=1)

        self.coef_ = fmin(loss_partial, initial_coef, disp=False)

    def predict(self, X):
        x_coef = X * self.coef_
        predictions = np.sum(x_coef, axis=1)
        return predictions