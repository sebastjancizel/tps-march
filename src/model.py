import sys
import os
import pandas as pd
import numpy as np
import config
import model_parameters

from utils import OptimizeAUC, print_score, split_fold
from tqdm import tqdm
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn import ensemble


def fit_ensamble(x_train, y_train, x_valid, y_valid):
    lgbm = LGBMClassifier(**model_parameters.LGB_PARAMS)
    lgbm.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        verbose=False,
    )
    print_score(lgbm, x_valid, y_valid)

    xgbm = XGBClassifier(model_parameters.XGB_PARAMS)
    xgbm.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="auc",
        verbose=False,
    )
    print_score(xgbm, x_valid, y_valid)

    ada = ensemble.AdaBoostClassifier()
    ada.fit(x_train, y_train)
    print_score(ada, x_valid, y_valid)

    return lgbm, xgbm, ada


def run():
    df = pd.read_csv(
        config.TRAIN_DATA,
    )

    for i in range(3):
        df[f"model_{i}"] = 0

    cat_cols = [col for col in df.columns if col.endswith("le")]
    cont_cols = [col for col in df.columns if col.startswith("cont")]
    features = cont_cols + cat_cols

    for fold in tqdm(range(10)):
        print(f"\n Starting fold: {fold}")

        x_train, y_train, x_valid, y_valid = split_fold(df, fold, features)

        models = fit_ensamble(x_train, y_train, x_valid, y_valid)

        predictions = []

        for i, model in enumerate(models):
            probs = model.predict_proba(x_valid)[:, 1]
            df.loc[df.kfold == fold, f"model_{i}"] = probs
            predictions.append(probs)
        print("-" * 50)

        column_stack = np.column_stack(predictions)

        opt = OptimizeAUC()
        opt = opt.fit(column_stack, y_valid)

        print("=" * 50)

    l2_features = [f"model_{i}" for i in range(3)]

    for fold in tqdm(range(10)):
        print(f"Starting fold: {fold}")

        x_train, y_train, x_valid, y_valid = split_fold(df, fold, l2_features)

        lgbm = LGBMClassifier()
        lgbm.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            verbose=False,
        )

        print_score(lgbm, x_valid, y_valid)


if __name__ == "__main__":
    run()
