import sys
import os
import pandas as pd
import numpy as np
import config
import model_parameters

from utils import OptimizeAUC
from tqdm import tqdm
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn import linear_model
from sklearn import ensamble


def fit_ensamble(x_train, y_train):
    lgbm = LGBMClassifier()
    lgbm.fit(x_train, y_train, verbose=False)

    xgbm = XGBClassifier(eval_metric="logloss")
    xgbm.fit(x_train, y_train, verbose=False)

    logres = linear_model.LogisticRegression(max_iter=1000)
    logres.fit(x_train, y_train)

    return lgbm, xgbm, logres


def run():
    df = pd.read_csv(
        config.TRAIN_DATA,
    )

    cat_cols = [col for col in df.columns if col.endswith("le")]
    cont_cols = [col for col in df.columns if col.startswith("cont")]
    features = cont_cols + cat_cols

    for fold in tqdm(range(10)):
        print(f"Starting fold: {fold}")
        x_train = df.loc[df.kfold != fold, features]
        y_train = df.loc[df.kfold != fold, "target"]
        x_valid = df.loc[df.kfold == fold, features]
        y_valid = df.loc[df.kfold == fold, "target"]

        models = fit_ensamble(x_train, y_train)

        predictions = []

        for model in models:
            probs = model.predict_proba(x_valid)[:, 1]
            print(f"Model {model.__class__.__name__}:")
            model_score = metrics.roc_auc_score(y_valid, probs)
            print(f"AUC score: {model_score}.")
            predictions.append(probs)
        print("-" * 50)

        column_stack = np.column_stack(predictions)

        opt = OptimizeAUC()
        opt = opt.fit(column_stack, y_valid)

        print("=" * 50)


if __name__ == "__main__":
    run()
