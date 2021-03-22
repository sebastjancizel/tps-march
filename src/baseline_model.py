import sys
import os
import pandas as pd
import numpy as np
import config
import model_parameters
from tqdm import tqdm

from lightgbm import LGBMClassifier
from sklearn import metrics


def run(fold):
    df = pd.read_csv(
        config.TRAIN_DATA,
    )

    cat_cols = [col for col in df.columns if col.endswith("le")]
    cont_cols = [col for col in df.columns if col.startswith("cont")]
    features = cont_cols + cat_cols

    x_train = df.loc[df.kfold != fold, features]
    y_train = df.loc[df.kfold != fold, "target"]
    x_valid = df.loc[df.kfold == fold, features]
    y_valid = df.loc[df.kfold == fold, "target"]

    lgbm = LGBMClassifier()
    lgbm.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)

    y_preds = lgbm.predict_proba(x_valid)[:, 1]
    auc_score = metrics.roc_auc_score(y_valid, y_preds)
    print(f"Validation AUC: {auc_score}")


if __name__ == "__main__":
    for fold in tqdm(range(10)):
        run(fold)