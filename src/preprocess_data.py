import config
import pandas as pd

from sklearn import model_selection
from sklearn import preprocessing


if __name__ == "__main__":
    df_train = pd.read_csv(config.RAW_TRAIN_DATA, index_col="id")
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_test = pd.read_csv(config.RAW_TEST_DATA, index_col="id")

    all_data = pd.concat([df_train.drop("target", axis=1), df_test]).reset_index(
        drop=True
    )

    assert all_data.shape == (
        df_train.shape[0] + df_test.shape[0],
        df_test.shape[1],
    ), f"Unexpected shape of all_data {all_data.shape}"

    cat_cols = [col for col in all_data.columns if col.startswith("cat")]

    for col in cat_cols:
        le = preprocessing.LabelEncoder()
        le.fit(all_data[col])
        df_train[col + "_le"] = le.transform(df_train[col].values)
        df_test[col + "_le"] = le.transform(df_test[col].values)

    kf = model_selection.StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

    for fold_, (train_, valid_) in enumerate(kf.split(X=df_train, y=df_train.target)):
        df_train.loc[valid_, "kfold"] = fold_

    df_train.to_csv(config.TRAIN_DATA, index=False)
    df_test.to_csv(config.TEST_DATA, index=False)
