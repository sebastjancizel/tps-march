from pathlib import Path

COMPETITION_NAME = "tabular-playground-series-mar-2021"

ROOT_DIR = Path("/Users/sebastjancizel/Documents/Projects.tmp/tps-march")
DATA_DIR = ROOT_DIR / "input"

RAW_TRAIN_DATA = DATA_DIR / "train.csv"
RAW_TEST_DATA = DATA_DIR / "test.csv"

TRAIN_DATA = DATA_DIR / "train_enc.csv"
TEST_DATA = DATA_DIR / "test_enc.csv"
TRAIN_FOLDS = DATA_DIR / "train_folds.csv"

RANDOM_STATE = 42


if __name__ == "__main__":
    print(f"Root dir is : {ROOT_DIR}")
    print(f"Data dir is : {DATA_DIR}")