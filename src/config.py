import sys
from pathlib import Path

COMPETITION_NAME = "tabular-playground-series-mar-2021"

if sys.platform == "darwin":
    ROOT_DIR = Path("/Users/sebastjancizel/Documents/Projects.tmp/tps-march")
else:
    ROOT_DIR = Path("/content/tps-march")

DATA_DIR = ROOT_DIR / "input"
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = ROOT_DIR / "runs"

RAW_TRAIN_DATA = DATA_DIR / "train.csv"
RAW_TEST_DATA = DATA_DIR / "test.csv"

TRAIN_DATA = DATA_DIR / "train_enc.csv"
TEST_DATA = DATA_DIR / "test_enc.csv"

RANDOM_STATE = 42


if __name__ == "__main__":
    print(f"Root dir is : {ROOT_DIR}")
    print(f"Data dir is : {DATA_DIR}")