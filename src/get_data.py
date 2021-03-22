import os
import config
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

if __name__ == "__main__":
    # os.environ["KAGGLE_USERNAME"] = USERNAME
    # os.environ["KAGGLE_KEY"] = API_KEY

    api = KaggleApi()
    api.authenticate()

    api.competition_download_file(
        config.COMPETITION_NAME, "train.csv", path=config.DATA_DIR
    )

    api.competition_download_file(
        config.COMPETITION_NAME, "test.csv", path=config.DATA_DIR
    )

    with zipfile.ZipFile(config.DATA_DIR / "train.csv.zip", "r") as zipref:
        zipref.extractall(config.DATA_DIR)

    with zipfile.ZipFile(config.DATA_DIR / "test.csv.zip", "r") as zipref:
        zipref.extractall(config.DATA_DIR)

    os.remove(config.DATA_DIR / "test.csv.zip")
    os.remove(config.DATA_DIR / "train.csv.zip")