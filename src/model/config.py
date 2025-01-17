# config.py file
from pathlib import Path

# Configurations here are used in the model.py file
# and are subject to change based on the project requirements

DATASET_VERSION = "v3" # using `merged_v3.csv`

RANDOM_STATE = 42
TEST_SIZE = 0.2

# current working directory(i.e root 'IPAGE') + data + models
MODEL_FILE_PATH = Path().absolute() / "data" / "models"
MODEL_FILE_PATH.mkdir(parents=True, exist_ok=True)

# Columns to drop from the dataset
# these columns are also subject to change based on the project requirements
ALL_TARGETS = ["SOC", "Zinc", "Boron"]
COLS_TO_DROP = ["longitude", "latitude", "Soil group", "Land class", "Soil type"]
