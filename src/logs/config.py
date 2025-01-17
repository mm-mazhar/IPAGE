from pathlib import Path

# Configurations here are used in the logger.py file
# and are subject to change based on the project requirements

LOGS_FILE_PATH = Path().absolute() / "src" / "logs"
DATA_LOG_FILE = LOGS_FILE_PATH / "data.log"
MODEL_LOG_FILE = LOGS_FILE_PATH / "model.log"