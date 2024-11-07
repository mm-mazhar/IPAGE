import sys
import os

# Add the project root directory to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(f"Path: {os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}")

from src.configs.common_configs import get_config, CONFIG_YML_PATH

print(f"CONFIG_YML_PATH: {CONFIG_YML_PATH}")

cfg = get_config()
print(f"ROOT_DIR: {cfg.ROOT_DIR}")