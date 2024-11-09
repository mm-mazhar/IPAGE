# -*- coding: utf-8 -*-
# """
# common_configs.py
# Created on Oct 21, 2024
# @ Author: Mazhar
# """

import logging
from pathlib import Path
from typing import Any

import os
import yaml
from colorama import Fore, Style, init
from easydict import EasyDict

# Initialize colorama for colored logs
init(autoreset=True)

CONFIG_YML_PATH = "./src/app/configs/configs.yml"

# # Use an absolute path for CONFIG_YML_PATH
# CONFIG_YML_PATH = os.path.join(os.path.dirname(__file__))
# print(f"CONFIG_YML_PATH: {CONFIG_YML_PATH}")

# Custom formatter to include colors in logs
class ColoredFormatter(logging.Formatter):
    def format(self, record) -> str:
        # Add colors based on the log level
        if record.levelno == logging.INFO:
            record.msg = f"{Fore.GREEN}{record.msg}{Style.RESET_ALL}"
        elif record.levelno == logging.DEBUG:
            record.msg = f"{Fore.CYAN}{record.msg}{Style.RESET_ALL}"
        elif record.levelno == logging.WARNING:
            record.msg = f"{Fore.YELLOW}{record.msg}{Style.RESET_ALL}"
        elif record.levelno == logging.ERROR:
            record.msg = f"{Fore.RED}{record.msg}{Style.RESET_ALL}"
        elif record.levelno == logging.CRITICAL:
            record.msg = f"{Fore.MAGENTA}{record.msg}{Style.RESET_ALL}"
        return super().format(record)


# Function to load configuration from YAML file
def cfg_from_yaml_file(cfg_file) -> EasyDict:
    with open(cfg_file, "r") as f:
        try:
            new_config: Any = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config: Any = yaml.load(f)

    cfg = EasyDict(new_config)
    cfg.ROOT_DIR = (Path(__file__).resolve().parent / "../").resolve()
    return cfg


# Logging setup based on configs.yml
def setup_logging(config_path=CONFIG_YML_PATH) -> logging.Logger:
    # Load the YAML config file for logging
    with open(config_path, "r") as file:
        config: Any = yaml.safe_load(file)

    # Access the logging variables from the YAML config
    LOG_FILE: str = config["LOGGING"].get("LOG_FILE", None)
    LOG_LEVEL: str = config["LOGGING"]["LOG_LEVEL"]
    LOG_FORMAT: str = config["LOGGING"]["LOG_FORMAT"]

    # Convert LOG_LEVEL to logging module constants
    log_level: int = logging.INFO if LOG_LEVEL == "INFO" else logging.DEBUG

    # Set up logging formatter and handler
    formatter = ColoredFormatter(LOG_FORMAT)
    handler: logging.Handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    # Basic logging configuration with or without file logging
    # To log to file, set/uncomment LOG_FILE in configs.yml
    if LOG_FILE:
        logging.basicConfig(
            level=log_level,
            handlers=[handler, logging.FileHandler(LOG_FILE)],
        )
    else:
        logging.basicConfig(
            level=log_level,
            handlers=[handler],
        )

    # Create and configure the logger object
    logger: logging.Logger = logging.getLogger()

    # Ensure the logger is only configured once
    if not logger.hasHandlers():
        logger.addHandler(handler)

    return logger


# Function to get logger, this will be used throughout the project
def get_logger(config_path=CONFIG_YML_PATH) -> logging.Logger:
    return setup_logging(config_path)


# # Distributed PyTorch initialization
# def init_dist_pytorch(batch_size, local_rank, backend="nccl") -> tuple[Any, int]:
#     if mp.get_start_method(allow_none=True) is None:
#         mp.set_start_method("spawn")
#     num_gpus: int = torch.cuda.device_count()
#     torch.cuda.set_device(local_rank % num_gpus)
#     dist.init_process_group(backend=backend)
#     assert (
#         batch_size % num_gpus == 0
#     ), "Batch size should be matched with GPUS: (%d, %d)" % (batch_size, num_gpus)
#     batch_size_each_gpu: int = batch_size // num_gpus
#     rank: int = dist.get_rank()
#     return batch_size_each_gpu, rank


# # Get distributed information for PyTorch
# def get_dist_info() -> tuple[int, int]:
#     if torch.__version__ < "1.0":
#         initialized: bool = dist._initialized
#     else:
#         if dist.is_available():
#             initialized: bool = dist.is_initialized()
#         else:
#             initialized = False
#     if initialized:
#         rank: int = dist.get_rank()
#         world_size: int = dist.get_world_size()
#     else:
#         rank: int = 0
#         world_size: int = 1
#     return rank, world_size


# Load the configuration once
_cfg: EasyDict = cfg_from_yaml_file(CONFIG_YML_PATH)


def get_config() -> EasyDict:
    """Return the loaded configuration."""
    return _cfg


# from common_config import get_logger

# logger = get_logger()

# logger.info("This is an info message.")
# logger.debug("This is a debug message.")
