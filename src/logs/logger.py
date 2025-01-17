"""
Module for logging
"""
import sys
import logging


def setLogger(name, filename, level=logging.DEBUG, display_console=True):
    logger = logging.getLogger(name)

    formatter = logging.Formatter(
        '{asctime}:{name}:{levelname}:{message}',
        datefmt='%Y-%m-%d %H:%M:%S',
        style='{',
    )

    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if display_console:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    
    logger.setLevel(level)

    return logger
