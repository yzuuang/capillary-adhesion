import os
import sys
import configparser
import logging


def read_configs(files: list):
    if not len(files):
        raise RuntimeError(f"Provide at least one config file.")
    config = configparser.ConfigParser()
    for file in files:
        if not os.path.exists(file):
            raise RuntimeError(f"Non-existing file {file}")
        print(f"Reading config from {file}")
        config.read(file)
    return config


def init_logging():
    """
    Config the logger such that logging.info(...) works like print(...)
    """
    root_logger = logging.getLogger()

    # clear any existing handlers
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # config logging to console as if calling print(...)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    root_logger.addHandler(console_handler)

    # set logging level
    root_logger.setLevel(logging.INFO)
