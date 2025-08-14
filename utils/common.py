import os
import configparser


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
