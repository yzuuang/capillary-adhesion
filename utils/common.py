import os
import configparser


repo_root = os.getenv('PYTHONPATH')


def get_runtime_dir(name: str):
    """A dedicated directory to dump outputs on-the-fly."""
    return os.path.join(repo_root, 'runtime', name)


def read_configs(files: list):
    config = configparser.ConfigParser()
    for file in files:
        if not os.path.exists(file):
            raise RuntimeError(f"Non-existing file {file}")
        print(f"Reading config from {file}")
        config.read(file)
    return config
