import os
import time
import hashlib
import logging
import shutil
import json
from pathlib import Path


repo_root = os.getenv('PYTHONPATH', os.getcwd())
runtime_dirname = "runtime"


def register_run(base_dir_path, script_path, *params_paths, runtime_root=None):
    """
    Each simulation run gets a dedicated folder to work with.
    ---
    base_dir_path: should be a relative path from runtime_root.
    """

    # default
    if runtime_root is None:
        runtime_root = os.path.join(repo_root, runtime_dirname)

    # get current time
    current_time = time.localtime()

    # compute a hash based on content of script and parameters
    with open(script_path, 'rb') as fp:
        content = fp.read()
    for each_params in params_paths:
        with open(each_params, 'rb') as fp:
            content += fp.read()
    run_hash = hashlib.sha1(content).hexdigest()

    # combine into a unique ID
    timestamp = time.strftime("%y%m%d-%H%M%S", current_time)
    run_id = f"{timestamp}-{run_hash[:6]}"

    # create a dedicated folder for this simulation run
    run_dir_path = os.path.join(runtime_root, base_dir_path, run_id)
    run_dir = RunDir(run_dir_path)

    # put in initial values
    for each_params in params_paths:
        run_dir.add_params_file(each_params)

    # collect metadata
    metadata = {
        "run_id": run_id,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S.%f", current_time),
        "script": str(script_path),
        "parameters": [str(each_params) for each_params in params_paths],
    }
    run_dir.update_metadata(metadata)

    return run_dir


class RunDir:

    def __init__(self, path):
        self.path = Path(path)

        # create the folder and all its sub-folders
        self.path.mkdir(parents=True, exist_ok=False)
        self.intermediate_dir.mkdir()
        self.parameters_dir.mkdir()
        self.results_dir.mkdir()
        self.visuals_dir.mkdir()
        self.log_file.touch()
        with open(self.metadata_file, 'w', encoding='utf-8') as fp:
            json.dump({}, fp)

        # prepare the logger
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        logger = self.get_logger()
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

    @property
    def intermediate_dir(self):
        return self.path / "_intermediate"

    @property
    def parameters_dir(self):
        return self.path / "parameters"

    @property
    def results_dir(self):
        return self.path / "results"

    @property
    def visuals_dir(self):
        return self.path / "visuals"

    @property
    def log_file(self):
        return self.path / "log.txt"

    @property
    def metadata_file(self):
        return self.path / "METADATA.json"

    def add_params_file(self, file_path):
        shutil.copy2(file_path, self.parameters_dir)

    def get_logger(self):
        return logging.getLogger(str(self.path))

    def update_metadata(self, new_info):
        with open(self.metadata_file, 'r', encoding='utf-8') as fp:
            metadata = json.load(fp)
        metadata.update(new_info)
        with open(self.metadata_file, 'w', encoding='utf-8') as fp:
            json.dump(metadata, fp, indent=2, sort_keys=True)
