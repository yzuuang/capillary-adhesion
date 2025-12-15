import os
import time
import subprocess
import shutil
import json
from pathlib import Path


repo_root = os.getenv("PYTHONPATH", os.getcwd())
results_dirname = "results"


def register_run(base_dir_path, script_path, *param_paths, runtime_root=None, with_hash: bool = True):
    """
    Each simulation run gets a dedicated folder to work with.
    ---
    base_dir_path: should be a relative path from runtime_root.
    """

    # default
    if runtime_root is None:
        runtime_root = os.path.join(repo_root, results_dirname)

    # get current time
    current_time = time.localtime()

    # use the timestamp as id
    run_id = time.strftime("%y%m%d-%H%M%S", current_time)

    # get the git hash
    if with_hash:
        git_hash = get_git_hash()
        if len(git_hash):
            run_id += f"-{git_hash[:6]}"

    # create a dedicated folder for this simulation run
    run_dir_path = os.path.join(runtime_root, base_dir_path, run_id)
    run_dir = RunDir(run_dir_path)
    run_dir.setup_directory()

    # put in initial values
    if len(param_paths):
        for each_path in param_paths:
            run_dir.add_parameter_file(each_path)

    # collect metadata
    metadata = {
        "run_id": run_id,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S", current_time),
        "script": str(os.path.abspath(script_path)),
    }
    if len(param_paths):
        metadata.update({"parameters": [str(os.path.abspath(each_params)) for each_params in param_paths]})
    if with_hash and len(git_hash):
        metadata.update({"git_hash": git_hash})
    run_dir.update_metadata(metadata)

    return run_dir


def retrieve_run(base_dir_path, run_id, runtime_root=None):
    """
    Retrieve the folder of the specified run.
    ---
    base_dir_path: should be a relative path from runtime_root.
    """

    # default
    if runtime_root is None:
        runtime_root = os.path.join(repo_root, results_dirname)

    # get the dedicated folder
    run_dir_path = os.path.join(runtime_root, base_dir_path, run_id)
    return RunDir(run_dir_path)


class RunDir:

    def __init__(self, path):
        self.path = Path(path)
        self.run_id = os.path.basename(self.path)

    def setup_directory(self):
        # create folders and files
        self.path.mkdir(parents=True, exist_ok=False)
        self.intermediate_dir.mkdir()
        self.parameters_dir.mkdir()
        self.results_dir.mkdir()
        self.visuals_dir.mkdir()
        self.log_file.touch()
        with open(self.metadata_file, "w", encoding="utf-8") as fp:
            json.dump({}, fp)

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

    def add_parameter_file(self, file_path):
        shutil.copy2(file_path, self.parameters_dir)

    def update_metadata(self, new_info):
        with open(self.metadata_file, "r", encoding="utf-8") as fp:
            metadata = json.load(fp)
        metadata.update(new_info)
        with open(self.metadata_file, "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2, sort_keys=True)


def get_git_hash():
    try:
        hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        return hash
    except Exception as e:
        print(f"Error getting git hash: {e}")
        return None
