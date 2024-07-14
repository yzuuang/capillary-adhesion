"""
Store data in a self-describing manner.
"""

import contextlib
# import dataclasses  # no specification; instead, use dict for flexibility
import json
import numpy as np
import os
import shutil


@contextlib.contextmanager
def working_directory(path, new: bool):
    """Manage changing directories via `with` statement."""

    cwd = os.getcwd()

    if new:
        # for a new start, remove everything in the target directory
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    os.chdir(path)
    try:
        yield
    finally:
        # always change back to the original directory
        os.chdir(cwd)


_dash = "---"
_UTF_8 = "UTF-8"
_CSV_separator = ";"
_JSON_indent = 2


def _save_here(label: str, data: dict):

    for key, value in data.items():
        # array
        if isinstance(value, np.ndarray):
            csv_file = f"{label}{_dash}{key}.csv"
            np.savetxt(csv_file, value, delimiter=_CSV_separator)
            data[key] = csv_file
        # dict (recursive)
        elif isinstance(value, dict):
            sub_label = f"{label}{_dash}{key}"
            _save_here(sub_label, value)
            data[key] = f"{sub_label}.json"
        # list of dict (recursive)
        elif isinstance(value, list):
            for count, nested in enumerate(value):
                if isinstance(nested, dict):
                    sub_label = f"{label}{_dash}{key}{_dash}{count}"
                    _save_here(sub_label, nested)
                    value[count] = f"{sub_label}.json"

    # save to JSON, label as filename
    with open(f"{label}.json", mode="w", encoding=_UTF_8) as file:
        json.dump(data, file, indent=_JSON_indent)


def _load_here(label: str):

    # load JSON, label as filename
    with open(f"{label}.json", mode="r", encoding=_UTF_8) as file:
        data: dict = json.load(file)

    for key, value in data.items():
        # array
        if isinstance(value, str) and value.endswith(".csv"):
            data[key] = np.loadtxt(value, delimiter=_CSV_separator)
        # dict (recursive)
        elif isinstance(value, str) and value.endswith(".json"):
            data[key] = _load_here(value[:-5])
        # list of dict (recursive)
        elif isinstance(value, list):
            for count, nested in enumerate(value):
                if isinstance(nested, str) and nested.endswith(".json"):
                    value[count] = _load_here(nested[:-5])

    return data


def save(path, label: str, data: dict):

    folder = os.path.join(path, label)
    with working_directory(folder, new=True):
        _save_here(label, data)


def load(path, label: str):

    folder = os.path.join(path, label)
    with working_directory(folder, new=False):
        return _load_here(label)
