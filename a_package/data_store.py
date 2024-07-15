"""
Store data in a self-describing manner.

So far, flat number, string, numeric array, nested dataclass, list of nested dataclass are being considered.
"""

import contextlib
import dataclasses as dc
import json
import numpy as np
import os
import shutil
import typing


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
        # save array into CSV and keep filename
        if isinstance(value, np.ndarray):
            csv_file = f"{label}{_dash}{key}.csv"
            np.savetxt(csv_file, value, delimiter=_CSV_separator)
            data[key] = csv_file
        # save nested dict into JSON and keep filename (recursive)
        elif isinstance(value, dict):
            sub_label = f"{label}{_dash}{key}"
            _save_here(sub_label, value)
            data[key] = f"{sub_label}.json"
        # save list of nested dict into several JSON and keep filenames (recursive)
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
        # recover array from CSV
        if isinstance(value, str) and value.endswith(".csv"):
            data[key] = np.loadtxt(value, delimiter=_CSV_separator)
        # recover nested dict from nested JSON (recursive)
        elif isinstance(value, str) and value.endswith(".json"):
            data[key] = _load_here(value[:-5])
        # recover list of nested dict from list of JSON (recursive)
        elif isinstance(value, list):
            for count, nested in enumerate(value):
                if isinstance(nested, str) and nested.endswith(".json"):
                    value[count] = _load_here(nested[:-5])

    return data


_archive_format = "tar"  # uncompressed


def save(path, label: str, data):

    if dc.is_dataclass(data):
        data = dc.asdict(data)

    folder = os.path.join(path, label)
    with working_directory(folder, new=True):
        _save_here(label, data)
        shutil.make_archive(folder, _archive_format)


def load(path, label: str, DataType = None):

    folder = os.path.join(path, label)
    with working_directory(folder, new=True):
        shutil.unpack_archive(f"{folder}.{_archive_format}")
        data = _load_here(label)

    if DataType is None:
        return data

    if dc.is_dataclass(DataType):
        return _recover_dataclass(data, DataType)

    return DataType(data)


def _recover_dataclass(data: dict, DataType):

    # check and construct the nested dataclass instances
    for field in dc.fields(DataType):
        # nested dataclass (recursive)
        if dc.is_dataclass(field.type):
            value = data[field.name]
            data[field.name] = _recover_dataclass(value, field.type)
        # list of nested dataclass (recursive)
        if typing.get_origin(field.type) is list:
            SubDataType = typing.get_args(field.type)[0]
            if dc.is_dataclass(SubDataType):
                value = data[field.name]
                data[field.name] = [_recover_dataclass(element, SubDataType) for element in value]

    return DataType(**data)
