"""
Store data in a self-describing manner.
"""

import contextlib
import dataclasses as dc
import datetime as dt
import json
import os
import shutil
import typing

import numpy as np


_dash = "---"
_UTF_8 = "UTF-8"
_CSV_separator = ";"
_JSON_indent = 2
_archive_format = "tar"  # uncompressed


@contextlib.contextmanager
def working_directory(path, read_only: bool):
    # Save original directory for later use
    origin = os.getcwd()

    # Create instance
    if read_only:
        store = FilesToRead(path)
    else:
        # Create directory if necessary
        if not os.path.exists(path):
            os.makedirs(path)
        store = FilesToReadWrite(path)

    # CD via context managing
    os.chdir(path)
    try:
        yield store
    finally:
        # A new record after changes
        if not read_only:
            store.pack()
        # CD back
        os.chdir(origin)


class FilesToRead:

    def __init__(self, path):
        self._path_root = path

    def load(self, label: str, DataType=None):
        data = _load_here(label)
        if DataType is None:
            return data
        if dc.is_dataclass(DataType):
            return _recover_dataclass(data, DataType)
        return DataType(data)


class FilesToReadWrite(FilesToRead):

    def save(self, label: str, data):
        if dc.is_dataclass(data):
            data = dc.asdict(data)
        _save_here(label, data)

    def pack(self):
        # For the sake of immutability, mark it with a timestamp
        timestamp = dt.datetime.now().strftime("%y%m%d-%H%M%S")
        target_path = f"{self._path_root}{_dash}{timestamp}"
        shutil.make_archive(target_path, _archive_format)

    def brand_new(self):
        for entry in os.scandir(self._path_root):
            if entry.is_dir():
                shutil.rmtree(entry.path)
            else:  # file or link
                os.remove(entry.path)


def _save_here(label: str, data: dict):
    """Support a few more types for json to dump"""
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
        elif isinstance(value, list):
            for count, nested in enumerate(value):
                # save list of nested dict into several JSON and keep filenames (recursive)
                if isinstance(nested, dict):
                    sub_label = f"{label}{_dash}{key}{_dash}{count}"
                    _save_here(sub_label, nested)
                    value[count] = f"{sub_label}.json"
                # list of arrays into several CSV and keep filenames
                elif isinstance(nested, np.ndarray):
                    csv_file = f"{label}{_dash}{key}{_dash}{count}.csv"
                    np.savetxt(csv_file, nested, delimiter=_CSV_separator)
                    value[count] = csv_file

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
            # Hack: skip this "troublemaking" value
            if not value.endswith("p.csv"):
                data[key] = np.loadtxt(value, delimiter=_CSV_separator)
        # recover nested dict from nested JSON (recursive)
        elif isinstance(value, str) and value.endswith(".json"):
            data[key] = _load_here(value[:-5])
        elif isinstance(value, list):
            for count, nested in enumerate(value):
                # recover list of nested dict from list of JSON (recursive)
                if isinstance(nested, str) and nested.endswith(".json"):
                    value[count] = _load_here(nested.removesuffix(".json"))
                # recover list of arrays from list of CSV
                elif isinstance(nested, str) and nested.endswith(".csv"):
                    value[count] = np.loadtxt(nested, delimiter=_CSV_separator)

    return data


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

    # Hack: skip this "troublemaking" value
    if "p" in data.keys():
        del data["p"]
    return DataType(**data)
