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
import pathlib

import numpy as np

from a_package.grid import Grid
from a_package.field import Field


class StepRecorder:

    record: dict
    step_count: int
    npy_io: "NpyIO"
    csv_io: "CsvIO"

    def __init__(self, root_path):
        self.step_count = 0
        self.npy_io = NpyIO(root_path)
        self.csv_io = CsvIO(root_path)

    def reset_step_count(self):
        self.step_count = 0

    def save_new_step(self, grid: Grid, fields: dict[str, Field] = {}, scalars: dict[str, float] = {}):
        for [name, field] in fields.items():
            self.npy_io.save_ndarray(grid, self.combine_filename(name, self.step_count), field.squeeze())
        for [name, scalar] in scalars.items():
            self.csv_io.append_value(grid, f"{name}", scalar)
        self.step_count += 1

    @staticmethod
    def combine_filename(name: str, step: int):
        return f"{name}_step{step}"

    def load_field(self, grid: Grid, name: str, step: int):
        return self.npy_io.load_ndarray(grid, self.combine_filename(name, step))

    def load_scalar_trajectory(self, grid: Grid, name: str):
        return self.csv_io.load_array(grid, name, self.step_count)


class NpyIO:

    root_path: pathlib.Path

    def __init__(self, root_path):
        self.root_path = pathlib.Path(root_path)

    def save_ndarray(self, grid: Grid, filename: str, field: Field):
        if grid.is_in_parallel:
            raise NotImplementedError()
        # np.savetxt(self.root_path / f"{filename}.csv", field.squeeze(), delimiter=";")
        np.save(self.root_path / f"{filename}.npy", field)

    def load_ndarray(self, grid: Grid, filename: str):
        if grid.is_in_parallel:
            raise NotImplementedError()
        # return np.loadtxt(self.root_path / f"{filename}.csv", delimiter=";")
        return np.load(self.root_path / f"{filename}.npy")


class CsvIO:

    root_path: pathlib.Path

    def __init__(self, root_path):
        self.root_path = pathlib.Path(root_path)

    def append_value(self, grid: Grid, filename: str, scalar: float):
        if grid.is_in_parallel:
            raise NotImplementedError()
        with open(self.root_path / f"{filename}.csv", "a", encoding="utf-8") as fp:
            fp.write(f"{scalar}")

    def load_array(self, grid: Grid, filename: str, nb_steps: int | None=None):
        if grid.is_in_parallel:
            raise NotImplementedError()
        arr = np.genfromtxt(self.root_path / f"{filename}.csv", delimiter="\n")
        return arr[0:nb_steps]


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
