"""
a common IO for data exchange between simulation and visualisation.
"""

import sys
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum

import numpy as np

from a_package.grid import Grid
from a_package.field import Field
from a_package.data_io import NpyIO


class SimulationIO:

    grid: Grid
    io: NpyIO

    def __init__(self, grid, store_dir) -> None:
        self.grid = grid
        self.io = NpyIO(store_dir)

    def save_constant(self, fields: dict[str, Field]={}, single_values: dict[str, float]={}):
        for [name, field] in fields.items():
            self.io.save_field(self.grid, name, field)

        for [name, value] in single_values.items():
            self.io.save_value_array(name, np.array([value]))

    def load_constant(self, field_names: list[str]=[], single_value_names: list[str]=[]):
        result = {}

        # For field, each step has its own file
        for name in field_names:
            result[name] = self.io.load_field(self.grid, name)

        # For single values, all steps shares one file
        for name in single_value_names:
            [result[name]] = self.io.load_value_array(name)

        return result

    def save_step(self, index: int, fields: dict[str, Field]={}, single_values: dict[str, float]={}):
        # For field, each step has its own file
        for [name, field] in fields.items():
            self.io.save_field(self.grid, format_filename(name, index), field)

        # For single values, all steps share one file
        for [name, value] in single_values.items():
            array = self.io.load_value_array(name)
            try:
                array[index] = value
            except IndexError:
                if index == array.size:
                    array = np.append(array, value)
                else:
                    raise ValueError()
            self.io.save_value_array(name, array)

    def load_step(self, index: int, field_names: list[str]=[], single_value_names: list[str]=[]):
        result = {}

        # For field, each step has its own file
        for name in field_names:
            result[name] = self.io.load_field(self.grid, format_filename(name, index))

        # For single values, all steps shares one file
        for name in single_value_names:
            result[name] = self.io.load_value_array(name)[index]

        return result

    def save_trajectory(self, fields: dict[str, list[Field]]={}, single_values: dict[str, np.ndarray]={}):
        result = {}
        # For field, every step is saved in one file. 
        for [name, traj] in fields.items():
            array = FieldArray(self.grid, self.io, name)
            for index in range(len(traj)):
                array[index] = traj[index]
        # For single values, a trajectory is saved as one file
        for [name, traj] in single_values.items():
            result[name] = self.io.save_value_array(name, traj)

    def load_trajectory(self, field_names: list[str]=[], single_value_names: list[str]=[]):
        result = {}
        # For field, every step is saved in one file. 
        for name in field_names:
            result[name] = FieldArray(self.grid, self.io, name)
        # For single values, a trajectory is saved as one file
        for name in single_value_names:
            result[name] = self.io.load_value_array(name)
        return result


class FieldArray:
    """A helper calss. It mimics an array but actually read / write corresponding files."""

    grid: Grid    
    io: NpyIO
    name: str

    def __init__(self, grid, io, name) -> None:
        self.grid = grid
        self.io = io
        self.name = name

    def __getitem__(self, index: int):
        return self.io.load_field(self.grid, format_filename(self.name, index))

    def __setitem__(self, index: int, value):
        self.io.save_field(self.grid, format_filename(self.name, index), value)


def format_filename(name: str, index: int):
    return f"{name}--{index}"


class Term(StrEnum):
    upper_solid = "upper"
    lower_solid = "lower"
    separation = "separation"
    pressure = "pressure"
    volume = "volume"
    gap = "gap"
    phase = "phase"
    energy = "energy"
    perimeter = "perimeter"
    phase_init = "phase_init"
    pressure_init = "pressure_init"
