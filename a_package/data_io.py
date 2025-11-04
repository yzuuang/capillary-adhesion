"""
Store data in a self-describing manner.
"""

import pathlib

import numpy as np

from a_package.grid import Grid
from a_package.field import Field, adapt_shape


class NpyIO:

    root_path: pathlib.Path
    extension: str

    def __init__(self, root_path):
        self.root_path = pathlib.Path(root_path)

    @property
    def extension(self):
        return "npy"

    def load_field(self, grid: Grid, name: str):
        try:
            return np.load(self.root_path / f"{name}.{self.extension}", allow_pickle=False)
        except FileNotFoundError:
            return adapt_shape(np.atleast_2d([]))

    def save_field(self, grid: Grid, name: str, field: Field):
        np.save(self.root_path / f"{name}.{self.extension}", field)

    def load_value_array(self, name: str):
        try:
            return np.load(self.root_path / f"{name}.{self.extension}", allow_pickle=False)
        except FileNotFoundError:
            return np.array([])

    def save_value_array(self, name: str, array: np.ndarray):
        np.save(self.root_path / f"{name}.{self.extension}", array)
