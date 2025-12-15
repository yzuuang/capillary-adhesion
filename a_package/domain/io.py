"""
IO for persisting fields and arrays.

This module handles low-level persistence. When Grid supports domain decomposition
for parallel execution, this module will handle local/global field mapping.
"""

import pathlib

import numpy as np

from .grid import Grid
from .field import Field, adapt_shape


class NpyIO:
    """
    NumPy-based persistence for fields and arrays.

    When parallel support is added, this class will coordinate
    saving/loading of distributed fields across ranks.
    """

    root_path: pathlib.Path

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
