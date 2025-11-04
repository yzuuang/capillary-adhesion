"""
Store data in a self-describing manner.
"""

import pathlib

import numpy as np
import NuMPI.IO

from a_package.grid import Grid
from a_package.field import Field, adapt_shape, field_component_ax, field_sub_pt_ax
from a_package.communicator import MPI, communicator


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
            if MPI:
                field = NuMPI.IO.load_npy(
                    self.root_path / f"{name}.{self.extension}", tuple(grid.subdomain_base),
                    tuple(grid.nb_elements))
            else:
                field = np.load(self.root_path / f"{name}.{self.extension}", allow_pickle=False)
        except FileNotFoundError:
            field = np.atleast_2d([])
        return adapt_shape(field)

    def save_field(self, grid: Grid, name: str, field: Field):
        if MPI:
            # FIXME: it cannot save all components / subpoints.
            NuMPI.IO.save_npy(
                self.root_path / f"{name}.{self.extension}", np.ascontiguousarray(field.squeeze()),
                tuple(grid.subdomain_base),
                tuple(grid.nb_elements_global))
        else:
            np.save(self.root_path / f"{name}.{self.extension}", field)

    def load_value_array(self, name: str):
        try:
            return np.load(self.root_path / f"{name}.{self.extension}", allow_pickle=False)
        except FileNotFoundError:
            return np.array([])

    def save_value_array(self, name: str, array: np.ndarray):
        if MPI and communicator.rank != 0:
            return
        np.save(self.root_path / f"{name}.{self.extension}", array)
