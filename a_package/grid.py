import typing
import functools
import operator

import numpy as np
import numpy.fft as fft
import muGrid

from a_package.communicator import communicator, factorize_closest


class Grid:
    """A discrete space in 2D."""

    lengths: tuple[float, ...]
    nb_elements: tuple[int, ...]
    subdomain_idx: tuple[int, ...]

    def __init__(
            self, lengths: typing.Sequence[float],
            nb_elements: typing.Sequence[int],
            nb_subdomains: typing.Sequence[int] | None = None,
            nb_ghost_layers: typing.Sequence[int] | None = None) -> None:
        self.lengths_global = lengths
        self.nb_elements_global = nb_elements
        self.element_sizes = [l / n for [l, n] in zip(lengths, nb_elements)]
        self.element_area = functools.reduce(operator.mul, self.element_sizes, 1.)

        if nb_subdomains is None:
            nb_subdomains = factorize_closest(communicator.size, self.nb_spatial_dims)
        # ghost layers, set all to 1 by default
        if nb_ghost_layers is None:
            nb_ghost_layers = [1] * self.nb_spatial_dims
        self.nb_ghost_layers = nb_ghost_layers

        self.decomposition = muGrid.CartesianDecomposition(
            communicator, nb_elements, nb_subdomains, nb_ghost_layers, nb_ghost_layers)
        self.lengths = tuple(d * n for [d, n] in zip(self.element_sizes, self.nb_elements))

    @property
    def nb_spatial_dims(self):
        return len(self.lengths_global)

    @property
    def nb_elements(self):
        return tuple(
            nb_pt - 2 * nb_ghost
            for [nb_pt, nb_ghost] in zip(self.decomposition.nb_subdomain_grid_pts, self.nb_ghost_layers))

    @property
    def subdomain_idx(self):
        return tuple(self.decomposition.subdomain_locations)

    def sync_subdomains(self, field: "muGrid.Field"):
        self.decomposition.communicate_ghosts(field)

    def form_index_axis(self, ax_index: int):
        return np.arange(self.nb_elements[ax_index])

    def form_index_mesh(self):
        return np.meshgrid(self.form_index_axis(0), self.form_index_axis(1))

    def form_nodal_axis(self, ax_index: int, with_endpoint: bool = False):
        d = self.element_sizes[ax_index]
        n = self.nb_elements[ax_index]
        if with_endpoint:
            n += 1
        return np.arange(n) * d

    def form_nodal_mesh(self, with_endpoint: bool = False):
        return np.meshgrid(self.form_nodal_axis(0, with_endpoint), self.form_nodal_axis(1, with_endpoint))

    def form_spectral_axis(self, ax_index: int):
        d = self.element_sizes[ax_index]
        n = self.nb_elements[ax_index]
        return (2 * np.pi) * fft.fftfreq(n, d)

    def form_spectral_mesh(self):
        return np.meshgrid(self.form_spectral_axis(0), self.form_spectral_axis(1))
