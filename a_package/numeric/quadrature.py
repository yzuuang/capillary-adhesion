import numpy as np

from a_package.grid import Grid
from a_package.field import Field, field_element_axs


class Quadrature:
    """Quadrature for numerical approximating an integral."""

    nb_quad_pts: int
    quad_pt_coords: np.ndarray
    quad_pt_weights: np.ndarray

    def __init__(self, quad_pt_coords: np.ndarray, quad_pt_weights: np.ndarray):
        assert quad_pt_coords.shape[0] == quad_pt_weights.size
        assert np.isclose(quad_pt_weights.sum(), 1.)
        self.quad_pt_coords = quad_pt_coords
        self.quad_pt_weights = quad_pt_weights

    @property
    def nb_quad_pts(self):
        return np.size(self.quad_pt_weights)

    def integrate(self, grid: Grid, field: Field):
        # Due to regular grid, it is possible to factor out the element area
        element_sum = grid.element_area * np.sum(field, axis=field_element_axs)
        return np.einsum("s, cs-> c", self.quad_pt_weights, element_sum)

    def propag_integral_weight(self, grid: Grid, field: Field):
        return grid.element_area * np.einsum("s, cs...-> cs...", self.quad_pt_weights, field)


centroid_gaussian_quadrature = Quadrature(np.array([[1 / 3, 1 / 3], [2 / 3, 2 / 3]]), np.array([0.5, 0.5]))
"""Numerical quadrature with points located at the centroid of the two triangular elements of each pixel."""
