"""This file addresses the numerical details of computing model functions.

- Domain discretization & decomposition
- Basis functions to discretize fields (interpolation & gradient)
- Quadrature (numerical integration) of functions of fields
"""

import abc
import dataclasses as dc
import typing as _t

import numpy as np
import muGrid
from scipy.interpolate import RegularGridInterpolator


class muGridField_t(_t.Protocol):
    """Type hints to muGrid Field. For the ease of coding."""

    @property
    def name(self) -> str:
        """The name of the field, which is the unique identifier for the field."""

    @property
    def nb_components(self) -> int:
        """The number of components of the field quantity."""

    @property
    def p(self) -> np.ndarray[tuple[int, ...], np.dtype]:
        """Quantity values on the field, with #components and #sub-pts ravelled together."""

    @property
    def s(self) -> np.ndarray[tuple[int, ...], np.dtype]:
        """Quantity values on the field, with #components and #sub-pts exposed."""

    @property
    def sg(self) -> np.ndarray[tuple[int, ...], np.dtype]:
        """Quantity values on the field and border, with #components and #sub-pts exposed.

        This seems necessary for all "interpolation".
        """


class muGridConvolutionOperator_t(_t.Protocol):
    """Type hints to muGrid ConvolutionOperator. For the ease of coding."""

    @property
    def spatial_dim(self):
        """Spatial dimensions."""

    @property
    def nb_operators(self):
        """Number of operators (directions)."""

    @property
    def nb_quad_pts(self):
        """Number of quadrature points per pixel."""

    def apply(self, nodal_field: muGridField_t, quadrature_field: muGridField_t):
        """Apply the mapping, from nodal field to quadrature field."""

    def transpose(self, quadrature_field: muGridField_t, nodal_field: muGridField_t):
        """Apply the mapping sensitivity (is this dual map?), from quadrature field to nodal field.

        Matrix-wise, it is the transpose of the operator.
        """


@dc.dataclass(init=True)
class Grid:
    """A regular grid with periodic boundaries.

    - It handles the decomposition (via muGrid.CartesianDecomposition).
    - It supports "world" communication (via muGrid.Communicator).
    - It manages the fields defined on it (via muGrid.GlobalFieldCollection).
    """

    def __init__(
        self,
        length: list[float],
        nb_pixels: list[int],
        nb_subdivisions: list[int] = [],
        nb_ghost_layers: list[int] = [],
    ):
        self.length = length
        self.nb_pixels = nb_pixels
        if len(nb_subdivisions) == 0:
            nb_subdivisions = [1] * self.nb_dims
        self.nb_subdivisions = nb_subdivisions
        if len(nb_ghost_layers) == 0:
            nb_ghost_layers = [1] * self.nb_dims
        self.nb_ghost_layers = nb_ghost_layers

        # Value check
        assert (
            len(self.nb_pixels) == self.nb_dims
        ), f"Dimension incompatible. Your field is {self.nb_dims}D. But you specify a {len(self.nb_subdivisions)}D pixels."
        assert (
            len(self.nb_subdivisions) == self.nb_dims
        ), f"Dimension incompatible. Your field is {self.nb_dims}D. But you specify a {len(self.nb_subdivisions)}D subdivisions."
        assert (
            len(self.nb_ghost_layers) == self.nb_dims
        ), f"Dimension incompatible. Your field is {self.nb_dims}D. But you specify a {len(self.nb_subdivisions)}D ghost layers"

        # A unified "world" communicator with or without MPI
        try:
            from mpi4py import MPI

            self._communicator = muGrid.Communicator(MPI.COMM_WORLD)
        except ImportError:
            print("INFO: MPI is not installed, using stub implementation.")
            self._communicator = muGrid.Communicator()

        # A further value check
        assert (
            np.multiply.reduce(self.nb_subdivisions) <= self._communicator.size
        ), f"Too many subdivisions. Only {self._communicator.size} nodes are available."

        # Domain decomposition
        self._decomposition = muGrid.CartesianDecomposition(
            self._communicator,
            self.nb_pixels,
            self.nb_subdivisions,
            self.nb_ghost_layers,
            self.nb_ghost_layers,
        )

        # Field manager
        self._field_collection: muGrid.GlobalFieldCollection = self._decomposition.collection

    @property
    def nb_dims(self):
        return len(self.length)

    @property
    def pixel_length(self):
        return [l / nb for (l, nb) in zip(self.length, self.nb_pixels)]

    @property
    def pixel_area(self):
        return np.multiply.reduce(self.pixel_length)

    @property
    def nb_pixels_in_section(self):
        return [
            nb_pts - 2 * nb_layers
            for (nb_pts, nb_layers) in zip(
                self._decomposition.nb_subdomain_grid_pts, self.nb_ghost_layers
            )
        ]

    @property
    def pixel_indices_in_section(self):
        return self._decomposition.icoords

    def add_sub_pt_scheme(self, tag: str, nb_sub_pts: int):
        self._field_collection.set_nb_sub_pts(tag, nb_sub_pts)

    def real_field(self, name: str, nb_components: int, sub_pts_tag: str = "pixel"):
        return Field(self._field_collection.real_field(name, nb_components, sub_pts_tag), self)

    def communicate_ghosts(self, field: muGridField_t):
        self._decomposition.communicate_ghosts(field)

    def sum(self, value: _t.Union[int, float, np.ndarray]):
        return self._communicator.sum(value)

    def get_world_communicator(self):
        return self._communicator


def factorize_closest(value: int, nb_ints: int):
    """Find the maximal combination of nb_ints integers whose product is less or equal to value."""
    nb_subdivisions = []
    for root_degree in range(nb_ints, 0, -1):
        max_divisor = int(value ** (1 / root_degree))
        nb_subdivisions.append(max_divisor)
        value //= max_divisor
    return nb_subdivisions


class Field:
    """A discrete field that takes care of the communication with other sections.

    Implemented as a thin wrapper around muGrid.Field.
    """

    def __init__(self, section: muGridField_t, grid: Grid):
        self.section = section
        """Each processor only owns its own section of the field data, hence the name."""
        self._grid = grid

    @property
    def data(self):
        return self.section.s

    @data.setter
    def data(self, value):
        self.section.s = value
        self._grid.communicate_ghosts(self.section)

    def sum(self):
        """Sum up values of all locations. While maintain components and subpoints.z"""
        # First sum inside the section, then sum over all sections.
        # Spatial dimensions are the last axes of the array.
        return self._grid.sum(np.sum(self.section.s, axis=tuple(range(-self._grid.nb_dims, 0))))

    def roll(self, shift: int, axis: int):
        """Roll the field values."""
        # Roll it one layer per time, as there is only one layer of ghosts
        for _ in range(shift):
            # Spatial dimensions are the last axes of the array.
            self.section.s = np.roll(self.section.s, np.sign(shift), -self._grid.nb_dims + axis)
            self._grid.communicate_ghosts(self.section)

    def bind_mapping(self, operator: muGridConvolutionOperator_t, result: "Field"):
        def evaluate():
            operator.apply(self.section, result.section)
            return result.section.s

        return evaluate

    def bind_mapping_sensitivity(self, operator: muGridConvolutionOperator_t, result: "Field"):
        def evaluate():
            operator.transpose(self.section, result.section)
            return result.section.s

        return evaluate


@dc.dataclass(init=True, frozen=True)
class Quadrature:
    """Quadrature for numerical approximating an integral."""

    tag: str
    nb_quad_pts: int
    nb_dims: int
    quad_pt_offset: np.ndarray
    quad_pt_weights: np.ndarray

    def integrate(self, integrand: np.ndarray, grid: Grid):
        # Sum over quadrature points (weighted) and over all pixels (equally)
        local_sum = np.einsum("cs..., s-> c...", integrand, self.quad_pt_weights)
        local_sum = grid.pixel_area * np.sum(local_sum, axis=tuple(range(-self.nb_dims, 0)))
        return grid.sum(local_sum)


centroid_quadrature = Quadrature(
    tag="centroid",
    nb_quad_pts=2,
    nb_dims=2,
    quad_pt_offset=np.array([[1 / 3, 1 / 3], [2 / 3, 2 / 3]]),
    quad_pt_weights=np.array([0.5, 0.5]),
)
"""Numerical intergration with quadrature points located at the centroid of the two triangles of
each pixel. It provides discrete operators for interpolation and gradient.
"""


class CubicSpline:

    # FIXME (when sliding?):
    # get the slice with left / right extra 1 grid. so it can interpolate the sliding part?

    def __init__(self, grid: Grid):
        self.nb_dims = grid.nb_dims
        self.sample_location_in_each_axis = [
            # Scipy only accepts the indices to be strictly increasing
            np.arange(-nb_layers, nb_pts + nb_layers)
            for (nb_pts, nb_layers) in zip(grid.nb_pixels_in_section, grid.nb_ghost_layers)
        ]
        self.nb_pixels = grid.nb_pixels_in_section
        self.pixel_origin_location = grid.pixel_indices_in_section

    def sample(self, field: Field):
        assert field.section.nb_components == 1
        data = np.squeeze(field.section.sg)
        self.interpolator = RegularGridInterpolator(self.sample_location_in_each_axis, data, method="cubic")

    def interpolate(self, offset_in_pixel: np.ndarray):
        """
        - offset_in_pixel: array of 2D coordinates located inside a (1,1) square.
        """
        [nb_sub_pts, nb_dims] = np.shape(offset_in_pixel)
        assert self.nb_dims == nb_dims

        result = np.empty([nb_sub_pts, *self.nb_pixels])
        for idx, loc in enumerate(offset_in_pixel):
            # Sum arrays with shape (nb_dims, nb_x, nb_y) and (nb_dims,)
            interp_location = self.pixel_origin_location + np.expand_dims(loc, axis=tuple(range(-self.nb_dims, 0)))
            result[idx] = self.interpolator(tuple(interp_location))
        # Keep the convention that 0-axis covers components
        return np.expand_dims(result, axis=0)


class Linear2DFiniteElementInPixel:
    """A unit pixel discretized with linear finite element basis.

    The vertices of the pixel are (0,0), (1,0), (0,1), (1,1). It is divided into two triangles by
    the line connecting vertices (1,0) and (0,1), x_1 + x_2 = 1. The triangle with (0,0) vertice is
    the "lower triangle", the other is the "upper triangle".
    """

    def create_field_value_approximation(
        self, offset_in_pixel: np.ndarray
    ) -> muGridConvolutionOperator_t:
        offset = [1, 1]
        pixel_operator = self.get_value_interpolation_coefficients(offset_in_pixel)
        return muGrid.ConvolutionOperator(offset, pixel_operator)

    def create_field_gradient_approximation(
        self, offset_in_pixel: np.ndarray, dx1: float, dx2: float
    ) -> muGridConvolutionOperator_t:
        offset = [1, 1]
        pixel_operator = self.get_gradient_interpolation_coefficients(offset_in_pixel)
        pixel_operator[0] = pixel_operator[0] / dx1
        pixel_operator[1] = pixel_operator[1] / dx2
        return muGrid.ConvolutionOperator(offset, pixel_operator)

    def get_value_interpolation_coefficients(self, offset_in_pixel):
        nb_sub_pts = np.size(offset_in_pixel, axis=0)
        res = np.empty([1, nb_sub_pts, 2, 2])
        for i, (x1, x2) in enumerate(offset_in_pixel):
            if x1 + x2 < 1:
                # Lower triangle
                res[:,i] = self.lower_triangle_shape_function(x1, x2)
            else:
                # Upper triangle
                res[:,i] = self.upper_triangle_shape_function(x1, x2)
        return res

    @staticmethod
    def lower_triangle_shape_function(x1, x2):
        return [
            [1 - x1 - x2, x2],
            [x1, 0],
        ]

    @staticmethod
    def upper_triangle_shape_function(x1, x2):
        return [
            [0, 1 - x1],
            [1 - x2, x1 + x2 - 1],
        ]

    def get_gradient_interpolation_coefficients(self, offset_in_pixel):
        res = np.empty([2, np.size(offset_in_pixel, axis=0), 2, 2])
        for i, (x1, x2) in enumerate(offset_in_pixel):
            if x1 + x2 < 1:
                # Lower triangle
                res[:, i] = self.lower_triangle_shape_function_gradient(x1, x2)
            else:
                # Upper triangle
                res[:, i] = self.uppper_triangle_shape_function_gradient(x1, x2)
        return res

    @staticmethod
    def lower_triangle_shape_function_gradient(x1, x2):
        return [
            [
                [-1, 0],
                [1, 0],
            ],
            [
                [-1, 1],
                [0, 0],
            ],
        ]

    @staticmethod
    def uppper_triangle_shape_function_gradient(x1, x2):
        return [
            [
                [0, -1],
                [0, 1],
            ],
            [
                [0, 0],
                [-1, 1],
            ],
        ]
