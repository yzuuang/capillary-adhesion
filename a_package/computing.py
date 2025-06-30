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
        return self._decomposition.nb_subdomain_grid_pts

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
        return self._grid.sum(np.sum(self.section.s, axis=np.arange(-self._grid.nb_dims, 0)))

    def roll(self, shift: int, axis: int):
        """Roll the field values."""
        # Roll it one layer per time, as there is only one layer of ghosts
        for _ in range(shift):
            # Spatial dimensions are the last axes of the array.
            self.section.s = np.roll(
                self.section.s, np.sign(shift), -self._grid.nb_dims + axis
            )
            self._grid.communicate_ghosts(self.section)


class CubicSpline:

    # FIXME (when sliding?):
    #
    # the global height, pad right for periodicity, interpolate.
    #
    # get the slice with left / right extra 1 grid. so it can interpolate sliding part?

    def __init__(self, grid: Grid, data: np.ndarray):
        self.nb_pixels = grid.section.nb_pixels
        # Due to periodic boundary, there is one more "hidden" grid point in the end, so plus 1
        # for axes in sampling data
        self.axis_pt_locs = tuple(np.arange(nb_pts + 1) for nb_pts in self.nb_pixels)
        # But we don't explicitly save the value at the end, so no plus 1 for grid in interpolation
        self.pixel_locs = np.stack(
            np.meshgrid(*(np.arange(nb_pts) for nb_pts in self.nb_pixels)), axis=0
        )
        self.sample(data)

    def sample(self, data: np.ndarray):
        grid_dim = len(self.axis_pt_locs)
        data = data.squeeze(axis=tuple(range(data.ndim - grid_dim)))
        pad_size = [(0, 1)] * grid_dim
        self.interpolator = RegularGridInterpolator(
            self.axis_pt_locs, np.pad(data, pad_size, mode="wrap"), method="cubic"
        )

    def interpolate(self, local_coords: np.ndarray):
        """
        - local_coords: array of 2D coordinates located inside a (1,1) square.
        """
        target_pt_locs = (
            self.pixel_locs[np.newaxis, ...] + local_coords[..., np.newaxis, np.newaxis]
        )
        result = np.empty((np.size(local_coords, axis=0), *self.nb_pixels))
        for idx, pt_loc in enumerate(target_pt_locs):
            result[idx] = self.interpolator(tuple(pt_loc))
        return np.expand_dims(result, axis=0)


class Quadrature(abc.ABC):
    """Base class for computing quadrature over a region."""

    @property
    @abc.abstractmethod
    def tag(self) -> str:
        """A unique tag for sub-pt scheme."""

    @property
    @abc.abstractmethod
    def nb_quad_pts(self) -> int:
        """# quadrature points."""

    @property
    @abc.abstractmethod
    def quad_pt_local_coords(self) -> np.ndarray:
        """Location of quadrature points, normalized in a (1,1) unit square."""

    @property
    @abc.abstractmethod
    def quad_pt_weights(self) -> np.ndarray:
        """Weights of each quadrature points. The sum weights shall equal to one."""

    def __init__(self, grid: Grid):
        self.pixel_size = grid.length
        self.pixel_area = grid.length**2
        self.coordinator = grid.coordinator
        grid.section.pixel_collection.set_nb_sub_pts(self.tag, self.nb_quad_pts)

    def integrate2D(self, integrand: np.ndarray):
        # Sum (weighted) over quadrature points
        pixel_values = np.einsum("csxy, s-> cxy", integrand, self.quad_pt_weights)
        local_sum = np.sum(pixel_values[self.coordinator.non_ghost], axis=(-1, -2))
        return self.pixel_area * self.coordinator.communicator.sum(local_sum)



class CentroidQuadrature(Quadrature):
    """Numerical intergration with quadrature points located at the centroid of the two triangles of
    each pixel. It provides discrete operators for interpolation and gradient.
    """

    @property
    def tag(self):
        return "centroid"

    @property
    def nb_quad_pts(self):
        return 2

    @property
    def quad_pt_local_coords(self):
        return np.array([[1 / 3, 1 / 3], [2 / 3, 2 / 3]])

    @property
    def quad_pt_weights(self):
        return np.array([0.5] * 2)


class ConvolutionOperator:
    pass

@dc.dataclass(init=True)
class FieldOp:
    """Ad-hoc the ConvolutionOperator with more attributes."""

    op: muGrid.ConvolutionOperator
    field_in: muGridField_t
    field_out: muGridField_t
    field_out_back: muGridField_t
    field_out_back_in: muGridField_t

    # def apply(self, *args):
    #     """Apply the mapping.

    #     - field_in: muGridField_t
    #     - field_out: muGridField_t
    #     """
    #     self.op.apply(*args)

    # def transpose(self, *args):
    #     """Apply the inverse mapping. Matrix-wise, the operator is transposed.

    #     - field_in: muGridField_t
    #     - field_out: muGridField_t
    #     """
    #     self.op.transpose(*args)


class LinearFiniteElement:
    """A unit pixel discretized with linear finite element basis.

    The vertices of the pixel are (0,0), (1,0), (0,1), (1,1). It is divided into two triangles by
    the line connecting vertices (1,0) and (0,1), x_1 + x_2 = 1. The triangle with (0,0) vertice is
    the "lower triangle", the other is the "upper triangle".
    """

    def __init__(self, grid: Grid, data: np.ndarray):
        self.coordinator = grid.coordinator
        self.collection = grid.section.pixel_collection
        try:
            self.nb_components_input = np.size(data, axis=-3)
        except IndexError:
            self.nb_components_input = 1
        self.field_sample = self.collection.real_field("input", self.nb_components_input)

    def sample(self, value: np.ndarray):
        self.field_sample.p = value
        self.coordinator.update(self.field_sample)

    def apply_operators(self, operators: list[FieldOp]) -> list[np.ndarray]:
        result = []
        for operator in operators:
            operator.op.apply(operator.field_in, operator.field_out)
            result.append(operator.field_out.s)
        return result

    def apply_transposed_to_values(self, values: list[np.ndarray], operators: list[FieldOp]):
        result = 0
        for value, operator in zip(values, operators):
            operator.field_out_back.s = value
            operator.op.transpose(operator.field_out_back, operator.field_out_back_in)
            result += operator.field_out_back_in.p

        # Sum over sensitivities
        return np.squeeze(result, axis=0)

    def setup_operators(self, quadrature: Quadrature):
        nb_sub_pts = np.size(quadrature.quad_pt_local_coords, axis=0)
        conv_pts_shape = [2, 2]
        nb_pixelnodal_pts = 1
        nb_inputs = np.multiply.reduce(conv_pts_shape) * nb_pixelnodal_pts

        # Interpolation operator
        nb_operators_interpolation = 1
        self.interpolation = FieldOp(
            muGrid.ConvolutionOperator(
                conv_pts_shape,
                np.reshape(
                    self.interpolate_coefficients(quadrature.quad_pt_local_coords),
                    shape=(-1, nb_inputs),
                    order="F",
                ),
                # nb_pixelnodal_pts,
                # nb_sub_pts,
                # nb_operators_interpolation,
            ),
            self.field_sample,
            self.collection.real_field(
                "field_interpolation",
                nb_operators_interpolation * self.nb_components_input,
                quadrature.tag,
            ),
            self.collection.real_field(
                "field_interpolation_back",
                nb_operators_interpolation * self.nb_components_input,
                quadrature.tag,
            ),
            self.collection.real_field("field_interpolation_back_in", self.nb_components_input),
        )

        # Gradient operator
        nb_operators_gradient = 2
        self.gradient = FieldOp(
            muGrid.ConvolutionOperator(
                conv_pts_shape,
                np.reshape(
                    self.gradient_coefficients(quadrature.quad_pt_local_coords)
                    / quadrature.pixel_size,
                    shape=(-1, nb_inputs),
                    order="F",
                ),
                # nb_pixelnodal_pts,
                # nb_sub_pts,
                # nb_operators_gradient,
            ),
            self.field_sample,
            self.collection.real_field(
                "field_gradient",
                nb_operators_gradient * self.nb_components_input,
                quadrature.tag,
            ),
            self.collection.real_field(
                "field_gradient_back",
                nb_operators_gradient * self.nb_components_input,
                quadrature.tag,
            ),
            self.collection.real_field("field_gradient_back_in", self.nb_components_input),
        )

    def interpolate_coefficients(self, local_coords):
        res = np.empty([np.size(local_coords, axis=0), 2, 2])
        for i, (x1, x2) in enumerate(local_coords):
            if x1 + x2 < 1:
                # Lower triangle
                res[i] = self.shape_lower_triangle(x1, x2)
            else:
                # Upper triangle
                res[i] = self.shape_upper_triangle(x1, x2)
        return res

    @staticmethod
    def shape_lower_triangle(x1, x2):
        return [
            [1 - x1 - x2, x2],
            [x1, 0],
        ]

    @staticmethod
    def shape_upper_triangle(x1, x2):
        return [
            [0, 1 - x1],
            [1 - x2, x1 + x2 - 1],
        ]

    def gradient_coefficients(self, local_coords):
        res = np.empty([2, np.size(local_coords, axis=0), 2, 2])
        for i, (x1, x2) in enumerate(local_coords):
            if x1 + x2 < 1:
                # Lower triangle
                res[:, i] = self.slope_lower_triangle(x1, x2)
            else:
                # Upper triangle
                res[:, i] = self.slope_upper_triangle(x1, x2)
        return res

    @staticmethod
    def slope_lower_triangle(x1, x2):
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
    def slope_upper_triangle(x1, x2):
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
