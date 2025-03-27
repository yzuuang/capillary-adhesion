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


@dc.dataclass(init=True)
class Grid:
    """A regular grid with periodic boundaries and is parallelized."""

    spacing: float
    nb_pixels: list[int]
    divising_dims: int

    def __post_init__(self):
        # A "unified" communicator with or without MPI
        try:
            from mpi4py import MPI

            communicator = muGrid.Communicator(MPI.COMM_WORLD)
        except ImportError:
            print("INFO: MPI is not installed, using stub implementation.")
            communicator = muGrid.Communicator()

        # Decompose the domain
        # FIXME: it doesn't garauntee all processes have a section. Maybe should fix in muGrid.
        nb_subdivisions = max_integer_combo(communicator.size, self.divising_dims)
        nb_ghosts_left = [1] * self.divising_dims
        nb_ghosts_right = [1] * self.divising_dims
        decomposition = muGrid.CartesianDecomposition(
            communicator, self.nb_pixels, nb_subdivisions, nb_ghosts_left, nb_ghosts_right
        )

        # One object for things that should be handled by each process itself
        self.section = Section(
            decomposition.collection,
            decomposition.nb_subdomain_grid_pts,
            decomposition.global_coords,
            # FIXME: is this a general problem of muGrid periodic boundary in parallel?
            [
                n - l - r
                for (n, l, r) in zip(
                    decomposition.nb_subdomain_grid_pts, nb_ghosts_left, nb_ghosts_right
                )
            ],
        )

        # One object for things that require cooperation of processes
        self.coordinator = Coordinator(communicator, decomposition, nb_ghosts_left, nb_ghosts_right)
        """NOTE: Although decomposition has a different communicator, it is created by calling 
        MPI_Cart_create with reoder=false. The ranks of processes in two communicators are the same.
        """


def max_integer_combo(value: int, nb_ints: int):
    """Find the maximal combination of nb_ints integers whose product is less or equal to value."""
    nb_subdivisions = []
    for root_degree in range(nb_ints, 0, -1):
        max_divisor = int(value ** (1 / root_degree))
        nb_subdivisions.append(max_divisor)
        value //= max_divisor
    return nb_subdivisions


class Field2D(_t.Protocol):
    """Some type hints for muGrid Field."""

    @property
    def name(self) -> str:
        """The name of the field, which is the unique identifier for the field."""

    @property
    def nb_components(self) -> int:
        """The number of components of the field quantity."""

    @property
    def p(self) -> np.ndarray[tuple[int, int, int], np.dtype]:
        """Quantity values on the field, with #components and #sub-pts ravelled together."""

    @property
    def s(self) -> np.ndarray[tuple[int, int, int, int], np.dtype]:
        """Quantity values on the field, with #components and #sub-pts exposed."""


@dc.dataclass(init=True)
class Coordinator:
    """It is aware that a domain is deomposed of several subdomains. So it must distinguish whether
    pixels are ghost-buffers.
    """

    communicator: muGrid.Communicator
    decomposition: muGrid.CartesianDecomposition
    nb_ghosts_left: list[int]
    nb_ghosts_right: list[int]

    @property
    def non_ghost(self):
        """Indexing slices to exclude ghost buffers"""
        return (
            Ellipsis,
            *tuple(
                slice(nb_l, -nb_r)
                for (nb_l, nb_r) in zip(self.nb_ghosts_left, self.nb_ghosts_right)
            ),
        )

    def update(self, field: Field2D):
        """Update the field. Fill valeus to ghost buffers."""
        self.decomposition.communicate_ghosts(field.name)

    def roll(self, field: Field2D, shift: int, axis: int):
        """Roll the field values.

        This is a speical implementation for nb_ghosts_left = nb_ghosts_right = [1, 1]
        """
        nb_axes_front = 2
        for _ in range(shift):
            self.decomposition.communicate_ghosts(field.name)
            field.s = np.roll(field.s, np.sign(shift), nb_axes_front + axis)

    # def sum(self, field: np.ndarray):
    #     """Sum over the region, omitting ghost buffers.

    #     NOTE: for energy functional, it sums the field defined on quadrature points.
    #     Omitting the left ghosts because the pixels are repeated in other subdomains;
    #     Omitting the right ghosts because the periodic boundary is not hold for a subdomain,
    #     those pixels don't exist in the domain.
    #     """
    #     local_sum = np.sum(field[self.non_ghost], axis=(-1, -2))
    #     return self.communicator.sum(local_sum)

    # FIXME: in parallelized code, one shall never explicitly ask for the whole domain.
    # def gather(self, field: np.ndarray):
    #     """Gather over the region, omitting ghost buffers.

    #     NOTE: for phase field, it gathers the nodal field. It is obvious to omit all ghosts.

    #     NOTE: for the jacobian of the energy functional, it gathers the nodal field.
    #     Omitting both left and right ghosts because the rest nodes are "self-contained"
    #     inside the subdomain, that is, the contribution from quadrature points to nodal
    #     points is completely computed. (A counter example is at the left ghosts, these
    #     nodes has the contribution from quadrature points located in the pixels inside
    #     this subdomain, but not the ones located in the pixels of the neighbouring subdomain.)
    #     """
    #     contiguous_copy = np.copy(np.squeeze(field[..., *self.non_ghost]), order='F')
    #     print("GATHER", contiguous_copy.flags)
    #     return communicator.gather(contiguous_copy)


@dc.dataclass(init=True)
class Section:
    """It views itself merely as a collection of pixels. So it treats all pixels the same."""

    pixel_collection: muGrid.GlobalFieldCollection
    nb_pixels: list[int]
    global_coords: np.ndarray
    nb_effective_pixels: list[int]

    # def set_nb_sub_pts(self, *args):
    #     """
    #     - tag: str
    #     - nb_sub_pts: int
    #     """
    #     self.pixel_collection.set_nb_sub_pts(*args)

    # def real_field(self, *args) -> Field2D:
    #     """
    #     - unique_name: str
    #     - nb_components: int
    #     - sub_pt_tag: str = "pixel"
    #     """
    #     return self.pixel_collection.real_field(*args)


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
        self.pixel_size = grid.spacing
        self.pixel_area = grid.spacing**2
        self.coordinator = grid.coordinator
        grid.section.pixel_collection.set_nb_sub_pts(self.tag, self.nb_quad_pts)

    def integrate2D(self, integrand: np.ndarray):
        # Sum (weighted) over quadrature points
        pixel_values = np.einsum("csxy, s-> cxy", integrand, self.quad_pt_weights)
        local_sum = np.sum(pixel_values[self.coordinator.non_ghost], axis=(-1,-2))
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


@dc.dataclass(init=True)
class FieldOp:
    """Ad-hoc the ConvolutionOperator with more attributes."""

    op: muGrid.ConvolutionOperator
    field_in: Field2D
    field_out: Field2D
    field_out_back: Field2D
    field_out_back_in: Field2D

    # def apply(self, *args):
    #     """Apply the mapping.

    #     - field_in: Field2d
    #     - field_out: Field2d
    #     """
    #     self.op.apply(*args)

    # def transpose(self, *args):
    #     """Apply the inverse mapping. Matrix-wise, the operator is transposed.

    #     - field_in: Field2d
    #     - field_out: Field2d
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
        print(f"In sample, value has shape {value.shape}, has values\n {value}")
        self.field_sample.p = np.expand_dims(value, axis=0)
        print(f"Communicate ghost")
        self.coordinator.update(self.field_sample)
        print(f"Finish communicating.")
        print(f"Now it has values\n {self.field_sample.p.squeeze()}")

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
                np.reshape(
                    self.interpolate_coefficients(quadrature.quad_pt_local_coords),
                    shape=(-1, nb_inputs),
                    order="F",
                ),
                conv_pts_shape,
                nb_pixelnodal_pts,
                nb_sub_pts,
                nb_operators_interpolation,
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
                np.reshape(
                    self.gradient_coefficients(quadrature.quad_pt_local_coords)
                    / quadrature.pixel_size,
                    shape=(-1, nb_inputs),
                    order="F",
                ),
                conv_pts_shape,
                nb_pixelnodal_pts,
                nb_sub_pts,
                nb_operators_gradient,
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
