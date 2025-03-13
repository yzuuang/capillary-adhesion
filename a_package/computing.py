"""This file addresses the numerical details of computing model functions.

- Discretization of the region
- Basis functions to discretize fields (quantities defined on the region)
- Quadrature (numerical integration) of functions of fields
- Domain decomposition
"""

import abc
import dataclasses as dc
import typing as _t

import numpy as np
import muGrid

from SurfaceTopography.Uniform.Interpolation import Bicubic


# The "unified" communicator
try:
    from mpi4py import MPI

    communicator = muGrid.Communicator(MPI.COMM_WORLD)
except ImportError:
    communicator = muGrid.Communicator()


@dc.dataclass(init=True)
class Region:
    """A region on which a field can be defined.

    It is discretized into a regular grid and decomposed into several subdivisions.
    It also handles communication between subdomains.
    """

    nb_domain_grid_pts: list[int]
    grid_spacing: float

    @property
    def spatial_dims(self):
        return len(self.nb_domain_grid_pts)

    def __post_init__(self):
        # Find an approproiate divising scheme
        # FIXME: when the residual is not 0.
        nb_processes = communicator.size
        max_divisor = int(nb_processes**0.5)
        nb_subdivisions = [max_divisor, nb_processes // max_divisor]

        # Add ghost buffers on both side
        nb_ghosts_left = [1] * self.spatial_dims
        nb_ghosts_right = [1] * self.spatial_dims

        # Index to exclude ghost buffers
        self.non_ghost = tuple(
            slice(nb_l, -nb_r) for (nb_l, nb_r) in zip(nb_ghosts_left, nb_ghosts_right)
        )

        # Domain decomposition
        self.decomposition = muGrid.CartesianDecomposition(
            communicator,
            self.nb_domain_grid_pts,
            nb_subdivisions,
            nb_ghosts_left,
            nb_ghosts_right,
        )
        """NOTE: Although decomposition has a different communicator, it is created
        by calling MPI_Cart_create with reoder=false. The ranks of processes in 
        two communicators are the same.
        """

    @property
    def collection(self):
        """The field manager (muGrid.GlobalFieldCollection)."""
        return self.decomposition.collection

    def update(self, field_name: str):
        """Update field values at ghost buffers."""
        self.decomposition.communicate_ghosts(field_name)

    def sum(self, field: np.ndarray):
        """Sum over the region, omitting ghost buffers.

        NOTE: for energy functional, it sums the field defined on quadrature points.
        Omitting the left ghosts because the pixels are repeated in other subdomains;
        Omitting the right ghosts because the periodic boundary is not hold for a subdomain,
        those pixels don't exist in the domain.
        """
        return communicator.sum(field[self.non_ghost])

    def gather(self, field: np.ndarray):
        """Gather over the region, omitting ghost buffers.

        NOTE: for phase field, it gathers the nodal field. It is obvious to omit all ghosts.

        NOTE: for the jacobian of the energy functional, it gathers the nodal field.
        Omitting both left and right ghosts because the rest nodes are "self-contained"
        inside the subdomain, that is, the contribution from quadrature points to nodal
        points is completely computed. (A counter example is at the left ghosts, these
        nodes has the contribution from quadrature points located in the pixels inside
        this subdomain, but not the ones located in the pixels of the neighbouring subdomain.)
        """
        return communicator.gather(field[self.non_ghost])

    @property
    def in_local_subdomain(self):
        """The index to get the slice of the subdomain in a field of the whole domain."""
        return tuple(
            slice(idx, idx + nb)
            for (idx, nb) in zip(
                self.decomposition.subdomain_locations, self.decomposition.nb_subdomain_grid_pts
            )
        )

    @property
    def global_coords(self):
        """The global coordinates of grid points in local subdomain"""
        return self.decomposition.global_coords


class Field(_t.Protocol):
    """Use type hints as a reminder that it is a muGrid Field."""

    @property
    def name(self) -> str:
        """The name of the field, which is the unique identifier for the field."""

    @property
    def nb_components(self) -> int:
        """The number of components of the field quantity."""

    @property
    def s(self) -> np.ndarray[tuple[int, ...], np.dtype]:
        """Quantity values on the field.

        NOTE: The model should be agonistic to discretization details, such as #sub-points, size of 
        regular grid, etc. So the return type marks its first dimension (len == #components).
        """


class ConvOp(_t.Protocol):
    """Use type hints as a reminder that it is a muGrid Convolution Operator."""

    name: str
    """A unique name. It is useful to name the output field properly in "Quadrature".

    FIXME: this is not implemented in muGrid, but ad-hoc in Python.
    """

    nb_operators: int
    """#operators per quadrature points. It is useful to determine the #components of output field
     properly in "Quadrature"

    FIXME: muGrid doesn't expose this property, it is ad-hoc in Python.
    """

    def apply(self, field_in: Field, field_out: Field):
        """Apply the mapping."""

    def transpose(self, field_in: Field, field_out: Field):
        """Apply the inverse mapping. Matrix-wise, the operator is transposed."""


class Quadrature(abc.ABC):
    """Base class for computing quadrature over a region."""

    quadrature_label: str
    nb_quad_pts: int
    coords_quad_pts: np.ndarray
    weights_quad_pts: np.ndarray
    region_size_ratio: float

    def __init__(self, quadrature_label: str, region: Region):
        self.quadrature_label = quadrature_label
        self.region = region
        # Get & config the "field manager"
        region.collection.set_nb_sub_pt(self.quadrature_label, self.nb_quad_pts)
        self.collection = region.collection

    def discrete_variable(self, field_name: str, nb_components: int) -> Field:
        return self.collection.real_field(field_name, nb_components, "pixel")

    def integrand_field(self, field_name: str, nb_components: int) -> Field:
        return self.collection.real_field(field_name, nb_components, self.quadrature_label)

    def apply_operators(self, field: Field, operators: list[ConvOp]) -> list[Field]:
        result = []
        for operator in operators:
            field_out = self.integrand_field(f"{field.name}_{operator.name}", operator.nb_operators)
            operator.apply(field, field_out)
            result.append(field_out)
        return result

    def field_integral(self, field: Field):
        return self.region_size_ratio * self.region.sum(  # Sum over the whole domain
            # Sum (weighted) over quadrature points
            np.einsum("cs..., s-> c...", field.s, self.weights_quad_pts)
        )

    def field_sensitivity(self, integrand_parts: list[Field], operators: list[ConvOp]):
        result = []
        for integrand_part, operator in zip(integrand_parts, operators):
            field_out = self.discrete_variable(
                f"{integrand_part.name}_{operator.name}_trans",
                integrand_part.nb_components // operator.nb_operators,
            )
            operator.transpose(integrand_part, field_out)
            result.append(field_out)

        return self.region_size_ratio * self.region.gather(  # Gather the whole domain
            # Sum over operator sensitivity
            sum(field.s for field in result)
        )


class CentroidQuadrature(Quadrature):
    """Numerical intergration with quadrature points located at the centroid of the two triangles of
    each pixel. It provides discrete operators for interpolation and gradient.
    """

    op_interpolation: ConvOp
    op_gradient: ConvOp

    def __init__(self, quadrature_label: str, region: Region):
        super().__init__(quadrature_label, region)

        # Quadrature points
        self.nb_quad_pts = 2
        self.coords_quad_pts = np.array([[1 / 3, 1 / 3], [2 / 3, 2 / 3]])
        self.weights_quad_pts = np.array([1.0] * 2)

        # Area of the triangle element
        self.region_size_ratio = 0.5 * region.grid_spacing**2

        # Common parameters for operators
        conv_pts_shape = [2, 2]
        nb_conv_pts = np.multiply.reduce(conv_pts_shape)
        nb_pixelnodal_pts = 1
        nb_quad_pts = 2

        # A unit pixel with shape functions implemented
        pixel = LinearFiniteElementPixel()

        # Interpolation operator
        nb_operators_in_interpolation = 1
        self.op_interpolation = muGrid.ConvolutionOperatorDefault(
            np.reshape(
                pixel.interpolate(self.coords_quad_pts),
                shape=(-1, nb_conv_pts),
                order="F",
            ),
            conv_pts_shape,
            nb_pixelnodal_pts,
            nb_quad_pts,
            nb_operators_in_interpolation,
        )
        # NOTE: ad-hoc in Python
        self.op_interpolation.name = "interpolation"
        self.op_interpolation.nb_operators = nb_operators_in_interpolation

        # Gradient operator
        nb_operators_in_gradient = 2
        self.op_gradient = muGrid.ConvolutionOperatorDefault(
            np.reshape(
                pixel.gradient(self.coords_quad_pts) / region.grid_spacing,
                shape=(-1, nb_conv_pts),
                order="F",
            ),
            conv_pts_shape,
            nb_pixelnodal_pts,
            nb_quad_pts,
            nb_operators_in_gradient,
        )
        # NOTE: ad-hoc in Python
        self.op_gradient.name = "gradient"
        self.op_gradient.nb_operators = nb_operators_in_gradient


class LinearFiniteElementPixel:
    """A unit pixel discretized with linear finite element basis.

    The vertices of the pixel are (0,0), (1,0), (0,1), (1,1). It is divided into two triangles by
    the line connecting vertices (1,0) and (0,1), x_1 + x_2 = 1. The triangle with (0,0) vertice is
    the "lower triangle", the other is the "upper triangle".
    """

    @staticmethod
    def check_valid(local_coords):
        if np.size(local_coords, axis=1) != 2:
            raise ValueError("The coordinates must be 2D.")
        for x in np.ravel(local_coords):
            if x < 0 or x > 1:
                raise ValueError("The point must locate inside a unit square.")

    def interpolate(self, local_coords):
        self.check_valid(local_coords)
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

    def gradient(self, local_coords):
        self.check_valid(local_coords)
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
