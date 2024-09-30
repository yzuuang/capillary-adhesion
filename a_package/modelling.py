"""
The physical perspective of the capillary bridge.
"""

import dataclasses as dc

import numpy as np
import numpy.fft as fft
import numpy.random as random
import scipy.sparse as sparse

from a_package.solving import NumOptEq


@dc.dataclass
class Region:
    """A discrete space in 2D."""
    lx: float
    ly: float
    nx: int
    ny: int

    def __post_init__(self):
        self.dx = self.lx / self.nx
        self.x = np.arange(self.nx) * self.dx
        self.qx = (2 * np.pi) * fft.fftfreq(self.nx, self.dx)

        self.dy = self.ly / self.ny
        self.y = np.arange(self.ny) * self.dy
        self.qy = (2 * np.pi) * fft.fftfreq(self.ny, self.dy)


def wavevector_norm(*q):
    # from N-axis to N-component of coordinates
    q_mesh = np.meshgrid(*q)
    # coordinates to norms
    return np.sqrt(sum(q_i ** 2 for q_i in q_mesh))


@dc.dataclass
class SelfAffineRoughness:
    C0: float
    qR: float
    qS: float
    H: float

    def mapto_psd(self, q: np.ndarray):
        C = np.empty_like(q)
        C[q < self.qR] = self.C0 * self.qR ** (-2 - 2 * self.H)
        self_affine_regime = (q >= self.qR) & (q < self.qS)
        C[self_affine_regime] = self.C0 * q[self_affine_regime] ** (-2 - 2 * self.H)
        C[q >= self.qS] = 0
        return C


def PSD_to_height(C: np.ndarray, seed=None):
    # <h^2> corresponding to PSD, thus, take the square-root
    h_norm = np.sqrt(C)

    # impose some random phase angle
    rng = random.default_rng(seed)
    phase_angle = np.exp(1j * rng.uniform(0, 2 * np.pi, C.shape))

    return fft.ifftn(h_norm * phase_angle).real


@dc.dataclass
class CapillaryBridge:
    """All necessary parameters for visualizing a capillary bridge.

    Data attributes: field is stored with nodal values as 2D array.
    1 refer to the solid top; 2 refer to the solid base.
    """
    region: Region
    eta: float
    h1: np.ndarray
    h2: np.ndarray
    ix1_iy1: tuple[int, int] = None
    z1: float = 0.
    g: np.ndarray = None
    no_gap: np.ndarray = None
    phi: np.ndarray = None

    def __post_init__(self):
        if self.ix1_iy1 is None:
            self.ix1_iy1 = [0, 0]
        self.h1_origin = np.roll(self.h1, [-self.ix1_iy1[0], -self.ix1_iy1[1]], axis=(0,1))
        self.inner = ComputeCapillary(self)

    def update_gap(self):
        # Lateral displacement of the solid top
        # NOTE: assume periodic boundary
        self.h1 = np.roll(self.h1_origin, self.ix1_iy1, axis=(0,1))
        # Gap
        self.g = self.h1 + self.z1 - self.h2

        # Check where the top and the base collide
        self.no_gap = self.g < 0
        # NOTE: assume interpenetration
        self.g[self.no_gap] = 0

        # Call the inner class method
        self.inner.update_gap(self.g.ravel())

    def update_phase_field(self):
        # Clean the phase-field where the solid bodies contact
        self.phi[self.no_gap] = 0

        # Call the inner class method
        self.inner.update_phase_field(self.phi.ravel())

    @property
    def displacement(self):
        return np.array([
            self.ix1_iy1[0] * self.region.dx,
            self.ix1_iy1[1] * self.region.dy,
            self.z1
        ])

    @property
    def volume(self):
        return self.inner.compute_volume()

    @property
    def energy(self):
        return self.inner.compute_energy()

    @property
    def force(self):
        return self.inner.compute_force()

    def formulate_with_constant_volume(self, volume: float):
        def objective(x: np.ndarray):
            self.inner.update_phase_field(x)
            return self.inner.compute_energy()

        def objective_jacobian(x: np.ndarray):
            self.inner.update_phase_field(x)
            return self.inner.compute_energy_jacobian()

        def constraint(x: np.ndarray):
            self.inner.update_phase_field(x)
            return self.inner.compute_volume() - volume

        def constraint_jacobian(x: np.ndarray):
            self.inner.update_phase_field(x)
            return self.inner.compute_volume_jacobian()

        return NumOptEq(objective, objective_jacobian, constraint, constraint_jacobian)

    def validate_phase_field(self):

        # phase field < 0
        if np.any(self.phi < 0):
            outlier = np.where(self.phi < 0, self.phi, np.nan)
            count = np.count_nonzero(~np.isnan(outlier))
            extreme = np.nanmin(outlier)
            print(f"Notice: phase field has {count} values < 0, min at {extreme:.2e}")

        # phase field > 1
        if np.any(self.phi > 1):
            outlier = np.where(self.phi > 1, self.phi, np.nan)
            count = np.count_nonzero(~np.isnan(outlier))
            extreme = np.nanmax(outlier)
            print(f"Notice: phase field has {count} values > 1, max at 1.0+{extreme - 1:.2e}.")


class ComputeCapillary:
    """Inner class
    
    Data attributes: field is stored with values of elements as 1D array (fast computation).

    Methods: most are computation that will be used for optimization. 
    Integral of double well potential is evaluated via a centroid rule of Quadrature for triangles.
    """

    def __init__(self, capi: CapillaryBridge):
        # The grid
        region = capi.region
        # Number of pixels
        n_pixel = region.nx * region.ny
        # 2 triangle elements per pixel
        n_element = 2 * n_pixel
        # Area of one element
        self.element_area = 0.5 * region.dx * region.dy
        # FE interpolation
        self.map = FirstOrderElement(region)

        # Copy simple parameters
        self.eta = capi.eta

        # The gap between two solid bodies
        self.g = np.empty((n_element))
        self.dg_dx = np.empty((n_element))
        self.dg_dy = np.empty((n_element))

        # Highest polynomial degree for computation
        poly_degr = 4
        # The phase-field and its powers; each stored as a column of a 2D array. 
        self.phi_powers = np.empty((poly_degr, n_element))
        """    
            Index  0 -- phi, set with 'update_phase_field';
            Index  1 -- phi^2, update with 'compute_energy', 'compute_energy_jacobian';
            Index  2 -- phi^3, update with 'compute_energy', 'compute_energy_jacobian';
            Index  3 -- phi^4, update with 'compute_energy'.
        """
        # The gradient of phase-field
        self.dphi_dx = np.empty((n_element))
        self.dphi_dy = np.empty((n_element))

        # Number of quadrature points per triangle element
        # n_quadrature = 1
        # NOTE: Though 1 quadrature point at the centroid can only approximates an integral of
        # a 4th oder polynomial, it seems sufficient to yield a solution in numerical experiments. 
        # weight = np.ones((1, n_quadrature * n_elem), dtype=float)

    def update_gap(self, g: np.ndarray):
        """
        g --- nodal value, flat.
        """
        self.g = self.map.K_centroid @ g
        self.dg_dx = self.map.Dx @ g
        self.dg_dy = self.map.Dy @ g
        # dg_dz = 1

    def update_phase_field(self, phi: np.ndarray):
        """
        phi --- nodal value, flat.
        """
        self.phi_powers[0] = self.map.K_centroid @ phi
        self.dphi_dx = self.map.Dx @ phi
        self.dphi_dy = self.map.Dy @ phi

    def compute_energy(self) -> float:
        # update necessary power terms
        self.phi_powers[1] = self.phi_powers[0] ** 2
        self.phi_powers[2] = self.phi_powers[0] * self.phi_powers[1]
        self.phi_powers[3] = self.phi_powers[1] ** 2
        # compute at quadrature points: (1/eta) (phi^2 - 2 phi^3 + phi^4)
        double_well = (1 / self.eta) * (self.phi_powers[1] - 2 * self.phi_powers[2] + self.phi_powers[3])
        # constant within the element: eta (dphi_dx^2 + dphi_dy^2)
        perimeter = self.eta * (self.dphi_dx ** 2 + self.dphi_dy ** 2)
        return ((double_well + perimeter) * self.g).sum() * self.element_area

    def compute_force(self):
        # update necessary power terms
        self.phi_powers[1] = self.phi_powers[0] ** 2
        self.phi_powers[2] = self.phi_powers[0] * self.phi_powers[1]
        self.phi_powers[3] = self.phi_powers[1] ** 2

        # the common part of 3 components
        # constant within the element: eta (dphi_dx^2 + dphi_dy^2)
        perimeter = self.eta * (self.dphi_dx ** 2 + self.dphi_dy ** 2)
        # compute at quadrature points: (1/eta)(phi^2 - 2 phi^3 + phi^4)
        double_well = (1 / self.eta) * (self.phi_powers[1] - 2 * self.phi_powers[2] + self.phi_powers[3])
        de_dg = (double_well + perimeter)

        # the different part of 3 components
        f_x = (de_dg * self.dg_dx).sum()
        f_y = (de_dg * self.dg_dy).sum()
        # dg_dz = 1
        f_z = de_dg.sum()

        return np.array([f_x, f_y, f_z]) * self.element_area

    def compute_energy_jacobian(self) -> np.ndarray:
        # update necessary power terms
        self.phi_powers[1] = self.phi_powers[0] ** 2
        self.phi_powers[2] = self.phi_powers[0] * self.phi_powers[1]
        # constant within the element: 2 eta g (Dx phi Dx + Dy phi Dy)
        perimeter_jacobian = (2 * self.eta) * ((self.g * self.dphi_dx) @ self.map.Dx + 
                                               (self.g * self.dphi_dy) @ self.map.Dy)
        # compute at quadrature points: (2/eta) g (phi - 3 phi^2 + phi^3)
        double_well_jacobian = (2 / self.eta) * ((self.phi_powers[0] - 3 * self.phi_powers[1] +
                                                  2 * self.phi_powers[2]) * self.g) @ self.map.K_centroid

        jacobian = (double_well_jacobian + perimeter_jacobian) * self.element_area
        return jacobian

    def compute_volume(self) -> float:
        return (self.g * self.phi_powers[0]).sum() * self.element_area

    def compute_volume_jacobian(self) -> np.ndarray:
        return (self.g @ self.map.K_centroid) * self.element_area


class FirstOrderElement:
    """
    Create matrices that maps a vector of nodal values into a vector of values of interest in finite elements. 

    Field is interpolated by triangular linear finite elements: 'a + b*xi + c*eta', with 'a' located at the 
    centroid. Gradient values are thus constant: '(b / dx, c / dy)'.
    """
    K_centroid: np.ndarray
    Dx: np.ndarray
    Dy: np.ndarray

    def __init__(self, region: Region):
        M = region.nx
        N = region.ny
        MN = M * N

        # K_a maps \phi grid points to central of the triangles, which are the quadrature points
        K_a_lower = sparse.lil_matrix((MN, MN), dtype=float)
        fill_cyclic_diagonal_2d(K_a_lower, (0, 0), (M, N), 1 / 3)
        fill_cyclic_diagonal_2d(K_a_lower, (0, 1), (M, N), 1 / 3)
        fill_cyclic_diagonal_2d(K_a_lower, (1, 0), (M, N), 1 / 3)

        K_a_upper = sparse.lil_matrix((MN, MN), dtype=float)
        fill_cyclic_diagonal_2d(K_a_upper, (0, 1), (M, N), 1 / 3)
        fill_cyclic_diagonal_2d(K_a_upper, (1, 0), (M, N), 1 / 3)
        fill_cyclic_diagonal_2d(K_a_upper, (1, 1), (M, N), 1 / 3)

        self.K_centroid = sparse.vstack([K_a_lower, K_a_upper], format="csr")

        # K_b maps \phi grid points to the difference in x direction
        K_b_lower = sparse.lil_matrix((MN, MN), dtype=float)
        fill_cyclic_diagonal_2d(K_b_lower, (0, 0), (M, N), -1)
        fill_cyclic_diagonal_2d(K_b_lower, (1, 0), (M, N), 1)

        K_b_upper = sparse.lil_matrix((MN, MN), dtype=float)
        fill_cyclic_diagonal_2d(K_b_upper, (0, 1), (M, N), -1)
        fill_cyclic_diagonal_2d(K_b_upper, (1, 1), (M, N), 1)

        self.Dx = sparse.vstack([K_b_lower, K_b_upper], format="csr") / region.dx
        # self.Dx_t = sparse.csr_matrix(self.Dx.T)

        # K_c maps \phi grid points to the difference in y direction
        K_c_lower = sparse.lil_matrix((MN, MN), dtype=float)
        fill_cyclic_diagonal_2d(K_c_lower, (0, 0), (M, N), -1)
        fill_cyclic_diagonal_2d(K_c_lower, (0, 1), (M, N), 1)

        K_c_upper = sparse.lil_matrix((MN, MN), dtype=float)
        fill_cyclic_diagonal_2d(K_c_upper, (1, 0), (M, N), -1)
        fill_cyclic_diagonal_2d(K_c_upper, (1, 1), (M, N), 1)

        self.Dy = sparse.vstack([K_c_lower, K_c_upper], format="csr") / region.dy
        # self.Dy_t = sparse.csr_matrix(self.Dy.T)


def fill_cyclic_diagonal_1d(mat: sparse.spmatrix, j: int, N: int, val: float):
    """Fill cyclically, element-wise in the j-th diagonal of a matrix.
    The matrix represents a mapping from 1D data to 1D data.
    """
    assert mat.ndim == 2
    i = np.arange(N)
    mat[i, (i + j) % N] = val


def fill_cyclic_diagonal_2d(mat: sparse.spmatrix, j: tuple[int, int], N: tuple[int, int], val: float):
    """Fill cyclically, element-wise in the j-th diagonal of a matrix.
    The matrix represents a mapping from 2D data to 2D data.
    However, the 2d array is ravelled into 1D array for efficiency.
    """
    assert mat.ndim == 2

    # cartesian product of range(N1) and range(N2)
    N1, N2 = N
    i1, i2 = np.mgrid[:N1, :N2]
    i1 = i1.ravel()
    i2 = i2.ravel()

    j1, j2 = j
    mat[i1 * N2 + i2, (i1 + j1) % N1 * N2 + (i2 + j2) % N2] = val


def fill_vertical_block_diagonal(mat: sparse.spmatrix, N: int, val: list[float]):
    """Fill cyclically, block-wise in the diagonal of a matrix.
    """
    assert mat.ndim == 2

    # cartesian product of range(N1) and range(N2)
    m = len(val)
    for i in range(N):
        mat[m * i:m * (i + 1), i] = val
