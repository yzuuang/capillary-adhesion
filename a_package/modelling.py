"""
The physical perspective of the capillary bridge.
"""

import dataclasses as dc

import numpy as np
import numpy.fft as fft
import numpy.random as random
import scipy.sparse as sparse

from a_package.data_record import NumOptEq


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
        C[(q >= self.qR) & (q < self.qS)] = self.C0 * q[(q >= self.qR) & (q < self.qS)] ** (-2 - 2 * self.H)
        C[q >= self.qS] = 0
        return C


def PSD_to_height(C: np.ndarray):
    # <h^2> corresponding to PSD, thus, take the square-root
    h_norm = np.sqrt(C)

    # impose some random phase angle
    rng = random.default_rng()
    phase_angle = np.exp(1j * rng.uniform(0, 2 * np.pi, C.shape))

    return fft.ifftn(h_norm * phase_angle).real


@dc.dataclass
class CapillaryBridge:
    """
    Field is interpolated by triangular linear finite elements as: `a + b*xi + c*eta`, with `a` located at the centroid.
    Gradient values are computed accordingly as: `(b / dx, c / dy)`
    Integral of double well potential is evaluated via Gaussian Quadrature-2 point scheme
    """
    region: Region
    eta: float
    h1: np.ndarray
    h2: np.ndarray
    phi: np.ndarray

    def __init__(self, region: Region, eta: float, h1: np.ndarray, h2: np.ndarray, phi: np.ndarray):
        self.region = region
        self.eta = eta
        self.h1 = np.ravel(h1)
        self.h2 = np.ravel(h2)
        self.phi = np.ravel(phi)

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
        self.Dx_t = sparse.csr_matrix(self.Dx.T)

        # K_c maps \phi grid points to the difference in y direction
        K_c_lower = sparse.lil_matrix((MN, MN), dtype=float)
        fill_cyclic_diagonal_2d(K_c_lower, (0, 0), (M, N), -1)
        fill_cyclic_diagonal_2d(K_c_lower, (0, 1), (M, N), 1)

        K_c_upper = sparse.lil_matrix((MN, MN), dtype=float)
        fill_cyclic_diagonal_2d(K_c_upper, (1, 0), (M, N), -1)
        fill_cyclic_diagonal_2d(K_c_upper, (1, 1), (M, N), 1)

        self.Dy = sparse.vstack([K_c_lower, K_c_upper], format="csr") / region.dy
        self.Dy_t = sparse.csr_matrix(self.Dy.T)

        # memory of phase-field related terms
        num_triangle = 2  # numer of triangles per pixel

        self.g_triangle = np.empty((num_triangle * MN))  # average of a triangle
        self.at_contact = []

        poly_deg = 4  # max degree of polymers
        self.phi_power = np.empty((poly_deg, num_triangle * MN))
        """the values of `phi` and its power terms
    
            index  0 -- phi, set with `update_phase_field`;
            index  1 -- phi^2, update with `compute_energy`, `compute_energy_jacobian`;
            index  2 -- phi^3, update with `compute_energy`, `compute_energy_jacobian`;
            index  3 -- phi^4, update with `compute_energy`.
        """
        self.phi_dx = np.empty((num_triangle * MN))
        self.phi_dy = np.empty((num_triangle * MN))

        num_quadrature = 1  # number of quadrature points per triangle
        self.weight = np.ones((1, num_triangle * num_quadrature * MN), dtype=float)
        self.triangle_area = 0.5 * region.dx * region.dy

    def update_phase_field(self, phi_flat) -> None:
        self.phi_power[0] = self.K_centroid @ phi_flat
        self.phi_dx[:] = self.Dx @ phi_flat
        self.phi_dy[:] = self.Dy @ phi_flat

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

    def update_separation(self, d) -> None:
        g_grid = (self.h1 - self.h2).ravel() + d
        # check contact
        self.at_contact = g_grid < 0
        g_grid[self.at_contact] = 0
        self.g_triangle[:] = self.K_centroid @ g_grid  # triangular average

    def compute_energy(self) -> float:
        # update necessary power terms
        self.phi_power[1] = self.phi_power[0] ** 2
        self.phi_power[2] = self.phi_power[0] * self.phi_power[1]
        self.phi_power[3] = self.phi_power[1] ** 2
        # compute values at grid points: (1/eta)(phi^2 - 2 phi^3 + phi^4)
        double_well = (1 / self.eta) * (self.phi_power[1] - 2 * self.phi_power[2] + self.phi_power[3])
        perimeter = self.eta * (self.phi_dx ** 2 + self.phi_dy ** 2)
        return ((double_well + perimeter) * self.g_triangle).sum() * self.triangle_area

    def compute_force(self) -> float:
        # update necessary power terms
        self.phi_power[1] = self.phi_power[0] ** 2
        self.phi_power[2] = self.phi_power[0] * self.phi_power[1]
        self.phi_power[3] = self.phi_power[1] ** 2
        # compute values at grid points: (1/eta)(phi^2 - 2 phi^3 + phi^4)
        double_well = (1 / self.eta) * (self.phi_power[1] - 2 * self.phi_power[2] + self.phi_power[3])
        perimeter = self.eta * (self.phi_dx ** 2 + self.phi_dy ** 2)
        return -(double_well + perimeter).sum() * self.triangle_area

    def compute_energy_jacobian(self) -> np.ndarray:
        # update necessary power terms
        self.phi_power[1] = self.phi_power[0] ** 2
        self.phi_power[2] = self.phi_power[0] * self.phi_power[1]
        # compute values at grid points: (2/eta)(phi - 3 phi^2 + phi^3)
        double_well_jacobian = (2 / self.eta) * ((self.phi_power[0] - 3 * self.phi_power[1] + 2 * self.phi_power[2])
                                                 * self.g_triangle) @ self.K_centroid
        perimeter_jacobian = (2 * self.eta) * (self.Dx_t @ (self.g_triangle * self.phi_dx)
                                               + self.Dy_t @ (self.g_triangle * self.phi_dy))
        jacob = (double_well_jacobian + perimeter_jacobian) * self.triangle_area
        return jacob

    def compute_volume(self) -> float:
        return (self.g_triangle * self.phi_power[0]).sum() * self.triangle_area

    def compute_volume_jacobian(self) -> np.ndarray:
        return (self.g_triangle @ self.K_centroid) * self.triangle_area

    def formulate_with_constant_volume(self, volume: float):
        def f(x: np.ndarray):
            self.update_phase_field(x)
            return self.compute_energy()

        def f_grad(x: np.ndarray):
            self.update_phase_field(x)
            return self.compute_energy_jacobian()

        def g(x: np.ndarray):
            self.update_phase_field(x)
            return self.compute_volume() - volume

        def g_grad(x: np.ndarray):
            self.update_phase_field(x)
            return self.compute_volume_jacobian()

        return NumOptEq(f, f_grad, g, g_grad)

    def formulate_with_constant_chemical_potential(self, chemical_potential: float):
        raise NotImplementedError()


def fill_cyclic_diagonal_1d(mat: sparse.spmatrix, j: int, N: int, val: float):
    """Fill cyclically, element-wise in the j-th diagonal of a matrix.
    The matrix represents a mapping from 1D data to 1D data.
    """
    assert mat.ndim == 2
    i = np.arange(N)
    mat[i, (i + j) % N] = val


def fill_cyclic_diagonal_2d(mat: sparse.spmatrix, j: tuple[int, int], N: tuple[int, int], val: float):
    """fill cyclically, element-wise in the j-th diagonal of a matrix.
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
