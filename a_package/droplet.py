from dataclasses import dataclass
import numpy as np
import scipy.sparse as sparse


@dataclass
class CapillaryDroplet:
    phi: np.ndarray  # phase-field, periodic, nodal values in dimension of (Nx, Ny)
    h1: np.ndarray   # roughness of the 1 plate
    h2: np.ndarray   # roughness of the 2 plate
    d: float         # separation of two plates
    eta: float       # interfacial width
    M: int           # number of pixels along x-axis
    N: int           # number of pixels along y-axis
    dx: float        # dimension of pixels along x-axis
    dy: float        # dimension of pixels along y-axis

    def update_phase_field(self, phi_flat) -> None:
        """update for calculation only, doesn't necessarily change the `phi` field."""
    def update_separation(self, d): ...
    def compute_energy(self) -> float: ...
    def compute_energy_jacobian(self) -> np.ndarray: ...
    def compute_volume(self) -> float: ...
    def compute_volume_jacobian(self) -> np.ndarray: ...


@dataclass
class QuadratureRoughDroplet(CapillaryDroplet):
    r"""
    Field is discretized into triangular finite elements arranged as:
        ------
        | \  |
        |  \ |
        ------
    Phase values are assumed linear elements as in form of:
        a + b\xi + c\eta
    assuming (0, 0) is the center of the triangle for ease of notation. 
    Gradient values are computed accordingly:
        (\frac{b}{\Delta x}, \frac{c}{\Delta y})
    Integral of double well potential is evaluated via Gaussian Quadrature-2 point scheme
    """

    def __post_init__(self):
        self.at_contact = []  # to save indices where there is contact
        MN = self.M * self.N

        # K_a maps \phi grid points to central of the triangles, which are the quadrature points
        K_a_lower = sparse.lil_matrix((MN, MN), dtype=float)
        fill_cyclic_diagonal_2d(K_a_lower, (0,0), (self.M, self.N), 1/3)
        fill_cyclic_diagonal_2d(K_a_lower, (0,1), (self.M, self.N), 1/3)
        fill_cyclic_diagonal_2d(K_a_lower, (1,0), (self.M, self.N), 1/3)

        K_a_upper = sparse.lil_matrix((MN, MN), dtype=float)
        fill_cyclic_diagonal_2d(K_a_upper, (0,1), (self.M, self.N), 1/3)
        fill_cyclic_diagonal_2d(K_a_upper, (1,0), (self.M, self.N), 1/3)
        fill_cyclic_diagonal_2d(K_a_upper, (1,1), (self.M, self.N), 1/3)

        self.K_centroid = sparse.vstack([K_a_lower, K_a_upper], format="csr")

        # K_b maps \phi grid points to the difference in x direction
        K_b_lower = sparse.lil_matrix((MN, MN), dtype=float)
        fill_cyclic_diagonal_2d(K_b_lower, (0,0), (self.M, self.N), -1)
        fill_cyclic_diagonal_2d(K_b_lower, (1,0), (self.M, self.N), 1)

        K_b_upper = sparse.lil_matrix((MN, MN), dtype=float)
        fill_cyclic_diagonal_2d(K_b_upper, (0,1), (self.M, self.N), -1)
        fill_cyclic_diagonal_2d(K_b_upper, (1,1), (self.M, self.N), 1)

        self.Dx = sparse.vstack([K_b_lower, K_b_upper], format="csr") / self.dx
        self.Dx_t = sparse.csr_matrix(self.Dx.T)

        # K_c maps \phi grid points to the difference in y direction
        K_c_lower = sparse.lil_matrix((MN, MN), dtype=float)
        fill_cyclic_diagonal_2d(K_c_lower, (0,0), (self.M, self.N), -1)
        fill_cyclic_diagonal_2d(K_c_lower, (0,1), (self.M, self.N), 1)

        K_c_upper = sparse.lil_matrix((MN, MN), dtype=float)
        fill_cyclic_diagonal_2d(K_c_upper, (1,0), (self.M, self.N), -1)
        fill_cyclic_diagonal_2d(K_c_upper, (1,1), (self.M, self.N), 1)

        self.Dy = sparse.vstack([K_c_lower, K_c_upper], format="csr") / self.dy
        self.Dy_t = sparse.csr_matrix(self.Dy.T)

        # memory of phase-field related terms
        num_triangle = 2   # numer of triangles per pixel

        self.g_triangle = np.empty((num_triangle*MN))  # average of a triangle
        self.at_contact = []
        self.update_separation(self.d)

        poly_deg = 4  # max degree of polymers
        self.phi_power = np.empty((poly_deg, num_triangle*MN))
        """the values of `phi` and its power terms
            index  0 -- phi, set with `update_phase_field`
            index  1 -- phi^2, update with `compute_energy`, `compute_energy_jacobian`
            index  2 -- phi^3, update with `compute_energy`, `compute_energy_jacobian`
            index  3 -- phi^4, update with `compute_energy`
        """
        self.phi_dx = np.empty((num_triangle*MN))
        self.phi_dy = np.empty((num_triangle*MN))
        self.update_phase_field(self.phi.ravel())

        num_quadrature = 1 # number of quadrature points per triangle
        self.weight = np.ones((1, num_triangle*num_quadrature*MN), dtype=float)
        self.triangle_area = 0.5 * self.dx * self.dy

    def update_phase_field(self, phi_flat) -> None:
        self.phi_power[0] = self.K_centroid @ phi_flat
        self.phi_dx[:] = self.Dx @ phi_flat
        self.phi_dy[:] = self.Dy @ phi_flat

    def update_separation(self, d) -> None:
        g_grid = (self.h1 - self.h2).ravel() + d
        # check contact
        self.at_contact = g_grid < 0
        g_grid[self.at_contact] = 0
        self.g_triangle[:] = self.K_centroid @ g_grid  # triangular average

    def compute_energy(self) -> float:
        # update necessary power terms
        self.phi_power[1] = self.phi_power[0]**2
        self.phi_power[2] = self.phi_power[0] * self.phi_power[1]
        self.phi_power[3] = self.phi_power[1]**2
        # compute values at grid points: (1/eta)(phi^2 - 2 phi^3 + phi^4)
        double_well = (1/self.eta) * (self.phi_power[1] - 2*self.phi_power[2] + self.phi_power[3])
        perimeter = self.eta * (self.phi_dx**2 + self.phi_dy**2)
        return ((double_well + perimeter) * self.g_triangle).sum() * self.triangle_area

    def compute_force(self) -> float:
        # update necessary power terms
        self.phi_power[1] = self.phi_power[0]**2
        self.phi_power[2] = self.phi_power[0] * self.phi_power[1]
        self.phi_power[3] = self.phi_power[1]**2
        # compute values at grid points: (1/eta)(phi^2 - 2 phi^3 + phi^4)
        double_well = (1/self.eta) * (self.phi_power[1] - 2*self.phi_power[2] + self.phi_power[3])
        perimeter = self.eta * (self.phi_dx**2 + self.phi_dy**2)
        return -(double_well + perimeter).sum() * self.triangle_area

    def compute_energy_jacobian(self) -> np.ndarray:
        # update necessary power terms
        self.phi_power[1] = self.phi_power[0]**2
        self.phi_power[2] = self.phi_power[0] * self.phi_power[1]
        # compute values at grid points: (2/eta)(phi - 3 phi^2 + phi^3)
        double_well_jacobian = (2/self.eta) * ((self.phi_power[0] - 3*self.phi_power[1] + 2*self.phi_power[2])* self.g_triangle) @ self.K_centroid
        perimeter_jacobian = (2*self.eta) * (self.Dx_t @ (self.g_triangle * self.phi_dx) + self.Dy_t @ (self.g_triangle * self.phi_dy))
        jacob =  (double_well_jacobian + perimeter_jacobian) * self.triangle_area
        return jacob

    def compute_volume(self) -> float:
        return (self.g_triangle * self.phi_power[0]).sum() * self.triangle_area

    def compute_volume_jacobian(self) -> np.ndarray:
        return (self.g_triangle @ self.K_centroid) * self.triangle_area


def fill_cyclic_diagonal_1d(mat: sparse.spmatrix, j: int, N: int, val: float):
    """Fill cyclically, element-wise in the j-th diagonal of a matrix.
    The matrix represents a mapping from 1D data to 1D data.
    """
    assert mat.ndim == 2
    i = np.arange(N)
    mat[i, (i + j)%N] = val


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
    mat[i1*N2 + i2, (i1 + j1)%N1*N2 + (i2 + j2)%N2] = val


def fill_vertical_block_diagonal(mat: sparse.spmatrix, N: int, val: list[float]):
    """Fill cyclically, block-wise in the diagonal of a matrix.
    """
    assert mat.ndim == 2

    # cartesian product of range(N1) and range(N2)
    m = len(val)
    for i in range(N):
        mat[m*i:m*(i+1), i] = val
