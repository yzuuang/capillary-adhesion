
import numpy as np
import scipy.sparse as sparse

from a_package.grid import Grid

class FirstOrderElement:
    """
    Create matrices that maps a vector of nodal values into a vector of values of interest in finite elements.

    Field is interpolated by triangular linear finite elements: 'a + b*xi + c*eta', with 'a' located at the
    centroid. Gradient values are thus constant: '(b / dx, c / dy)'.

    It combines interpolation and quadrature.
    """

    K_centroid: np.ndarray
    Dx: np.ndarray
    Dy: np.ndarray

    def __init__(self, grid: Grid):
        [M, N] = grid.nb_elements
        self.nb_quad_pts = 2
        self.grid_shape = (M, N)
        MN = M * N

        # K_a maps \phi grid points to central of the triangles, which are the quadrature points
        K_a_lower = sparse.lil_matrix((MN, MN), dtype=float)
        fill_cyclic_diagonal_pseudo_2d(K_a_lower, (0, 0), (M, N), 1 / 3)
        fill_cyclic_diagonal_pseudo_2d(K_a_lower, (0, 1), (M, N), 1 / 3)
        fill_cyclic_diagonal_pseudo_2d(K_a_lower, (1, 0), (M, N), 1 / 3)

        K_a_upper = sparse.lil_matrix((MN, MN), dtype=float)
        fill_cyclic_diagonal_pseudo_2d(K_a_upper, (0, 1), (M, N), 1 / 3)
        fill_cyclic_diagonal_pseudo_2d(K_a_upper, (1, 0), (M, N), 1 / 3)
        fill_cyclic_diagonal_pseudo_2d(K_a_upper, (1, 1), (M, N), 1 / 3)

        self.K_centroid = sparse.vstack([K_a_lower, K_a_upper], format="csr")

        # K_b maps \phi grid points to the difference in x direction
        K_b_lower = sparse.lil_matrix((MN, MN), dtype=float)
        fill_cyclic_diagonal_pseudo_2d(K_b_lower, (0, 0), (M, N), -1)
        fill_cyclic_diagonal_pseudo_2d(K_b_lower, (1, 0), (M, N), 1)

        K_b_upper = sparse.lil_matrix((MN, MN), dtype=float)
        fill_cyclic_diagonal_pseudo_2d(K_b_upper, (0, 1), (M, N), -1)
        fill_cyclic_diagonal_pseudo_2d(K_b_upper, (1, 1), (M, N), 1)

        self.Dx = sparse.vstack([K_b_lower, K_b_upper], format="csr") / grid.element_sizes[0]
        # self.Dx_t = sparse.csr_matrix(self.Dx.T)

        # K_c maps \phi grid points to the difference in y direction
        K_c_lower = sparse.lil_matrix((MN, MN), dtype=float)
        fill_cyclic_diagonal_pseudo_2d(K_c_lower, (0, 0), (M, N), -1)
        fill_cyclic_diagonal_pseudo_2d(K_c_lower, (0, 1), (M, N), 1)

        K_c_upper = sparse.lil_matrix((MN, MN), dtype=float)
        fill_cyclic_diagonal_pseudo_2d(K_c_upper, (1, 0), (M, N), -1)
        fill_cyclic_diagonal_pseudo_2d(K_c_upper, (1, 1), (M, N), 1)

        self.Dy = sparse.vstack([K_c_lower, K_c_upper], format="csr") / grid.element_sizes[1]
        # self.Dy_t = sparse.csr_matrix(self.Dy.T)

    def interp_value_centroid(self, data: np.ndarray):
        """Map nodal values to the interpolated values at centroid."""
        return (self.K_centroid @ data.ravel()).reshape(-1, self.nb_quad_pts, *self.grid_shape)

    def prop_sens_value_centroid(self, data: np.ndarray):
        """Propogate the sensitivity of corresponding interpolation backward."""
        return (data.ravel() @ self.K_centroid).reshape(-1, *self.grid_shape)

    def interp_gradient_x(self, data: np.ndarray):
        """Map nodal values to the interpolated gradient values, component in x."""
        return (self.Dx @ data.ravel()).reshape(-1, self.nb_quad_pts, *self.grid_shape)

    def prop_sens_gradient_x(self, data: np.ndarray):
        """Propogate the sensitivity of corresponding interpolation backward."""
        return (data.ravel() @ self.Dx).reshape(-1, *self.grid_shape)

    def interp_gradient_y(self, data: np.ndarray):
        """Map nodal values to the interpolated gradient values, component in y."""
        return (self.Dy @ data.ravel()).reshape(-1, self.nb_quad_pts, *self.grid_shape)

    def prop_sens_gradient_y(self, data: np.ndarray):
        """Propogate the sensitivity of corresponding interpolation backward."""
        return (data.ravel() @ self.Dy).reshape(-1, *self.grid_shape)


def fill_cyclic_diagonal_1d(mat: sparse.spmatrix, j: int, N: int, val: float):
    """Fill cyclically, element-wise in the j-th diagonal of a matrix.
    The matrix represents a mapping from 1D data to 1D data.
    """
    assert mat.ndim == 2
    i = np.arange(N)
    mat[i, (i + j) % N] = val


def fill_cyclic_diagonal_pseudo_2d(mat: sparse.spmatrix, j: tuple[int, int], N: tuple[int, int], val: float, row_maj: bool=True):
    """Fill cyclically, element-wise in the j-th diagonal of a matrix.
    The matrix represents a mapping from 2D data to 2D data.
    However, the 2D data is ravelled and represented as a 1D array.
    """
    assert mat.ndim == 2

    # cartesian product of range(N1) and range(N2)
    N1, N2 = N
    i1, i2 = np.mgrid[:N1, :N2]
    i1 = i1.ravel()
    i2 = i2.ravel()

    j1, j2 = j
    if row_maj:
        # the 2D data is flattened with row-major (contiguous in 1st axis)
        mat[i1 * N2 + i2, (i1 + j1) % N1 * N2 + (i2 + j2) % N2] = val
    else:
        # the 2D data is flattened with column-major (contiguous in 2nd axis)
        mat[i1 + i2 * N1, (i1 + j1) % N1 + (i2 + j2) % N2 * N1] = val


def fill_vertical_block_diagonal(mat: sparse.spmatrix, N: int, val: list[float]):
    """Fill cyclically, block-wise in the diagonal of a matrix."""
    assert mat.ndim == 2

    # cartesian product of range(N1) and range(N2)
    m = len(val)
    for i in range(N):
        mat[m * i : m * (i + 1), i] = val
