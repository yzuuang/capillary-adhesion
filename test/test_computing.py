import numpy as np
import scipy.sparse as sparse

from a_package.computing import Region, CentroidQuadrature


# FIXME: add a test to compare the implementation with matrix notations


class Quadrature_SparseMatrix:
    """A reference implementation with SciPy SparseMatrix"""

    def __init__(self, region: Region):
        M, N = region.nb_subdomain_grid_pts
        MN = M * N

        # K_a maps \phi grid points to central of the triangles, which are the quadrature points
        coeff_interp = 1 / 3
        K_a_lower = sparse.lil_matrix((MN, MN), dtype=float)
        fill_cyclic_diagonal_2d(K_a_lower, (0, 0), (M, N), coeff_interp)
        fill_cyclic_diagonal_2d(K_a_lower, (0, 1), (M, N), coeff_interp)
        fill_cyclic_diagonal_2d(K_a_lower, (1, 0), (M, N), coeff_interp)

        K_a_upper = sparse.lil_matrix((MN, MN), dtype=float)
        fill_cyclic_diagonal_2d(K_a_upper, (0, 1), (M, N), coeff_interp)
        fill_cyclic_diagonal_2d(K_a_upper, (1, 0), (M, N), coeff_interp)
        fill_cyclic_diagonal_2d(K_a_upper, (1, 1), (M, N), coeff_interp)

        self.K_centroid = sparse.vstack([K_a_lower, K_a_upper], format="csr")

        # K_b maps \phi grid points to the difference in x direction
        coeff_grad = 1 / region.grid_spacing
        K_b_lower = sparse.lil_matrix((MN, MN), dtype=float)
        fill_cyclic_diagonal_2d(K_b_lower, (0, 0), (M, N), -coeff_grad)
        fill_cyclic_diagonal_2d(K_b_lower, (1, 0), (M, N), coeff_grad)

        K_b_upper = sparse.lil_matrix((MN, MN), dtype=float)
        fill_cyclic_diagonal_2d(K_b_upper, (0, 1), (M, N), -coeff_grad)
        fill_cyclic_diagonal_2d(K_b_upper, (1, 1), (M, N), coeff_grad)

        self.Dx = sparse.vstack([K_b_lower, K_b_upper], format="csr")
        # self.Dx_t = sparse.csr_matrix(self.Dx.T)

        # K_c maps \phi grid points to the difference in y direction
        K_c_lower = sparse.lil_matrix((MN, MN), dtype=float)
        fill_cyclic_diagonal_2d(K_c_lower, (0, 0), (M, N), -coeff_grad)
        fill_cyclic_diagonal_2d(K_c_lower, (0, 1), (M, N), coeff_grad)

        K_c_upper = sparse.lil_matrix((MN, MN), dtype=float)
        fill_cyclic_diagonal_2d(K_c_upper, (1, 0), (M, N), -coeff_grad)
        fill_cyclic_diagonal_2d(K_c_upper, (1, 1), (M, N), coeff_grad)

        self.Dy = sparse.vstack([K_c_lower, K_c_upper], format="csr")


def fill_cyclic_diagonal_1d(mat: sparse.spmatrix, j: int, N: int, val: float):
    """Fill cyclically, element-wise in the j-th diagonal of a matrix.
    The matrix represents a mapping from 1D data to 1D data.
    """
    assert mat.ndim == 2
    i = np.arange(N)
    mat[i, (i + j) % N] = val


def fill_cyclic_diagonal_2d(
    mat: sparse.spmatrix, j: tuple[int, int], N: tuple[int, int], val: float
):
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
    """Fill cyclically, block-wise in the diagonal of a matrix."""
    assert mat.ndim == 2

    # cartesian product of range(N1) and range(N2)
    m = len(val)
    for i in range(N):
        mat[m * i : m * (i + 1), i] = val
