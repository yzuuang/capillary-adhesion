import numpy as np
import numpy.random as random
import scipy.sparse as sparse

from a_package.computing import Grid, Linear2DFiniteElementInPixel, centroid_quadrature
from a_package.simulating import random_initial_guess


rng = random.default_rng()


def test_linear_finite_element():
    # A grid for testing the interpolation
    test_size = [5, 5]
    grid = Grid([1.0, 1.0], test_size)
    grid.add_sub_pt_scheme(centroid_quadrature.tag, centroid_quadrature.nb_quad_pts)

    # A few fields for input / output values
    input_field = grid.real_field('input', 1)
    value_field = grid.real_field('value', 1, centroid_quadrature.tag)
    gradient_field = grid.real_field('gradient', 2, centroid_quadrature.tag)

    # Implemented
    input_field.data = random_initial_guess(grid, rng)
    fem = Linear2DFiniteElementInPixel()
    value_op = fem.create_field_value_approximation(centroid_quadrature.quad_pt_offset)
    evaluate_value = input_field.bind_mapping(value_op, value_field)
    gradient_op = fem.create_field_gradient_approximation(
        centroid_quadrature.quad_pt_offset, *grid.pixel_length
    )
    evaluate_gradient = input_field.bind_mapping(gradient_op, gradient_field)
    value = evaluate_value()
    gradient = evaluate_gradient()

    # Reference implementation
    fem_ref = CentroidLFM_SparseMatrix(grid)
    value_ref = fem_ref.interpolate_values(input_field.data.ravel()).reshape(value.shape)
    gradient_ref = fem_ref.interpolate_gradients(input_field.data.ravel()).reshape(gradient.shape)

    # Assertions
    assert np.allclose(value, value_ref)
    assert np.allclose(gradient, gradient_ref)


class CentroidLFM_SparseMatrix:
    """A reference implementation with SciPy SparseMatrix"""

    def interpolate_values(self, sample):
        return self._interp_centroid @ sample

    def interpolate_gradients(self, sample):
        return np.concatenate([self._interp_D_x @ sample, self._interp_D_y @ sample], axis=0)

    def __init__(self, grid: Grid):
        M, N = grid.nb_pixels_in_section
        MN = M * N

        # K_a maps nodal values to centroid of the triangles, which are the quadrature points
        coeff_interp = 1 / 3
        K_a_lower = sparse.lil_matrix((MN, MN), dtype=float)
        fill_cyclic_diagonal_2d(K_a_lower, (0, 0), (M, N), coeff_interp)
        fill_cyclic_diagonal_2d(K_a_lower, (0, 1), (M, N), coeff_interp)
        fill_cyclic_diagonal_2d(K_a_lower, (1, 0), (M, N), coeff_interp)

        K_a_upper = sparse.lil_matrix((MN, MN), dtype=float)
        fill_cyclic_diagonal_2d(K_a_upper, (0, 1), (M, N), coeff_interp)
        fill_cyclic_diagonal_2d(K_a_upper, (1, 0), (M, N), coeff_interp)
        fill_cyclic_diagonal_2d(K_a_upper, (1, 1), (M, N), coeff_interp)

        self._interp_centroid = sparse.vstack([K_a_lower, K_a_upper], format="csr")
        # self._sens_centroid = sparse.csr_matrix(self._interp_centroid.T)

        # K_b maps nodal values to the difference in x direction
        [dx, dy] = grid.pixel_length
        coeff_grad_x = 1 / dx
        K_b_lower = sparse.lil_matrix((MN, MN), dtype=float)
        fill_cyclic_diagonal_2d(K_b_lower, (0, 0), (M, N), -coeff_grad_x)
        fill_cyclic_diagonal_2d(K_b_lower, (1, 0), (M, N), coeff_grad_x)

        K_b_upper = sparse.lil_matrix((MN, MN), dtype=float)
        fill_cyclic_diagonal_2d(K_b_upper, (0, 1), (M, N), -coeff_grad_x)
        fill_cyclic_diagonal_2d(K_b_upper, (1, 1), (M, N), coeff_grad_x)

        self._interp_D_x = sparse.vstack([K_b_lower, K_b_upper], format="csr")
        # self._sens_D_x = sparse.csr_matrix(self._interp_D_x.T)

        # K_c maps nodal values to the difference in y direction
        coeff_grad_y = 1 / dy
        K_c_lower = sparse.lil_matrix((MN, MN), dtype=float)
        fill_cyclic_diagonal_2d(K_c_lower, (0, 0), (M, N), -coeff_grad_y)
        fill_cyclic_diagonal_2d(K_c_lower, (0, 1), (M, N), coeff_grad_y)

        K_c_upper = sparse.lil_matrix((MN, MN), dtype=float)
        fill_cyclic_diagonal_2d(K_c_upper, (1, 0), (M, N), -coeff_grad_y)
        fill_cyclic_diagonal_2d(K_c_upper, (1, 1), (M, N), coeff_grad_y)

        self._interp_D_y = sparse.vstack([K_c_lower, K_c_upper], format="csr")
        # self._sens_D_y = sparse.csr_matrix(self._interp_D_y.T)


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
    [N1, N2] = N
    [i1, i2] = np.mgrid[:N1, :N2]
    i1 = i1.ravel()
    i2 = i2.ravel()

    # Unpack offsets
    j1, j2 = j

    # Here 'N2' is assumed the contiguous dimension
    mat[i1 * N2 + i2, (i1 + j1) % N1 * N2 + (i2 + j2) % N2] = val
