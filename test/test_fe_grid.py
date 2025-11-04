import numpy as np
import scipy.sparse as sparse

from a_package.grid import Grid
from a_package.numeric.fem import LinearFiniteElementPixel, FirstOrderElement


def test_first_order_element():
    # set up
    test_pts = np.array([[0.25, 0.25], [0.5, 0.5], [0.75, 0.75]])
    nb_element = 4
    field_data = np.diagflat(np.arange(nb_element) + 1)

    # implementation (parallel) to test
    grid = Grid([1., 1.], [nb_element, nb_element])
    fe = FirstOrderElement(grid, test_pts)
    field_in_parallel = np.asarray(field_data[grid.subdomain_slice], dtype=float, order="F")
    field_value = fe.interpolate_value(field_in_parallel)
    field_gradient = fe.interpolate_gradient(field_in_parallel)
    field_value_sens = fe.propag_sens_value(field_value)
    field_gradient_sens = fe.propag_sens_gradient(field_gradient)

    # reference implementation (serial)
    global_grid = GlobalGrid([1., 1.], [nb_element, nb_element])
    ref_fe = ReferenceFirstOrderElement(global_grid, test_pts)
    field_in_serial = np.asarray(field_data, dtype=float, order="C")
    expected_field_value = ref_fe.interpolate_value(field_in_serial)
    expected_field_gradient = ref_fe.interpolate_gradient(field_in_serial)
    expected_field_value_sens = ref_fe.propag_sens_value(expected_field_value)
    expected_field_gradient_sens = ref_fe.propag_sens_gradient(expected_field_gradient)

    # assertions
    assert np.allclose(field_value, expected_field_value[grid.subdomain_slice])
    assert np.allclose(field_gradient, expected_field_gradient[grid.subdomain_slice])
    assert np.allclose(field_value_sens, expected_field_value_sens[grid.subdomain_slice])
    assert np.allclose(field_gradient_sens, expected_field_gradient_sens[grid.subdomain_slice])


class GlobalGrid:

    def __init__(self, lengths: list[float], nb_elements: list[int]):
        self.nb_elements = nb_elements
        self.element_sizes = [l/nb for [l, nb] in zip(lengths, nb_elements)]


class ReferenceFirstOrderElement:
    """A reference, implemented with SciPy sparse matrix"""

    def __init__(self, grid: GlobalGrid, target_pts: np.ndarray) -> None:
        self.field_shape = grid.nb_elements
        self.nb_target_pts = target_pts.shape[0]

        nb_total_elements = np.multiply.reduce(grid.nb_elements)
        fe_pixel = LinearFiniteElementPixel()

        val_interp_coeffs = fe_pixel.compute_value_interpolation_coefficients(target_pts)
        sub_matrices = []
        for [_, coeffs] in enumerate(val_interp_coeffs):
            sub_matrix = sparse.lil_matrix((nb_total_elements, nb_total_elements), dtype=float)
            for [nodal_idxs, weight] in coeffs.items():
                self.fill_cyclic_diagonal_pseudo_2d(sub_matrix, nodal_idxs, grid.nb_elements, weight, row_maj=True)
            sub_matrices.append(sub_matrix)
        self.value_op_matrix = sparse.vstack(sub_matrices, format="csr")

        grad_interp_coeffs = fe_pixel.compute_gradient_interpolation_coefficients(target_pts)
        sub_matrices = []
        for [_, coeffs] in enumerate(grad_interp_coeffs):
            sub_matrix_x1 = sparse.lil_matrix((nb_total_elements, nb_total_elements), dtype=float)
            for [nodal_idxs, weight] in coeffs["x1"].items():
                self.fill_cyclic_diagonal_pseudo_2d(sub_matrix_x1, nodal_idxs, grid.nb_elements, weight, row_maj=True)
            sub_matrices.append(sub_matrix_x1 / grid.element_sizes[0])
        for [_, coeffs] in enumerate(grad_interp_coeffs):
            sub_matrix_x2 = sparse.lil_matrix((nb_total_elements, nb_total_elements), dtype=float)
            for [nodal_idxs, weight] in coeffs["x2"].items():
                self.fill_cyclic_diagonal_pseudo_2d(sub_matrix_x2, nodal_idxs, grid.nb_elements, weight, row_maj=True)
            sub_matrices.append(sub_matrix_x2 / grid.element_sizes[1])
        self.gradient_op_matrix = sparse.vstack(sub_matrices, format="csr")

    def interpolate_value(self, field: np.ndarray):
        return (self.value_op_matrix @ field.ravel()).reshape(-1, self.nb_target_pts, *self.field_shape)

    def interpolate_gradient(self, field: np.ndarray):
        return (self.gradient_op_matrix @ field.ravel()).reshape(-1, self.nb_target_pts, *self.field_shape)

    def propag_sens_value(self, field: np.ndarray):
        return (field.ravel() @ self.value_op_matrix).reshape(-1, *self.field_shape)

    def propag_sens_gradient(self, field: np.ndarray):
        return (field.ravel() @ self.gradient_op_matrix).reshape(-1, *self.field_shape)

    @staticmethod
    def fill_cyclic_diagonal_pseudo_2d(
            mat: sparse.lil_matrix, j: tuple[int, int],
            N: tuple[int, int],
            val: float, row_maj: bool = True):
        """Fill cyclically, element-wise in the j-th diagonal of a matrix.
        The matrix represents a mapping from 2D data to 2D data.
        However, the 2D data is ravelled and represented as a 1D array.
        """
        assert mat.ndim == 2

        # cartesian product of range(N1) and range(N2)
        [N1, N2] = N
        [i1, i2] = np.mgrid[:N1, :N2]
        i1 = i1.ravel()
        i2 = i2.ravel()

        [j1, j2] = j
        if row_maj:
            # the 2D data is flattened with row-major (contiguous in 2nd axis)
            mat[i1 * N2 + i2, (i1 + j1) % N1 * N2 + (i2 + j2) % N2] = val
        else:
            # the 2D data is flattened with column-major (contiguous in 1st axis)
            mat[i1 + i2 * N1, (i1 + j1) % N1 + (i2 + j2) % N2 * N1] = val
