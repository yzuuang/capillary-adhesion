import numpy as np
import numpy.linalg as la
import scipy.optimize as optimize
import SurfaceTopography.Uniform.GeometryAnalysis as ga


def _borders(n: int):
    """Indices to split an array of size `n` into 2, 4, 8, ..., k (2^(k+1) >= n) parts."""
    n_sect = 2
    indices = [0]
    while n_sect < n:
        div = n // n_sect
        indices.extend([index + div for index in indices])
        indices.sort()
        yield indices
        n_sect *= 2


def _count(blocks: list[np.ndarray]):
    """Count the number of blocks which are marked."""
    return sum(np.logical_or.reduce(elem, axis=None) for elem in blocks)


def box_count_dimension(mark: np.ndarray):
    """
    :param mark: either boolean or non-negative integer
    """
    nx, ny = mark.shape
    assert nx == ny, "Only implemented for square-shape data"

    size = []
    num = []
    for ix in _borders(nx):
        rows = np.split(mark, ix, axis=0)
        size.append(nx / len(ix))
        num.append(sum(_count(np.split(row, ix, axis=1)) for row in rows))

    return np.array(size), np.array(num)


def least_squares(x: np.ndarray, y: np.ndarray):
    """y = a x^b <=> logy = loga + b logx; parameter fitting with y = Phi theta formulation."""
    # def err(p, x, y):
    #     return np.log(y) - (np.log(p[0]) + p[1] * np.log(x))
    # res = optimize.least_squares(err, [1, -1], args=(x, y))
    # return res['x']
    y = np.log2(y)
    phi = np.column_stack([np.ones_like(x), np.log2(x)])
    theta = la.pinv(phi) @ y
    theta[0] = np.exp2(theta[0])
    return theta


def gridwise_area_dimension(patch: np.ndarray):
    l_box, n_box = box_count_dimension(patch)
    pref, slope = least_squares(l_box**2, n_box)
    return pref, 1 - slope


def gridwise_perimeter_dimension(patch: np.ndarray, outer: bool=True):
    if outer:
        peri = ga.outer_perimeter_area(patch, True)
    else:
        peri = ga.inner_perimeter_area(patch, True)

    l_box, n_box = box_count_dimension(peri)
    pref, slope = least_squares(l_box, n_box)
    return pref, 1 - slope


def patchwise_area_dimension(patch: np.ndarray):
    n_patch, patch = ga.assign_patch_numbers_area(patch, True)
    dims = np.empty((n_patch, 2))
    for i_patch in np.arange(n_patch):
        mark = patch == i_patch + 1  # the non-patch area is marked with 0; first index for patches is 1
        dims[i_patch] = gridwise_area_dimension(mark)
    return dims


def patchwise_perimeter_dimension(patch: np.ndarray):
    n_patch, patch = ga.assign_patch_numbers_area(patch, True)
    dims = np.empty((n_patch, 2))
    for i_patch in np.arange(n_patch):
        mark = patch == i_patch + 1  # the non-patch area is marked with 0; first index for patches is 1
        dims[i_patch] = gridwise_perimeter_dimension(mark)
    return dims
