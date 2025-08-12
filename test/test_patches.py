# import numpy as np
# import numpy.random as random
# import matplotlib.pyplot as plt

# from a_package.patches import *


# show_me_plots = False
# _rng = random.default_rng()


# def test_area_fractal_dimension():
#     # Generate a square patch
#     n_pixel = 2048
#     square = random_square_patch(n_pixel)

#     # Curve fitting to find the dimension
#     l_box, n_box = box_count_dimension(square)
#     pref, slope = fit_power_curve(l_box, n_box, is_area=True)
#     dim = -slope
#     print(f"Dimension={dim}")

#     if show_me_plots:
#         fig, ax = plt.subplots()
#         plot_patches_dimension_fitting(ax, l_box, n_box, pref, slope)
#         fig.tight_layout()
#         plt.show()

#     # Compare to reference value
#     ref_val = 2.0
#     tol = 0.1
#     assert np.isclose(dim, ref_val, atol=tol), \
#         f"The value of area fractal dimension is not in the range of {ref_val:.2f}+-{tol:.2f}."


# def test_perimeter_fractal_dimension():
#     # Generate a square patch
#     n_pixel = 2048
#     square = random_square_patch(n_pixel)

#     # Get the perimeter
#     square_border =  ga.outer_perimeter_area(square, periodic=True)

#     # Get the dimension via curve fittingd
#     l_box, n_box = box_count_dimension(square_border)
#     pref, slope = fit_power_curve(l_box, n_box, is_area=False)
#     dim = -slope
#     print(f"Dimension={dim}")

#     if show_me_plots:
#         fig, ax = plt.subplots()
#         plot_patches_dimension_fitting(ax, l_box, n_box, pref, slope)
#         fig.tight_layout()
#         plt.show()

#     # Compare to reference value
#     ref_val = 1.0
#     tol = 0.02
#     assert np.isclose(dim, ref_val, atol=tol), \
#         f"The value of area fractal dimension is not in the range of {ref_val:.2f}+-{tol:.2f}."


# def random_square_patch(n_pixel: int):
#     """Generate a square randomly located in a 'n_pixel' by 'n_pixel' region."""
#     square = np.zeros((n_pixel, n_pixel), dtype=bool)
#     i_min = int(_rng.random() * 0.5 * n_pixel)
#     i_max = i_min + n_pixel//4
#     square[i_min:i_max, i_min:i_max] = 1
#     return square
