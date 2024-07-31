import numpy as np
import matplotlib.pyplot as plt
import SurfaceTopography.Generation as gen
import SurfaceTopography.Uniform.GeometryAnalysis as ga

from a_package.patches import box_count_dimension, least_squares, \
    gridwise_area_dimension, gridwise_perimeter_dimension, patchwise_area_dimension, patchwise_perimeter_dimension


if __name__ == '__main__':
    topo = gen.fourier_synthesis((2048, 2048), (1, 1), 0.8, rms_height=1, long_cutoff=0.1)
    # topo.plot()

    patch = topo.heights() > 2
    plt.figure()
    plt.subplot(111, aspect=1)
    plt.pcolormesh(patch)

    l, n = box_count_dimension(patch)
    plt.figure()
    plt.loglog(l**2, n, "x")
    theta = least_squares(l**2, n)
    print(theta)
    plt.loglog(l, theta[0] * (l**2)**theta[1], "--")
    plt.title("Area box count - Box dimension")

    plt.figure()
    peri = ga.outer_perimeter_area(patch, True)
    l, n = box_count_dimension(peri)
    plt.loglog(l, n, "x")
    theta = least_squares(l, n)
    print(theta)
    plt.loglog(l, theta[0] * l**theta[1], "--")
    plt.title("Perimeter box count - Box dimension")

    plt.show()
