"""
Tests of the `modelling.py` file.
"""

import math
import sys

import numpy as np
import matplotlib.pyplot as plt

from a_package.simulating import *


show_me_plot = True


def test_capillary_bridge_compute_energy_jacobian():
    grid_shape = 10, 10
    bridge = create_capillary_bridge(grid_shape)

    # Determine the lowest step by machine precision
    lowest_magnitude = math.floor(0.5 * math.log10(sys.float_info.epsilon))
    highest_magnitude = 0
    # All the mini-steps to evaluate the numerical jacobian, for a view of accuracy
    deltas = np.power(10.0, np.arange(lowest_magnitude, highest_magnitude + 1))

    # Compute jacobian numerically (central difference)
    phi = bridge.phase_nodal_field.data
    field_shape = np.shape(phi)
    numeric_jacobian = np.empty((deltas.size, *field_shape))
    for idx_delta, delta in enumerate(deltas):
        for locs in np.ndindex(field_shape):
            ref_val = np.copy(phi[locs])
            phi[locs] = ref_val + delta
            plus_val = bridge.compute_energy(phi).item()
            phi[locs] = ref_val - delta
            minus_val = bridge.compute_energy(phi).item()
            numeric_jacobian[(idx_delta, *locs)] = (plus_val - minus_val) / delta * 0.5
            phi[locs] = ref_val

    # Compute jacobian from the implementation
    implemented_jacobian = bridge.compute_energy_jacobian(phi)

    # Measure the difference
    diffs = np.linalg.norm(
        abs(implemented_jacobian.squeeze() - numeric_jacobian.squeeze()),
        axis=tuple(range(-len(grid_shape), 0)),
    )

    # Plots
    if show_me_plot:
        plt.plot(
            deltas,
            diffs,
            "x-",
            label=r"Difference from a numerical method of $\mathcal{O}(\delta^2)$",
        )

        plt.loglog()
        plt.xlabel(r"Step size $\delta$")
        plt.ylabel(r"Deviation $\varepsilon$")
        plt.legend()

        plt.show()

    # Assertion
    eps = 1e-6
    # here choose the minimal value because accuracy varies w.r.t the step size
    assert np.amin(diffs) < eps, f"The difference exceeds the tolerance {eps:.2e}"


def create_capillary_bridge(nb_pixels: list[int]):
    # Region
    domain_size = [1.0, 1.0]
    grid = Grid(domain_size, nb_pixels, [1, 1], [1, 1])
    [a, _] = grid.pixel_length

    # Upper solid has a sphere surface
    [x, y] = tuple(a * np.arange(nb) for nb in grid.nb_pixels_in_section)
    [xm, ym] = np.meshgrid(x, y)
    [Lx, Ly] = np.array(grid.nb_pixels_in_section) * a
    R = 2 * min(Lx, Ly)
    h1 = np.sqrt(R**2 - (xm - 0.5 * Lx) ** 2 - (ym - 0.5 * Ly) ** 2)
    h1 = np.expand_dims(h1, axis=[0, 1])

    # Lower solid is flat
    h0 = np.zeros([1, 1, *grid.nb_pixels_in_section])

    # A phase field
    phi = np.ones([1, 1, *grid.nb_pixels_in_section])

    # Required model
    z = 3 * a
    solid_solid = SolidSolidContact(h0, h1, z)

    eta = 1 * a
    gamma = np.cos(np.pi / 3)
    vapour_liquid = CapillaryVapourLiquid(eta, gamma, solid_solid.gap_height())

    # The solver for optimization
    e_conv = 1e-8
    e_volume = 1e-6
    max_iter = 3000
    c0 = 1e-2
    beta = 3.0
    k_max = 20
    solver = AugmentedLagrangian(e_conv, e_volume, max_iter, c0, beta, k_max)

    # Capillary Bridge
    bridge = CapillaryBridge(
        grid,
        solid_solid,
        vapour_liquid,
        solver,
        h0,
        h1,
        phi,
    )
    return bridge


if __name__ == "__main__":
    test_capillary_bridge_compute_energy_jacobian()
