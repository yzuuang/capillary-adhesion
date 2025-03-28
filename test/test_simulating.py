"""
Tests of the `modelling.py` file.
"""

import math
import sys

import numpy as np
import matplotlib.pyplot as plt

from a_package.simulating import *


show_me_plot = False


def test_capillary_bridge_compute_energy_jacobian():
    grid_shape = 10, 10
    bridge = create_capillary_bridge(grid_shape)

    # Determine the lowest step by machine precision
    lowest_magnitude = math.floor(
        0.5 * math.log10(sys.float_info.epsilon)
    )
    highest_magnitude = 0
    # All the mini-steps to evaluate the numerical jacobian, for a view of accuracy
    deltas = np.pow(10.0, np.arange(lowest_magnitude, highest_magnitude + 1))

    # Compute jacobian numerically (central difference)
    phi: Field2D = bridge.phase_field.field_sample
    field_shape = np.shape(phi.p)
    numeric_jacobian = np.empty((deltas.size, *field_shape))
    for idx_delta, delta in enumerate(deltas):
        for locs in np.ndindex(field_shape):
            ref_val = np.copy(phi.p[locs])
            phi.p[locs] = ref_val + delta
            plus_val = bridge.compute_energy(phi.p).item()
            phi.p[locs] = ref_val - delta
            minus_val = bridge.compute_energy(phi.p).item()
            numeric_jacobian[(idx_delta, *locs)] = (plus_val - minus_val) / delta * 0.5
            phi.p[locs] = ref_val

    # Compute jacobian from the implementation
    impl_jacobian = bridge.compute_energy_jacobian(phi.p)

    # Measure the difference
    diffs = np.amax(abs(impl_jacobian[np.newaxis, :] - numeric_jacobian), axis=1)

    # Plots
    if show_me_plot:
        plt.plot(
            deltas,
            diffs,
            "x-",
            label=r"Difference from a numerical method of $\mathcal{O}(\delta^2)$",
        )

        plt.loglog()
        plt.xlabel(r"$\delta$")
        plt.ylabel(r"$\varepsilon$")
        plt.legend()

        plt.show()

    # Assertion
    eps = 1e-6
    assert np.amin(diffs) < eps, f"The difference exceeds the tolerance {eps:.2e}"


def create_capillary_bridge(grid_shape: list[int]):
    # Region
    grid_spacing = 1.0
    grid = Grid(grid_spacing, grid_shape, len(grid_shape))

    # Upper solid has a sphere surface
    [x, y] = (grid_spacing * np.arange(nb) for nb in grid.section.nb_pixels)
    [xm, ym] = np.meshgrid(x, y)
    [Lx, Ly] = np.array(grid.section.nb_pixels) * grid_spacing
    R = 2 * min(Lx, Ly)
    h1 = np.sqrt(R**2 - (xm - 0.5 * Lx) ** 2 - (ym - 0.5 * Ly) ** 2)

    # Lower solid is flat
    h0 = np.zeros(grid.section.nb_pixels)

    # A phase field
    phi = np.array(grid.section.nb_pixels)

    # Required model
    z = 3 * grid_spacing
    solid_solid = SolidSolidContact(z, h0)

    eta = 1 * grid_spacing
    gamma = np.cos(np.pi / 3)
    vapour_liquid = CapillaryVapourLiquid(eta, gamma, solid_solid.gap_height(h1))


    # Required numerics
    height_upper = CubicSpline(grid, h0)
    height_lower = CubicSpline(grid, h1)
    phase_field = LinearFiniteElement(grid, phi)
    quadrature = CentroidQuadrature(grid)

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
        height_lower,
        height_upper,
        vapour_liquid,
        phase_field,
        quadrature,
        solver,
    )
    return bridge


if __name__ == "__main__":
    test_capillary_bridge_compute_energy_jacobian()
