"""
Tests of the `modelling.py` file.
"""

import math
import sys

import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

from a_package.simulating import *
from .utilities import *


rng = random.default_rng()
show_me = False


def test_capillary_bridge_compute_energy_jacobian():
    grid_shape = 5, 5
    bridge = create_capillary_bridge(grid_shape)

    # Determine the lowest step by machine precision
    lowest_magnitude = math.floor(0.5 * math.log10(sys.float_info.epsilon))
    highest_magnitude = 1
    # All the mini-steps to evaluate the numerical jacobian, for a view of accuracy
    deltas = np.power(10.0, np.arange(lowest_magnitude, highest_magnitude + 1))

    # Compute jacobian numerically (central difference)
    phi = bridge.phase_nodal_field.data
    field_shape = np.shape(phi)
    numeric_jacobian = np.empty((deltas.size, *field_shape))
    for idx_delta, delta in enumerate(deltas):
        numeric_jacobian[idx_delta] = np.squeeze(
            central_difference_jacobian(bridge.compute_energy, phi, delta)
        )

    # Compute jacobian from the implementation
    implemented_jacobian = bridge.compute_energy_jacobian(phi)

    # Measure the difference
    diffs = np.linalg.norm(
        abs(implemented_jacobian.squeeze() - numeric_jacobian.squeeze()),
        axis=tuple(range(-len(grid_shape), 0)),
    )

    # Plots
    if show_me:
        print(f"Numerical\n{numeric_jacobian}")
        print(f"Implemented\n{implemented_jacobian}")
        fig = plot_precisions(deltas, diffs)
        plt.show()

    # Assertion
    eps = 1e-6
    # here choose the minimal value because accuracy varies w.r.t the step size
    assert np.amin(diffs) < eps, f"The difference exceeds the tolerance {eps:.2e}"


def test_capillary_bridge_compute_volume_jacobian():
    grid_shape = 5, 5
    bridge = create_capillary_bridge(grid_shape)

    # Determine the lowest step by machine precision
    lowest_magnitude = math.floor(0.5 * math.log10(sys.float_info.epsilon))
    highest_magnitude = 1
    # All the mini-steps to evaluate the numerical jacobian, for a view of accuracy
    deltas = np.power(10.0, np.arange(lowest_magnitude, highest_magnitude + 1))

    # Compute jacobian numerically (central difference)
    phi = bridge.phase_nodal_field.data
    field_shape = np.shape(phi)
    numeric_jacobian = np.empty((deltas.size, *field_shape))
    for idx_delta, delta in enumerate(deltas):
        numeric_jacobian[idx_delta] = np.squeeze(
            central_difference_jacobian(bridge.compute_volume, phi, delta)
        )

    # Compute jacobian from the implementation
    implemented_jacobian = bridge.compute_volume_jacobian(phi)

    # Measure the difference
    diffs = np.linalg.norm(
        abs(implemented_jacobian.squeeze() - numeric_jacobian.squeeze()),
        axis=tuple(range(-len(grid_shape), 0)),
    )

    # Plots
    if show_me:
        print(f"Numerical\n{numeric_jacobian}")
        print(f"Implemented\n{implemented_jacobian}")
        fig = plot_precisions(deltas, diffs)
        plt.show()

    # Assertion
    eps = 1e-6
    # here choose the minimal value because accuracy varies w.r.t the step size
    assert np.amin(diffs) < eps, f"The difference exceeds the tolerance {eps:.2e}"


def test_jacobian_in_constant_liquid_volume_formulation():
    grid_shape = 5, 5
    bridge = create_capillary_bridge(grid_shape)
    grid = bridge.grid

    V_l = 0.5 * np.multiply.reduce(grid.length)
    bridge.formulate_with_constant_volume(V_l)

    phi = random_initial_guess(grid, rng)
    l_D_x = bridge.solver.dx_l
    lam = 0.0
    c = 0.01
    implemented_jacobian = l_D_x(phi, lam, c)

    # Determine the lowest step by machine precision
    lowest_magnitude = math.floor(0.5 * math.log10(sys.float_info.epsilon))
    highest_magnitude = 1
    # All the mini-steps to evaluate the numerical jacobian, for a view of accuracy
    deltas = np.power(10.0, np.arange(lowest_magnitude, highest_magnitude + 1))

    # Wrap the call function to be single argument
    l = bridge.solver.l
    def call_with_single_arg(phi):
        return l(phi, lam, c)

    # Compute jacobian numerically
    field_shape = np.shape(phi)
    numeric_jacobian = np.empty((deltas.size, *field_shape))
    for idx_delta, delta in enumerate(deltas):
        numeric_jacobian[idx_delta] = np.squeeze(
            central_difference_jacobian(call_with_single_arg, phi, delta)
        )

    # Measure the difference
    diffs = np.linalg.norm(
        abs(implemented_jacobian.squeeze() - numeric_jacobian.squeeze()),
        axis=tuple(range(-len(grid_shape), 0)),
    )

    # Plots
    if show_me:
        print(f"Numerical\n{numeric_jacobian}")
        print(f"Implemented\n{implemented_jacobian}")
        fig = plot_precisions(deltas, diffs)
        plt.show()

    # Assertion
    eps = 1e-6
    # here choose the minimal value because accuracy varies w.r.t the step size
    assert np.amin(diffs) < eps, f"The difference exceeds the tolerance {eps:.2e}"


def create_capillary_bridge(nb_pixels: list[int]):
    """With spherical top and flat substrate."""
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
    phi = rng.random([1, 1, *grid.nb_pixels_in_section])

    # Required model
    z = 3 * a
    solid_solid = SolidSolidContact(h0, h1, z)

    eta = 1 * a
    gamma = np.cos(np.pi / 3)
    vapour_liquid = CapillaryVapourLiquid(eta, gamma, solid_solid.gap_height())

    # The solver for optimization
    e_conv = 1e-6
    e_volume = 1e-8
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
