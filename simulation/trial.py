import os
import numpy as np
from a_package.storing import working_directory
from a_package.simulating import *


def main():
    base_path = __file__.replace(".py", "_results")
    data_path = os.path.join(base_path, "data")
    figures_path = os.path.join(base_path, "figures")

    simulate(data_path)
    animate(data_path, figures_path)


def simulate(data_path: str):
    # Region
    grid_spacing = 1.0
    spatial_dims = 2
    nb_grid_pts = [8] * spatial_dims
    grid = Grid(grid_spacing, nb_grid_pts, spatial_dims)

    # Surface roughness
    C0 = 1e6  # prefactor
    qR = 2e0  # roll-off
    qS = 2e1  # cut-off
    H = 0.95  # Hurst exponent
    roughness = SelfAffineRoughness(C0, qR, qS, H)

    # Two planes based on the roughness model & random patternW
    seed = None
    rng = get_rng(grid, seed)
    h0 = generate_height_profile(grid, roughness, rng)
    h1 = generate_height_profile(grid, roughness, rng)

    # The gap between contact
    z1 = 2 * grid_spacing
    contact = SolidSolidContact(z1, h0)
    gap_height = contact.gap_height(h1)

    # The capillary phase field
    eta = 0.1
    gamma = 0.1
    vapour_liquid = CapillaryVapourLiquid(eta, gamma, gap_height)

    # Random initial values of phase field
    phi_init = random_uniform(grid, rng)

    # Specify numerical methods
    h0 = CubicSpline(grid, h0)
    h1 = CubicSpline(grid, h1)
    quadrature = CentroidQuadrature(grid)
    phi = LinearFiniteElement(grid, phi_init)

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
        grid, contact, h0, h1, vapour_liquid, phi, quadrature, solver
    )

    # Directory path for saving data
    with working_directory(data_path, read_only=False) as store:
        # Specify volume and formulate the numopt
        liquid_volume = 0.2
        bridge.formulate_with_constant_volume(liquid_volume)

        # Trajectory of the relative displacement
        z_trajectory = np.array([3, 2]) * grid_spacing
        trajectory = np.column_stack(
            [np.zeros_like(z_trajectory), np.zeros_like(z_trajectory), z_trajectory]
        )

        # Solve the configuration with given trajectory
        bridge.simulate_with_trajectory(trajectory, phi_init, store)


def animate(data_path: str, figures_path: str):
    pass


if __name__ == "__main__":
    main()
