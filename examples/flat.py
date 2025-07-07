import os
import numpy as np
import matplotlib.pyplot as plt

from a_package.storing import working_directory
from a_package.simulating import *


show_me = False


def main():
    base_path = __file__.replace(".py", "_results")
    data_path = os.path.join(base_path, "data")
    figures_path = os.path.join(base_path, "figures")

    simulate(data_path)
    animate(data_path, figures_path)


def simulate(data_path: str):
    # Region
    nb_spatial_dims = 2
    domain_length = [1.0] * nb_spatial_dims
    nb_pixels = [50] * nb_spatial_dims
    nb_subdivisions = [1, 1]
    nb_ghost_layers = [1, 1]
    grid = Grid(domain_length, nb_pixels, nb_subdivisions, nb_ghost_layers)
    a = np.maximum.reduce(grid.pixel_length)

    # Two flat planar solid
    h0 = np.zeros([1, 1, *grid.nb_pixels_in_section])
    h1 = np.zeros([1, 1, *grid.nb_pixels_in_section])

    # The gap between contact
    delta = 2 * a
    solid_solid = SolidSolidContact(h0, h1, delta)
    gap_height = solid_solid.gap_height()
    if show_me:
        fig, ax = plt.subplots()

        g = np.squeeze(gap_height) / a
        vmax = g.max()
        vmin = 0
        [lx, ly] = grid.length
        border = np.array([0, lx / a, 0, ly / a])
        im = ax.imshow(
            g,
            cmap='afmhot',
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            extent=border,
        )
        fig.colorbar(im)

        plt.show()

    # The capillary phase field
    eta = a
    gamma = 0.1
    liquid_vapour = CapillaryVapourLiquid(eta, gamma, gap_height)

    # Random initial values of phase field
    rng = get_rng(grid, seed=None)
    phi_init = random_initial_guess(grid, rng)

    # The solver for optimization
    e_conv = 1e-6
    e_volume = 1e-4
    max_iter = 3000
    c0 = 1e-3
    beta = 3.0
    k_max = 20
    solver = AugmentedLagrangian(e_conv, e_volume, max_iter, c0, beta, k_max)

    # Capillary Bridge
    bridge = CapillaryBridge(grid, solid_solid, liquid_vapour, solver, h0, h1, phi_init)

    # Directory path for saving data
    with working_directory(data_path, read_only=False) as store:
        # Specify volume and formulate the numopt
        liquid_volume = 0.2
        bridge.formulate_with_constant_volume(liquid_volume)

        # Trajectory of the relative displacement
        z_trajectory = np.array([3, 2]) * delta
        trajectory = np.column_stack(
            [np.zeros_like(z_trajectory), np.zeros_like(z_trajectory), z_trajectory]
        )

        # Solve the configuration with given trajectory
        bridge.simulate_with_trajectory(trajectory, phi_init, store)


def animate(data_path: str, figures_path: str):
    pass


if __name__ == "__main__":
    main()
