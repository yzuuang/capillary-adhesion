"""
Load-unload simulation case.

Usage:
    python -m cases.load_unload config.toml [config2.toml ...]

The first config file provides base parameters. Additional config files
can override specific values (useful for sweep specifications).
"""

import os
import sys
import logging

import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np
import numpy.random as random

from a_package.config import load_config, save_config, expand_sweeps, count_sweep_combinations, get_surface_shape, Config
from a_package.domain import Grid
from a_package.physics.surfaces import generate_surface
from a_package.simulation.formulation import NodalFormCapillary
from a_package.simulation.simulation import Simulation
from a_package.simulation.runner import (
    create_grid_from_config,
    build_capillary_args,
    build_solver_args,
    build_trajectory,
    compute_liquid_volume,
)
from a_package.runtime.dirs import RunDir, register_run
from a_package.runtime.logging import reset_logging, switch_log_file

from cases.visualise_onerun import create_overview_animation


show_me_preview = False
logger = logging.getLogger(__name__)


def main():
    reset_logging()

    if len(sys.argv) < 2:
        print("Usage: python -m cases.load_unload config.toml")
        sys.exit(1)

    config_file = sys.argv[1]
    config = load_config(config_file)

    # visual check
    if show_me_preview:
        preview_surface_and_gap(config)

    # setup run directory
    case_name = os.path.splitext(os.path.basename(__file__))[0]
    upper_shape = get_surface_shape(config.geometry.upper)
    lower_shape = get_surface_shape(config.geometry.lower)
    shape_name = f'{upper_shape}-on-{lower_shape}'
    base_dir = os.path.join(case_name, shape_name)
    run = register_run(base_dir, __file__, config_file)

    # check if parameter sweep is specified in config
    nb_configs = count_sweep_combinations(config)
    if nb_configs == 1:
        # No sweeps, single run
        switch_log_file(run.log_file)
        io = run_one_trip(run, config)
        create_overview_animation(run.path, io.grid)
    else:
        # Parameter sweep
        for index, expanded_config in enumerate(expand_sweeps(config)):
            sub_run = register_run(run.intermediate_dir, __file__, with_hash=False)
            switch_log_file(sub_run.log_file)
            logger.info(f"Run #{index + 1} of {nb_configs} runs.")
            # Save the expanded config for reproducibility
            save_config(expanded_config, sub_run.parameters_dir / f"subrun-{index}.toml")
            io = run_one_trip(sub_run, expanded_config)
            create_overview_animation(sub_run.path, io.grid)


def preview_surface_and_gap(config: Config):
    """A visual check before running simulations."""
    grid = create_grid_from_config(config)
    h1 = generate_surface(grid, config.geometry.upper.shape, **config.geometry.upper.params)
    h0 = generate_surface(grid, config.geometry.lower.shape, **config.geometry.lower.params)
    trajectory = build_trajectory(config)

    # create the figure and axes
    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)

    def update_frame(i_frame: int):
        # clear content of last frames
        for ax in (ax1, ax2):
            ax.clear()

        [xm, ym] = grid.form_nodal_mesh()
        a = min(grid.element_sizes)
        # 3D surface plot of upper and lower rigid body
        ax1.plot_surface(xm, ym, h0 / a, cmap="berlin")
        ax1.plot_surface(xm, ym, (h1 + trajectory[i_frame]) / a, cmap="plasma")
        ax1.view_init(elev=0, azim=-45)
        ax1.set_xlabel(r"Position $x/a$")
        ax1.set_ylabel(r"Position $y/a$")
        ax1.set_zlabel(r"Position $z/a$")

        # 2D colour map of the gap
        h_diff = h1 - h0 + trajectory[i_frame]
        gap = np.clip(h_diff, 0, None)
        contact = np.ma.masked_where(gap > 0, gap)
        [nx, ny] = grid.nb_elements
        border = (0, nx, 0, ny)
        ax2.imshow(gap / a, vmin=0, interpolation="nearest", cmap="hot", extent=border)
        ax2.imshow(contact, cmap="Greys", vmin=-1, vmax=1, alpha=0.4, interpolation="nearest", extent=border)
        ax2.set_xlabel(r"Position $x/a$")
        ax2.set_ylabel(r"Position $y/a$")

        return *ax1.images, *ax2.images

    # draw the animation
    nb_steps = len(trajectory)
    _ = ani.FuncAnimation(fig, update_frame, nb_steps, interval=200, repeat_delay=3000)

    # allow to exit if it does not look right
    plt.show()
    skip = input("Run simulation [Y/n]? ").strip().lower() in ("n", "no")
    if skip:
        sys.exit(0)


def run_one_trip(run: RunDir, config: Config):
    """Run a single simulation with given configuration."""
    # Grid
    grid = create_grid_from_config(config)

    # Surfaces (bridge config -> physics primitives)
    upper = generate_surface(grid, config.geometry.upper.shape, **config.geometry.upper.params)
    lower = generate_surface(grid, config.geometry.lower.shape, **config.geometry.lower.params)

    # Trajectory
    trajectory = build_trajectory(config)

    # Capillary model args
    capi_args = build_capillary_args(config)

    # Solver args
    solver_args = build_solver_args(config)

    # Liquid volume from percentage specification
    volume = compute_liquid_volume(grid, config, upper, lower, capi_args, trajectory)

    # Run simulation
    phi_init = random.rand(1, 1, *grid.nb_elements)
    simulation = Simulation(grid, run.results_dir, capi_args, solver_args)
    return simulation.simulate_approach_retraction_with_constant_volume(
        upper, lower, volume, trajectory, phase_init=phi_init)


if __name__ == "__main__":
    main()
