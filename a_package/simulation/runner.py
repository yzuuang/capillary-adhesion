"""
Black-box simulation runner.

Provides a simple interface: config in -> results out.
"""

import logging
import pathlib
from typing import Any

import numpy as np
import numpy.random as random

from a_package.grid import Grid
from a_package.config import Config
from a_package.physics.surfaces import generate_surface
from a_package.simulation.simulation import Simulation
from a_package.simulation.formulation import NodalFormCapillary
from a_package.simulation.io import SimulationIO


logger = logging.getLogger(__name__)


def run_simulation(config: Config, output_dir: str | pathlib.Path) -> SimulationIO:
    """
    Run a simulation from a configuration.

    This is the main entry point for black-box simulation execution.
    Takes a config object and output directory, runs the full simulation,
    and returns the IO object for accessing results.

    Parameters
    ----------
    config : Config
        Complete simulation configuration.
    output_dir : str | Path
        Directory to store simulation results.

    Returns
    -------
    SimulationIO
        IO object for accessing saved results.
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create grid
    grid = create_grid_from_config(config)

    # Generate surfaces
    upper = generate_surface(grid, config.geometry.upper)
    lower = generate_surface(grid, config.geometry.lower)

    # Build capillary arguments
    capillary_args = build_capillary_args(config)

    # Build solver arguments
    solver_args = build_solver_args(config)

    # Build trajectory
    trajectory = build_trajectory(config)

    # Compute liquid volume
    volume = compute_liquid_volume(grid, config, upper, lower, capillary_args, trajectory)

    # Random initial phase field
    rng = random.default_rng()
    phase_init = rng.random((1, 1, *grid.nb_elements))

    # Run simulation
    logger.info(f"Starting simulation with output to {output_dir}")
    simulation = Simulation(grid, output_dir, capillary_args, solver_args)
    io = simulation.simulate_approach_retraction_with_constant_volume(
        upper, lower, volume, trajectory, phase_init=phase_init
    )

    return io


def create_grid_from_config(config: Config) -> Grid:
    """Create a Grid from configuration."""
    a = config.geometry.grid.pixel_size
    N = config.geometry.grid.nb_pixels
    L = a * N
    return Grid([L, L], [N, N])


def build_capillary_args(config: Config) -> dict[str, Any]:
    """Build capillary model arguments from configuration."""
    theta = (np.pi / 180) * config.physics.capillary.contact_angle_degree
    eta = config.physics.capillary.interface_thickness
    return {"eta": eta, "theta": theta}


def build_solver_args(config: Config) -> dict[str, Any]:
    """Build solver arguments from configuration."""
    solver = config.simulation.solver
    return {
        "max_inner_iter": solver.max_nb_iters,
        "max_outer_loop": solver.max_nb_loops,
        "tol_convergence": solver.tol_convergence,
        "tol_constraint": solver.tol_constraints,
        "init_penalty_weight": solver.init_penalty_weight,
    }


def build_trajectory(config: Config) -> np.ndarray:
    """Build separation trajectory from configuration."""
    traj = config.simulation.trajectory
    d_min = traj.min_separation
    d_max = traj.max_separation
    d_step = traj.step_size
    nb_steps = round((d_max - d_min) / d_step) + 1
    # Start from max (approach), go to min
    return np.linspace(d_max, d_min, nb_steps)


def compute_liquid_volume(
    grid: Grid,
    config: Config,
    upper: np.ndarray,
    lower: np.ndarray,
    capillary_args: dict[str, Any],
    trajectory: np.ndarray,
) -> float:
    """
    Compute liquid volume from percentage specification.

    The percentage is relative to the maximum possible volume
    (full liquid at minimum separation).
    """
    # Create formulation at minimum separation to compute reference volume
    formulation = NodalFormCapillary(grid, capillary_args)
    z_min = np.amin(trajectory)
    gap = np.clip(upper + z_min - lower, 0, None)
    formulation.set_gap(gap)

    # Full liquid phase field
    full_liquid = np.ones([1, 1, *grid.nb_elements])
    formulation.set_phase(full_liquid)

    # Compute volume as percentage of full
    V_percent = 0.01 * config.physics.capillary.liquid_volume_percent
    return formulation.get_volume() * V_percent
