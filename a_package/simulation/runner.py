"""
Black-box simulation runner.

Provides a simple interface: config in -> results out.

This module is the orchestration layer that bridges config (raw dicts) to
physics/numerics classes. All semantic knowledge of parameter translation
lives here.
"""

import logging
import pathlib
from typing import Any

import numpy as np
import numpy.random as random

from a_package.domain import Grid
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

    # Generate surfaces (bridge config -> physics primitives)
    upper = generate_surface_from_config(grid, config.physics["upper"])
    lower = generate_surface_from_config(grid, config.physics["lower"])

    # Build capillary arguments
    capillary_args = build_capillary_args(config)

    # Build solver arguments
    solver_args = build_solver_args(config)

    # Build trajectory
    trajectory = build_trajectory(config)

    # Random initial phase field
    rng = random.default_rng()
    phase_init = rng.random((1, 1, *grid.nb_elements))

    # Create simulation
    logger.info(f"Starting simulation with output to {output_dir}")
    simulation = Simulation(grid, output_dir, capillary_args, solver_args)

    # Dispatch based on constraint type
    constraint_cfg = config.simulation["constraint"]
    constraint_type = constraint_cfg["type"]

    if constraint_type == "constant_volume":
        volume = compute_liquid_volume(
            grid, constraint_cfg, upper, lower, capillary_args, trajectory
        )
        io = simulation.run_with_constant_volume(
            upper, lower, trajectory, volume, phase_init=phase_init
        )
    else:
        raise ValueError(f"Unknown constraint type: {constraint_type}")

    return io


def create_grid_from_config(config: Config) -> Grid:
    """Create a Grid from configuration."""
    grid_cfg = config.domain["grid"]
    a = grid_cfg["pixel_size"]
    N = grid_cfg["nb_pixels"]
    L = a * N
    return Grid([L, L], [N, N])


def generate_surface_from_config(grid: Grid, surface_cfg: dict[str, Any]) -> np.ndarray:
    """
    Generate a surface from configuration dict.

    Extracts shape and passes remaining params to generate_surface.
    """
    cfg = dict(surface_cfg)  # copy to avoid mutation
    shape = cfg.pop("shape")
    return generate_surface(grid, shape, **cfg)


def build_capillary_args(config: Config) -> dict[str, Any]:
    """
    Build capillary model arguments from configuration.

    Translates user-facing config parameters to physics class parameters:
    - contact_angle_degree -> theta (radians)
    - interface_thickness -> eta
    """
    capillary = config.physics["capillary"]
    theta = (np.pi / 180) * capillary["contact_angle_degree"]
    eta = capillary["interface_thickness"]
    return {"eta": eta, "theta": theta}


def build_solver_args(config: Config) -> dict[str, Any]:
    """
    Build solver arguments from configuration.

    Translates user-facing config parameters to solver class parameters:
    - max_nb_iters -> max_inner_iter
    - max_nb_loops -> max_outer_loop
    - tol_constraints -> tol_constraint
    """
    solver = config.numerics["solver"]
    return {
        "max_inner_iter": solver["max_nb_iters"],
        "max_outer_loop": solver["max_nb_loops"],
        "tol_convergence": solver["tol_convergence"],
        "tol_constraint": solver["tol_constraints"],
        "init_penalty_weight": solver["init_penalty_weight"],
    }


def build_trajectory(config: Config) -> np.ndarray:
    """Build separation trajectory from configuration."""
    traj_cfg = config.simulation["trajectory"]
    traj_type = traj_cfg["type"]

    if traj_type == "approach_retract":
        d_min = traj_cfg["min_separation"]
        d_max = traj_cfg["max_separation"]
        d_step = traj_cfg["step_size"]
        round_trip = traj_cfg.get("round_trip", True)

        nb_steps = round((d_max - d_min) / d_step) + 1
        # Start from max (approach), go to min
        separations = np.linspace(d_max, d_min, nb_steps)

        if round_trip:
            separations = np.concatenate([separations, np.flip(separations)[1:]])

        return separations

    elif traj_type == "explicit":
        return np.array(traj_cfg["values"])

    else:
        raise ValueError(f"Unknown trajectory type: {traj_type}")


def compute_liquid_volume(
    grid: Grid,
    constraint_cfg: dict[str, Any],
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
    V_percent = 0.01 * constraint_cfg["liquid_volume_percent"]
    return formulation.get_volume() * V_percent
