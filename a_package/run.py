"""
Black-box simulation execution.

Provides run_simulation() and run_sweep() - config in, results out.
"""

import logging

import numpy as np
import numpy.random as random

from a_package.config import Config, save_config, expand_sweeps, count_sweep_combinations
from a_package.simulation.setup import (
    create_grid_from_config,
    generate_surface_from_config,
    build_capillary_args,
    build_solver_args,
    build_trajectory,
    compute_liquid_volume,
)
from a_package.simulation.simulation import Simulation
from a_package.simulation.io import SimulationIO
from a_package.runtime.dirs import RunDir, register_run
from a_package.runtime.logging import switch_log_file


logger = logging.getLogger(__name__)


def run_simulation(config: Config, run_dir: RunDir) -> SimulationIO:
    """
    Run a single simulation from config.

    This is the black-box interface: config in, results out.
    Dispatches to appropriate Simulation method based on constraint type.

    Parameters
    ----------
    config : Config
        Complete simulation configuration.
    run_dir : RunDir
        Run directory object with results_dir, parameters_dir, etc.

    Returns
    -------
    SimulationIO
        IO object for accessing saved results.
    """
    # Build primitives from config
    grid = create_grid_from_config(config)
    upper = generate_surface_from_config(grid, config.physics["upper"])
    lower = generate_surface_from_config(grid, config.physics["lower"])
    capillary_args = build_capillary_args(config)
    solver_args = build_solver_args(config)
    trajectory = build_trajectory(config)

    # Random initial phase field
    rng = random.default_rng()
    phase_init = rng.random((1, 1, *grid.nb_elements))

    # Create simulation object
    logger.info(f"Starting simulation with output to {run_dir.results_dir}")
    simulation = Simulation(grid, capillary_args, solver_args)

    # Dispatch based on constraint type
    constraint_cfg = config.simulation["constraint"]
    constraint_type = constraint_cfg["type"]

    if constraint_type == "constant_volume":
        volume = compute_liquid_volume(
            grid, constraint_cfg, upper, lower, capillary_args, trajectory
        )
        return simulation.run_with_constant_volume(
            upper, lower, trajectory, volume, run_dir.results_dir, phase_init=phase_init
        )
    else:
        raise ValueError(f"Unknown constraint type: {constraint_type}")


def run_sweep(config: Config, run_dir: RunDir) -> list[SimulationIO]:
    """
    Run simulation(s) from config, handling sweeps if present.

    If config has no sweeps, runs single simulation.
    If config has sweeps, expands and runs each in a subdirectory.

    Parameters
    ----------
    config : Config
        Configuration, possibly with sweep definitions.
    run_dir : RunDir
        Base run directory. Sub-runs will be created in intermediate_dir.

    Returns
    -------
    list[SimulationIO]
        List of IO objects, one per run.
    """
    nb_configs = count_sweep_combinations(config)

    if nb_configs == 1:
        # No sweeps, single run
        switch_log_file(run_dir.log_file)
        return [run_simulation(config, run_dir)]

    # Parameter sweep
    results = []
    for index, expanded_config in enumerate(expand_sweeps(config)):
        # Create sub-run directory
        sub_run = register_run(run_dir.intermediate_dir, __file__, with_hash=False)
        switch_log_file(sub_run.log_file)

        logger.info(f"Run #{index + 1} of {nb_configs} runs.")

        # Save expanded config for reproducibility
        save_config(expanded_config, sub_run.parameters_dir / f"config.toml")

        io = run_simulation(expanded_config, sub_run)
        results.append(io)

    return results
