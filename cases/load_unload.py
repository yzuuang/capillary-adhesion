import os
import sys
import logging

import numpy as np
import numpy.random as random

from a_package.storing import working_directory
from a_package.formulating import Formulation
from a_package.simulating import simulate_quasi_static_pull_push
from a_package.postprocessing import post_process
from a_package.storing import working_directory

from utils.logging import reset_logging, switch_log_file
from utils.runtime import register_run

from cases.configs import (
    read_config_files,
    preview_surface_and_gap,
    save_config_to_file,
    extract_sweeps,
    get_grid_specs,
    match_shape_and_get_height,
    get_capillary,
    get_optimizer,
)
from cases.visualise_onerun import create_overview_animation


show_me = False
logger = logging.getLogger(__name__)


def main():
    reset_logging()
    config_files = sys.argv[1:]
    config = read_config_files(config_files)

    # visual check
    if show_me:
        preview_surface_and_gap(config["Grid"], config["UpperSurface"], config["LowerSurface"], config["Trajectory"])

    # setup run directory
    case_name = os.path.splitext(os.path.basename(__file__))[0]
    shape_name = f'{config["UpperSurface"]["shape"]}-over-{config["LowerSurface"]["shape"]}'
    base_dir = os.path.join(case_name, shape_name)
    run = register_run(base_dir, __file__, *config_files)

    # check if parameter sweep is specified in config
    sweep_section_prefix = "ParameterSweep"
    sweeps = extract_sweeps(config, sweep_section_prefix)
    if sweeps is None:
        switch_log_file(run.log_file)
        run_one_trip(run, config)
        create_overview_animation(run.path)
    else:
        nb_subruns = len(sweeps)
        for index, config in enumerate(sweeps.iter_config(config)):
            sub_run = register_run(run.intermediate_dir, __file__, with_hash=False)
            switch_log_file(sub_run.log_file)
            logger.info(f"Run #{index} of {nb_subruns} runs.")
            save_config_to_file(config, sub_run.parameters_dir / f"subrun-{index}.ini")
            run_one_trip(sub_run, config)
            create_overview_animation(sub_run.path)


def run_one_trip(run, config: dict[str, dict[str, str]]):

    # grid
    grid = get_grid_specs(config["Grid"])

    # surfaces
    upper = match_shape_and_get_height(grid, config["UpperSurface"])
    lower = match_shape_and_get_height(grid, config["LowerSurface"])

    # trajectory
    d_min = float(config["Trajectory"]["min_separation"])
    d_max = float(config["Trajectory"]["max_separation"])
    d_step = float(config["Trajectory"]["step_size"])
    nb_steps = round((d_max - d_min) / d_step) + 1
    trajectory = np.linspace(d_max, d_min, nb_steps)

    # capillary model
    capi = get_capillary(config["Capillary"])

    # solver
    solver = get_optimizer(config["Solver"])

    # liquid volume from a percentage specification
    formulation = Formulation(grid, upper, lower, capi)
    z1 = np.amin(trajectory)
    formulation.update_gap(z1)
    full_liquid = np.ones((grid.nx, grid.ny))
    formulation.update_phase_field(full_liquid)
    V_percent = 0.01 * float(config["Capillary"]["liquid_volume_percent"])
    V = formulation.get_volume() * V_percent

    # run simulation
    with working_directory(run.results_dir, read_only=False) as store:
        # start sim with random initial guess
        phi_init = random.rand(grid.nx, grid.ny)
        sim = simulate_quasi_static_pull_push(store, formulation, solver, V, phi_init, trajectory)

    # post-process
    with working_directory(run.results_dir, read_only=False) as store:
        p_sim = post_process(sim)
        store.save("final", p_sim)


if __name__ == "__main__":
    main()
