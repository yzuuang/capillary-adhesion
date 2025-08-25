import os
import sys

import numpy as np
import numpy.random as random

from a_package.modelling import Region, CapillaryBridge
from a_package.solving import AugmentedLagrangian
from a_package.storing import working_directory
from a_package.routine import simulate_quasi_static_pull_push, post_process

from utils.logging import reset_logging, switch_log_file
from utils.runtime import register_run
from utils.overview import create_overview_animation

from cases.configs import read_config_files, get_region_specs, match_shape_and_get_height, preview_surface_and_gap


show_me = False


def main():
    reset_logging()
    config_files = sys.argv[1:]

    # setup run directory
    case_name = os.path.basename(os.path.dirname(__file__))
    run = register_run(case_name, __file__, *config_files)

    config = read_config_files(config_files)

    # visual check
    if show_me:
        preview_surface_and_gap(config)

    switch_log_file(run.log_file)
    run_one_trip(run, config)


def run_one_trip(run, config: dict[str, dict[str, str]]):
    # grid
    region = get_region_specs(config["Grid"])

    # surfaces
    upper = match_shape_and_get_height(region, config["UpperSurface"])
    lower = match_shape_and_get_height(region, config["LowerSurface"])

    # trajectory
    d_min = float(config["Trajectory"]["min_separation"])
    d_max = float(config["Trajectory"]["max_separation"])
    d_step = float(config["Trajectory"]["step_size"])
    nb_steps = round((d_max - d_min) / d_step) + 1
    trajectory = np.linspace(d_max, d_min, nb_steps)

    # capillary model
    eta = float(config["Capillary"]["interface_thickness"])
    theta = (np.pi / 180) * float(config["Capillary"]["contact_angle_degree"])
    capi = CapillaryBridge(region, eta, theta, upper, lower)

    # solver
    i_max = int(config["Solver"]["max_nb_iters"])
    l_max = int(config["Solver"]["max_nb_loops"])
    tol_conver = float(config["Solver"]["tol_convergence"])
    tol_constr = float(config["Solver"]["tol_constraints"])
    c_init = float(config["Solver"]["init_penalty_weight"])
    solver = AugmentedLagrangian(i_max, l_max, tol_conver, tol_constr, c_init)

    # liquid volume from a percentage specification
    V_percent = 0.01 * float(config["Capillary"]["liquid_volume_percent"])
    capi.z1 = np.amin(trajectory)
    capi.update_gap()
    capi.phi = np.ones((region.nx, region.ny))
    capi.update_phase_field()
    V = capi.volume * V_percent

    # run simulation routine
    with working_directory(run.intermediate_dir, read_only=False) as store:
        # start sim with random initial guess
        capi.phi = random.rand(region.nx, region.ny)
        capi.update_phase_field()
        sim = simulate_quasi_static_pull_push(store, capi, solver, V, trajectory)

    # post-process
    with working_directory(run.results_dir, read_only=False) as store:
        p_sim = post_process(sim)
        store.save("result", p_sim)

    # visualise
    create_overview_animation(run.path)


if __name__ == "__main__":
    main()
