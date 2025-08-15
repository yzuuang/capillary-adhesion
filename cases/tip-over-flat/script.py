import os
import sys

import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

from a_package.modelling import Region, CapillaryBridge
from a_package.solving import AugmentedLagrangian
from a_package.storing import working_directory
from a_package.routine import simulate_quasi_static_pull_push, post_process

from utils.common import init_logging, read_configs
from utils.runtime import register_run
from utils.overview import create_overview_animation


init_logging()
show_me = False


def main():
    # setup folder for running
    case_name = os.path.basename(os.path.dirname(__file__))
    params_files = sys.argv[1:]
    run = register_run(case_name, __file__, *params_files)

    # read parameters
    config = read_configs(params_files)

    # grid
    a = config["Grid"].getfloat("pixel_size")
    N = config["Grid"].getint("nb_pixels")
    L = a * N
    region = Region(a, L, L, N, N)

    # height profile of a spherical tip
    R = config["Surface"].getfloat("tip_radius")
    h1 = -np.sqrt(np.clip(R**2 - (region.xm - 0.5 * region.lx) ** 2 - (region.ym - 0.5 * region.ly) ** 2, 0, None))
    # set lowest point to zero
    h1 += np.amax(abs(h1))

    # height profile of flat
    h2 = np.zeros((N, N))

    # specify the trajectory
    d_min = config["Trajectory"].getfloat("min_separation")
    d_max = config["Trajectory"].getfloat("max_separation")
    d_step = config["Trajectory"].getfloat("step_size")

    # create the capillary model
    eta = config["Capillary"].getfloat("interface_thickness")
    theta = config["Capillary"].getfloat("contact_angle_degree")
    gamma = -np.cos(theta / 180.0 * np.pi)
    capi = CapillaryBridge(region, eta, gamma, h1, h2)

    # specify liquid volume by a percentage
    capi.z1 = d_min
    capi.update_gap()
    capi.phi = np.ones((N, N))
    capi.update_phase_field()
    V_percent = 0.01 * config["Capillary"].getfloat("liquid_volume_percent")
    V = capi.volume * V_percent

    # solving parameters
    i_max = config["Solver"].getint("max_nb_iters")
    l_max = config["Solver"].getint("max_nb_loops")
    tol_conver = config["Solver"].getfloat("tol_convergence")
    tol_constr = config["Solver"].getfloat("tol_constraints")
    c_init = config["Solver"].getfloat("init_penalty_weight")
    solver = AugmentedLagrangian(i_max, l_max, tol_conver, tol_constr, c_init)

    # Visual check before running
    if show_me:
        fig, ax = plt.subplots()
        # image = ax.pcolormesh(region.xm/a, region.ym/a, h1/a, cmap='hot')
        image = ax.imshow(h1 / a, interpolation="bicubic", cmap="plasma", extent=[0, N, 0, N])
        fig.colorbar(image)

        fig, ax = plt.subplots()
        # image = ax.pcolormesh(region.xm/a, region.ym/a, gap/a, cmap='hot')
        image = ax.imshow(capi.g / a, vmin=0, interpolation="bicubic", cmap="hot", extent=[0, N, 0, N])
        fig.colorbar(image)

        plt.show()
        skip = input("Run simulation [Y/n]? ")[:1].upper() == "N"
        if skip:
            sys.exit(0)

    # run simulation routine
    with working_directory(run.intermediate_dir, read_only=False) as store:
        # start sim with random initial guess
        capi.phi = random.rand(N, N)
        capi.update_phase_field()
        sim = simulate_quasi_static_pull_push(store, capi, solver, V, d_min, d_max, d_step)

    # post-process
    with working_directory(run.results_dir, read_only=False) as store:
        p_sim = post_process(sim)
        store.save("result", p_sim)

    # visualise
    create_overview_animation(case_name, run.run_id)


if __name__ == "__main__":
    main()
