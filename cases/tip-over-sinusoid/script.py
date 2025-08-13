import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from a_package.modelling import Region, CapillaryBridge
from a_package.solving import AugmentedLagrangian
from a_package.storing import working_directory
from a_package.routine import simulate_quasi_static_pull_push
from a_package.visualizing import *


show_me = False

# define the working path by file name
from utils.common import get_runtime_dir, read_configs

case_name = os.path.basename(os.path.dirname(__file__))
working_dir = get_runtime_dir(case_name)


def main():
    config = read_configs(sys.argv[1:])

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

    # height profile of sinusoid
    wave_num = config['Surface'].getint('wavenumber')
    wave_amp = config['Surface'].getfloat('amplitude')
    xm = region.xm
    qx = (2*np.pi / L) * wave_num
    ym = region.ym
    qy = (2*np.pi / L) * wave_num
    h2 = wave_amp * np.cos(qx * xm) * np.cos(qy * ym)

    # specify the trajectory
    d_min = wave_amp + config["Trajectory"].getfloat("min_separation")
    d_max = wave_amp + config["Trajectory"].getfloat("max_separation")
    d_step = config["Trajectory"].getfloat("step_size")

    # the capillary model object
    eta = config["Capillary"].getfloat("interface_thickness")
    theta = config["Capillary"].getfloat("contact_angle_degree")
    gamma = -np.cos(theta / 180.0 * np.pi)
    capi = CapillaryBridge(region, eta, gamma, h1, h2)

    # set liquid volume
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
        image = ax.imshow(capi.g / a, interpolation="bicubic", cmap="hot", extent=[0, N, 0, N])
        fig.colorbar(image)

        plt.show()

    # run the sim
    with working_directory(working_dir, read_only=False) as store:
        store.brand_new()
        # random initial guess
        capi.phi = np.random.rand(N, N)
        capi.update_phase_field()
        simulate_quasi_static_pull_push(store, capi, solver, V, d_min, d_max, d_step)


if __name__ == "__main__":
    main()
