import os
import sys

import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

from a_package.modelling import Region, wavevector_norm, SelfAffineRoughness, PSD_to_height, CapillaryBridge
from a_package.routine import simulate_quasi_static_pull_push
from a_package.solving import AugmentedLagrangian
from a_package.storing import working_directory
from a_package.visualizing import *

from utils.common import get_runtime_dir, read_configs


show_me = False

# define the working path by file name
case_name = os.path.basename(os.path.dirname(__file__))
working_dir = get_runtime_dir(case_name)

# get a random seed
seed = random.SeedSequence().entropy
print(f"seed = {seed}")
rng = random.default_rng(seed)


def main():
    config = read_configs(sys.argv[1:])

    # grid
    a = config["Grid"].getfloat("pixel_size")
    N = config["Grid"].getint("nb_pixels")
    L = a * N
    region = Region(a, L, L, N, N)

    # generate roughness PSD
    C0 = config['Roughness'].getfloat("prefactor")
    nR = config['Roughness'].getint("wavenumber_rolloff")
    qR = (2*np.pi/L) * nR  # roll-off wave vector
    nS = config['Roughness'].getint("wavenumber_cutoff")
    qS = (2*np.pi/L) * nS  # cut-off
    H = config['Roughness'].getfloat("hurst_exponent")
    roughness = SelfAffineRoughness(C0, qR, qS, H)
    q_2D = wavevector_norm(region.qx, region.qy)
    C_2D = roughness.mapto_psd(q_2D)

    # generate rough surface from PSD
    h1 = PSD_to_height(C_2D, rng=rng)
    h2 = PSD_to_height(C_2D, rng=rng)

    # specify the trajectory
    d_min = config["Trajectory"].getfloat("min_separation")
    d_max = config["Trajectory"].getfloat("max_separation")
    d_step = config["Trajectory"].getfloat("step_size")

    # the capillary model object
    eta = config["Capillary"].getfloat("interface_thickness")
    theta = config["Capillary"].getfloat("contact_angle_degree")
    gamma = -np.cos(theta / 180.0 * np.pi)
    capi = CapillaryBridge(region, eta, gamma, h1, h2)

    capi.update_gap()

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

    # visual check before running
    if show_me:
        fig, ax = plt.subplots()
        # image = ax.pcolormesh(region.xm/a, region.ym/a, h1/a, cmap='hot')
        image = ax.imshow(h1 / a, interpolation="bicubic", cmap="plasma", extent=[0, N, 0, N])
        fig.colorbar(image)

        fig, ax = plt.subplots()
        # image = ax.pcolormesh(region.xm/a, region.ym/a, gap/a, cmap='hot')
        image = ax.imshow(capi.g / a, vmin=0, interpolation="bicubic", vmin=0, cmap="hot", extent=[0, N, 0, N])
        fig.colorbar(image)

        plt.show()
        skip = input("Run simulation [Y/n]? ")[:1].upper() == "N"
        if skip:
            sys.exit(0)

    # run the sim
    with working_directory(working_dir, read_only=False) as store:
        # clean the working dir
        store.brand_new()

        # start sim with a random initial guess
        capi.phi = rng.random([N, N])
        capi.update_phase_field()
        simulate_quasi_static_pull_push(store, capi, solver, V, d_min, d_max, d_step)


if __name__ == "__main__":
    main()
