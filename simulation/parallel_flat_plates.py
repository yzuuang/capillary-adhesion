import numpy as np
import numpy.random as random

from a_package.modelling import Region, CapillaryBridge
from a_package.solving import AugmentedLagrangian
from a_package.storing import working_directory
from a_package.routine import simulate_quasi_static_pull_push


if __name__ == '__main__':
    # primary parameters
    eta = 0.1     # interface width
    gamma = -1e-2  # surface tensions: (SL - SG) / LG
    N = 256  # num of pixels along one axis
    L = N * eta  # lateral size
    V = L**2 * eta  # volume of the droplet
    region = Region(L, L, N, N)

    # height profile of flat surfaces
    h1 = np.zeros((N, N))
    h2 = np.zeros((N, N))

    # random initial guess
    phi = random.rand(N, N)

    # combine into the model object
    capi = CapillaryBridge(region, eta, gamma, h1, h2, z1=2*eta, phi=phi)
    capi.update_gap()
    capi.update_phase_field()

    # solving parameters
    k_max = 2000
    e_conv = 1e-8
    e_volume = 1e-6
    c_init = 1e-1
    c_upper = 1e4
    beta = 3.0
    solver = AugmentedLagrangian(k_max, e_conv, e_volume, c_init, c_upper, beta)

    # run simulation routine
    # run simulation routine
    d_min = 2 * eta
    d_max = 3 * eta
    d_step = 0.5 * eta
    path = __file__.replace(".py", ".data")
    with working_directory(path, read_only=False) as store:
        store.brand_new()
        simulate_quasi_static_pull_push(store, capi, solver, V, d_min, d_max, d_step)
