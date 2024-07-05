import numpy as np
import numpy.random as random

import matplotlib.pyplot as plt

from a_package.data_record import DropletData, save_record
from a_package.roughness import generate_isotropic_psd, interpolate_isotropic_psd_in_2d, convert_psd_to_surface
from a_package.routine import sim_quasi_static_pull_push


if __name__ == "__main__":
    # primary parameters
    L = 10.0    # lateral size
    V = 10.0    # volume of the droplet
    eta = 0.05  # interface width
    M = 1000    # num of pixels along x-axis
    N = 1000    # num of pixels along y-axis

    # derived from primary ones
    dx = L / M
    dy = L / N
    x = np.arange(M) * dx
    y = np.arange(N) * dy

    # random initial guess
    phi_init = random.rand(M, N)

    # generate roughness PSD
    C0 = 1e7
    qR = 2e0  # roll-off
    qS = 2e1  # cut-off
    H = 0.95  # Hurst exponent
    q_mapto_C = generate_isotropic_psd(C0, qR, qS, H)
    qx, qy, C_2d = interpolate_isotropic_psd_in_2d(M, dx, N, dy, q_mapto_C)

    # from PSD to random surface roughness
    h1 = convert_psd_to_surface(C_2d)
    h2 = convert_psd_to_surface(C_2d)

    # data holder
    data = DropletData(
        V, eta, L, M, N, phi_init, h1, h2, 0.0, x, y, dx, dy
    )

    # simulating routine
    d_min = 3 * data.eta
    d_max = 9 * data.eta
    d_step = 0.2 * data.eta
    rec = sim_quasi_static_pull_push(data, phi_init, d_min, d_max, d_step)

    # save
    filename = f"{__file__}.data"
    save_record(rec, filename)
