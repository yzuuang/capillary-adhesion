import numpy as np
import numpy.random as random

from a_package.data_record import DropletData, save_record
from a_package.routine import sim_quasi_static_pull_push



if __name__ == '__main__':
    # primary parameters
    L = 10.0    # lateral size
    V = 10.0    # volume of the droplet
    eta = 0.05  # interface width
    M = 200     # num of pixels along x-axis
    N = 200     # num of pixels along y-axis

    # derived from primary ones
    dx = L / M
    dy = L / N
    x = np.arange(M) * dx
    y = np.arange(N) * dy

    # height profile of flat surfaces
    h1 = np.zeros((M, N))
    h2 = np.zeros((M, N))

    # random initial guess
    phi_init = random.rand(M, N)

    # data holder
    data = DropletData(
        V, eta, L, M, N, phi_init, h1, h2, 0.0, x, y, dx, dy
    )

    # simulating routine
    d_min = 2 * data.eta
    d_max = 10 * data.eta
    d_step = 0.02
    rec = sim_quasi_static_pull_push(data, phi_init, d_min, d_max, d_step)

    # save
    filename = f"{__file__}.data"
    save_record(rec, filename)

