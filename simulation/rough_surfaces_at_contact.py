import numpy.random as random

from a_package.modelling import Region, wavevector_norm, SelfAffineRoughness, PSD_to_height, CapillaryBridge
from a_package.solving import AugmentedLagrangian
from a_package.routine import sim_quasi_static_pull_push


if __name__ == "__main__":
    # modelling parameters
    eta = 0.05  # interface width
    L = 1e3 * eta     # lateral size
    V = 2e6 * eta**3  # volume of the droplet
    N = 200  # num of nodes along one axis

    # the region where simulation runs
    region = Region(L, L, N, N)

    # generate rough plates
    C0 = 1e8  # prefactor
    qR = 2e0  # roll-off
    qS = 2e1  # cut-off
    H = 0.95  # Hurst exponent
    roughness = SelfAffineRoughness(C0, qR, qS, H)
    q_2D = wavevector_norm(region.qx, region.qy)
    C_2D = roughness.mapto_psd(q_2D)
    h1 = PSD_to_height(C_2D)
    h2 = PSD_to_height(C_2D)

    # random phase field to start
    rng = random.default_rng()
    phi = rng.random((N, N))

    # combine into the model object
    capi = CapillaryBridge(region, eta, h1, h2, phi)

    # solving parameters
    k_max = 2000
    e_conv = 1e-6
    e_volume = 1e-4
    c_init = 1e-3
    c_upper = 1e3
    beta = 3.0

    solver = AugmentedLagrangian(k_max, e_conv, e_volume, c_init, c_upper, beta)

    # run simulation routine
    d_min = 3 * eta
    d_max = 9 * eta
    d_step = 0.2 * eta
    sim_quasi_static_pull_push(capi, solver, V, d_min, d_max, d_step)
