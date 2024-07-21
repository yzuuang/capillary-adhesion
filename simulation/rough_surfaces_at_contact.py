import numpy.random as random

from a_package.modelling import Region, wavevector_norm, SelfAffineRoughness, PSD_to_height, CapillaryBridge
from a_package.routine import simulate_quasi_static_pull_push
from a_package.solving import AugmentedLagrangian
from a_package.storing import working_directory


if __name__ == "__main__":
    # modelling parameters
    eta = 0.05            # interface width
    N = 200              # num of nodes along one axis
    L = N * eta           # lateral size
    V = L**2 * (2 * eta)  # volume of the droplet

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
    e_conv = 1e-8
    e_volume = 1e-6
    c_init = 1e-2
    c_upper = 1e3
    beta = 3.0
    solver = AugmentedLagrangian(k_max, e_conv, e_volume, c_init, c_upper, beta)

    # run simulation routine
    d_min = 4 * eta
    d_max = 5 * eta
    d_step = 0.2 * eta
    path = __file__.replace(".py", ".data")
    with working_directory(path, read_only=False) as store:
        store.brand_new()
        simulate_quasi_static_pull_push(store, capi, solver, V, d_min, d_max, d_step)
