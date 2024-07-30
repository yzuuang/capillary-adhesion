import numpy as np

from a_package.modelling import Region, wavevector_norm, SelfAffineRoughness, PSD_to_height, CapillaryBridge
from a_package.solving import AugmentedLagrangian
from a_package.storing import working_directory
from a_package.routine import simulate_quasi_static_slide


if __name__ == "__main__":
    # modelling parameters
    eta = 0.05            # interface width
    N = 200               # num of nodes along one axis
    L = N * eta           # lateral size

    # the region where simulation runs
    region = Region(L, L, N, N)

    # generate "one-peak" plates
    C0 = 1e6  # prefactor+
    qR = 2*np.pi / L  # roll-off
    qS = 2e1  # cut-off
    H = 0.95  # Hurst exponent
    roughness = SelfAffineRoughness(C0, qR, qS, H)
    q_2D = wavevector_norm(region.qx, region.qy)
    C_2D = roughness.mapto_psd(q_2D)

    # the base body is the negation of the top
    h1 = PSD_to_height(C_2D)
    h2 = np.negative(h1)

    # start with full field
    phi = np.ones((N, N))

    # combine into the model object
    capi = CapillaryBridge(region, eta, h1, h2, phi)

    # set the volume such that it is almost full at the start of the simulation
    z1 = abs(capi.h1.min()) + capi.h2.max() - eta
    capi.update_displacement(z1=z1)
    V = capi.compute_volume()

    # solving parameters
    k_max = 2000
    e_conv = 1e-8
    e_volume = 1e-6
    c_init = 1e-2
    c_upper = 1e3
    beta = 3.0
    solver = AugmentedLagrangian(k_max, e_conv, e_volume, c_init, c_upper, beta)

    # run simulation routine
    m_track = [(0, i) for i in range(10)]
    path = __file__.replace(".py", ".data")
    with working_directory(path, read_only=False) as store:
        # store.brand_new()
        simulate_quasi_static_slide(store, capi, solver, V, m_track)
