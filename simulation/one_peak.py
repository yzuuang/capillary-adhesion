import numpy as np
import matplotlib.pyplot as plt

from a_package.modelling import Region, wavevector_norm, SelfAffineRoughness, PSD_to_height, CapillaryBridge
from a_package.solving import AugmentedLagrangian
from a_package.storing import working_directory
from a_package.routine import simulate_quasi_static_slide
from a_package.visualizing import *


# For the reproducibility of surface heights
seed = 0


if __name__ == "__main__":
    # modelling parameters
    eta = 0.05    # interface width
    N = 256       # num of nodes along one axis
    L = N * eta   # lateral size

    # the region where simulation runs
    region = Region(L, L, N, N)

    # generate "one-peak" plates
    C0 = 1e6  # prefactor+
    qR = 2*np.pi / L  # roll-off
    qS = qR * 1.02    # cut-off
    H = 0.95  # Hurst exponent
    roughness = SelfAffineRoughness(C0, qR, qS, H)
    q_2D = wavevector_norm(region.qx, region.qy)
    C_2D = roughness.mapto_psd(q_2D)

    # the base body is the negation of the top
    h1 = PSD_to_height(C_2D, seed)
    h2 = np.negative(h1)  # Base height is the inverse of the top
    # h2 = np.zeros_like(h1)  # Flat base

    # combine into the model object
    capi = CapillaryBridge(region, eta, h1, h2)

    # Set the gap such that only the peaks are in contact
    capi.z1 = abs(capi.h1.min()) + capi.h2.max() - eta
    capi.update_gap()

    # set the volume such that it is almost full at the start of the simulation
    capi.phi = np.ones((region.nx, region.ny))
    capi.update_phase_field()
    V = capi.volume * 0.85

    # Check the gap profile
    data = DropletData(region, eta, h1, h2, capi.displacement, capi.g, capi.phi, capi.force)
    data.g = capi.g
    fig, ax = plt.subplots()
    m = plot_gap_topography(ax, data)
    plot_contact_topography(ax, data)
    fig.colorbar(m)
    plt.show()

    skip = input("Run simulation [Y/n]? ").upper() == "N"
    if skip:
        quit()

    # solving parameters
    k_max = 2000
    e_conv = 1e-8
    e_volume = 1e-6
    c_init = 1e-1
    c_upper = 1e3
    beta = 3.0
    solver = AugmentedLagrangian(k_max, e_conv, e_volume, c_init, c_upper, beta)

    # run simulation routine
    m_track = [(0, i) for i in range(3)]
    path = __file__.replace(".py", ".data")
    with working_directory(path, read_only=False) as store:
        # store.brand_new()
        simulate_quasi_static_slide(store, capi, solver, V, m_track)
