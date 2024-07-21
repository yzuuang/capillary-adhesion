"""
Tests of the `modelling.py` file.
"""

import math
import sys

import numpy as np
import matplotlib.pyplot as plt

from a_package.modelling import Region, CapillaryBridge


def test_capillary_bridge_compute_energy_jacobian():
    L = 10.0
    N = 100
    region = Region(L, L, N, N)

    eta = L / N

    xm, ym = np.meshgrid(region.x, region.y)
    # a ball on top and a flat plate on base
    h1 = L * np.sqrt(1 - (xm/L - 0.5)**2 - (ym/L - 0.5)**2)
    h2 = np.zeros_like(h1)

    # a circular phase field
    phi = np.ones_like(h1)
    phi[(xm/L)**2 + (ym/L)**2 >= 0.5] = 0.0

    capi = CapillaryBridge(region, eta, h1, h2, phi)
    capi.update_displacement(eta * 2)

    # all the step lengths to be used for finite difference computation
    lowest_magnitude = math.floor(0.5 * math.log10(sys.float_info.epsilon))  # machine precision determined
    highest_magnitude = 1
    deltas = np.pow(10.0, np.arange(lowest_magnitude, highest_magnitude))

    # compute jacobian numerically (2-order finite difference)
    numeric_jacobian = np.empty((deltas.size, phi.size))
    for i, delta in enumerate(deltas):
        for j1, j2 in np.ndindex(phi.shape):
            phi[j1, j2] += delta
            capi.phi = phi
            plus_val = capi.compute_energy()
            phi[j1, j2] -= 2*delta
            capi.phi = phi
            minus_val = capi.compute_energy()
            numeric_jacobian[i, j1*region.ny + j2] = (plus_val - minus_val) / delta * 0.5

            phi[j1, j2] += delta

    # compute jacobian from the implementation
    capi.phi = phi
    impl_jacobian = capi.compute_energy_jacobian()

    # measure the difference and plot
    diffs = np.amax(abs(impl_jacobian[np.newaxis, :] - numeric_jacobian), axis=1)
    plt.plot(deltas, diffs , "x-", label=r"Difference from a numerical method of $\mathcal{O}(\delta^2)$")

    # format plot
    plt.loglog()
    plt.xlabel(r"$\delta$")
    plt.ylabel(r"$\varepsilon$")
    plt.legend()

    plt.show()

    # assertion
    eps = 1e-6
    assert min(diffs) < eps, f"The difference exceeds the tolerance {eps:.2e}"
