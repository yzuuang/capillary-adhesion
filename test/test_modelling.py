import numpy as np
import numpy.random as random
from a_package.modelling import CapillaryVapourLiquid

from .utilities import *


rng = random.default_rng()


def test_energy_density_sensitivity():
    # A not too big random field
    test_size = 5
    vapour_liquid = CapillaryVapourLiquid(1.0, 0.5, rng.normal(0.0, size=(1, test_size)))
    phi = rng.random(size=(1, test_size))
    phi_grad = rng.random(size=(2, test_size))

    # Implemented
    [e_D_phi, e_D_phi_grad] = vapour_liquid.energy_density_sensitivity(phi, phi_grad)

    # Finite difference
    def call_with_one_arg(arg):
        phi = arg[0]
        phi_grad = arg[-2:]
        return vapour_liquid.energy_density(phi, phi_grad)
    arg = np.concatenate([phi, phi_grad], axis=0)
    delta = 1e-4
    ref = central_difference_jacobian(call_with_one_arg, arg, delta)
    ref_e_D_phi = ref[0]
    ref_e_D_phi_grad = ref[-2:]

    # Because in "modellling", all functions only modify the "components" axis, and don't touch
    # the "grid shape", every data point is independent.
    ref_e_D_phi = ref_e_D_phi[np.nonzero(ref_e_D_phi)].reshape(e_D_phi.shape)
    ref_e_D_phi_grad = ref_e_D_phi_grad[np.nonzero(ref_e_D_phi_grad)].reshape(e_D_phi_grad.shape)

    # Assertions
    assert np.allclose(e_D_phi, ref_e_D_phi)
    assert np.allclose(e_D_phi_grad, ref_e_D_phi_grad)



def test_liquid_height_sensitivity():
    # A not too big random field
    test_size = 5
    vapour_liquid = CapillaryVapourLiquid(1.0, 0.5, rng.normal(0.0, size=(1, test_size)))
    phase = rng.random(size=(1, test_size))

    # Implemented
    [h_D_phi] = vapour_liquid.liquid_height_sensitivity(phase)

    # Finite difference
    delta = 1e-4
    ref_h_D_phi = np.squeeze(central_difference_jacobian(vapour_liquid.liquid_height, phase, delta))

    # Because in "modellling", all functions only modify the "components" axis, and don't touch
    # the "grid shape", every data point is independent.
    ref_h_D_phi = ref_h_D_phi[np.nonzero(ref_h_D_phi)].reshape(h_D_phi.shape)


    # Assertion
    assert np.allclose(h_D_phi, ref_h_D_phi)
