import numpy as np
import numpy.random as random

from a_package.modelling import CapillaryVapourLiquid


rng = random.default_rng()


def test_sensitivity():
    vapour_liquid = CapillaryVapourLiquid(1.0, 0.5, rng.normal(0.0, size=1))
    phi = rng.random(size=1)
    d_phi = rng.random(size=2)

    [sens_phi, sens_d_phi] = vapour_liquid.energy_density_sensitivity(phi, d_phi)

    def call_with_flat_args(phi, d1_phi, d2_phi):
        return vapour_liquid.energy_density(np.array([phi]), np.array([d1_phi, d2_phi]))

    ref_sensitivity = finite_difference_jacobian(call_with_flat_args, *phi, *d_phi)

    assert np.isclose(sens_phi, ref_sensitivity[0])
    assert np.isclose(sens_d_phi[0], ref_sensitivity[1])
    assert np.isclose(sens_d_phi[1], ref_sensitivity[2])


def finite_difference_jacobian(func, *args):
    delta = 2**(-23)

    args = np.array(args)
    jacobian = np.full_like(args, np.nan)

    for index, arg in enumerate(args):
        # Backup the original value
        original_value = np.copy(arg)

        # Get finite difference jacobian
        args[index] = original_value + delta
        value_plus = func(*args).item()
        args[index] = original_value - delta
        value_minus = func(*args).item()
        jacobian[index] = (value_plus - value_minus) / (2 * delta)

        # reset back to origin
        args[index] = original_value

    return jacobian
