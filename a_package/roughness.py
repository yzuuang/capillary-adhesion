from typing import Callable

import numpy as np
import numpy.fft as fft
from numpy.random import default_rng


rng = default_rng()


def generate_isotropic_psd(C0: float, qR: float, qS: float, H: float):
    """

    :param C0: prefactor
    :param qR: roll-off frequency
    :param qS: cut-off frequency
    :param H: Hurst exponent
    :return: mapping from q to C
    """

    def q_mapto_C(q: np.ndarray):
        """generate PSD given an array of wavevector norm (frequency)

        :param q: norm of wavevector, can be 1D or 2D
        :return: PSD
        """
        C = np.empty_like(q)
        C[q < qR] = C0 * qR**(-2-2*H)
        C[(q >= qR) & (q < qS)] = C0 * q[(q >= qR) & (q < qS)]**(-2-2*H)
        C[q >= qS] = 0
        return C

    return q_mapto_C


def interpolate_isotropic_psd_in_2d(Nx: int, dx: float, Ny: int, dy: float, PSD_mapping: Callable[[np.ndarray], np.ndarray]):
    """Convert an isotropic PSD (1D function) into plain PSD (2D function)"""
    qx = (2*np.pi) * fft.fftfreq(Nx, dx)
    qy = (2*np.pi) * fft.fftfreq(Ny, dy)
    qxx, qyy = np.meshgrid(qx, qy)
    q_2d = (qxx**2 + qyy**2)**(1/2)
    C_2d = PSD_mapping(q_2d)
    return qx, qy, C_2d


# from PSD to surface
def convert_psd_to_surface(C_2d: np.ndarray):
    h_q = np.sqrt(C_2d)

    # impose some random phase angel
    phi = rng.uniform(0, 2*np.pi, C_2d.shape)
    phase = np.exp(1j * phi)

    h_r = fft.ifft2(h_q * phase).real
    return h_r
