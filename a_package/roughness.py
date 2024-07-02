import numpy as np
import numpy.fft as fft
from numpy.random import default_rng


rng = default_rng()

# TODO: Get rid of interpolation. Use 2*pi/L = qL and directly computed from linearly sampled (linspace) q-s.


def generate_isotropic_psd(C0: float, qL: float, qR: float, qS: float, H: float, N: int):
    """Generate isotropic PSD from a series of log-sampled q-s

    Args:
        C0: prefactor
        qR: _description_
        qL: roll-off frequency
        qS: cut-off frequency
        H: Hurst exponent
        N: # samples in spectral domain

    Returns:
        _type_: _description_
    """
    q_iso = np.geomspace(qL, qS, N)
    C_iso = np.full_like(q_iso, C0)
    C_iso[q_iso <= qR] *= qR**(-2-2*H)
    C_iso[q_iso > qR] *= q_iso[q_iso > qR]**(-2-2*H)
    return q_iso, C_iso


def interpolate_isotropic_psd_in_2d(q_iso: np.ndarray, C_iso: np.ndarray, Nx: int, dx: float, Ny: int, dy: float):
    """Convert an isotropic PSD (1D function) into plain PSD (2D function)"""
    qx = (2*np.pi) * fft.fftfreq(Nx, dx)
    qy = (2*np.pi) * fft.fftfreq(Ny, dy)
    qxx, qyy = np.meshgrid(qx, qy)
    q_2d = (qxx**2 + qyy**2)**(1/2)
    C_2d = np.interp(q_2d, q_iso, C_iso, left=0, right=0)
    return qx, qy, C_2d


# from PSD to surface
def convert_psd_to_surface(C_2d: np.ndarray):
    h_q = np.sqrt(C_2d)

    # impose some random phase angel
    phi = rng.uniform(0, 2*np.pi, C_2d.shape)
    phase = np.exp(1j * phi)

    h_r = fft.ifft2(h_q * phase).real
    return h_r
