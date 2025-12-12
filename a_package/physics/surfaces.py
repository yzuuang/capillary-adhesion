"""
Surface geometry generators.

Each generator creates a height field on a given grid based on configuration parameters.
"""

from typing import Any

import numpy as np
import numpy.random as random

from a_package.grid import Grid
from a_package.physics.models import SelfAffineRoughness, psd_to_height


def generate_surface(grid: Grid, shape: str, params: dict[str, Any]) -> np.ndarray:
    """
    Generate a surface height field based on shape type and parameters.

    Parameters
    ----------
    grid : Grid
        The computational grid.
    shape : str
        Surface shape: "flat", "tip", "sinusoid", "rough", or "pattern".
    params : dict
        Shape-specific parameters.

    Returns
    -------
    np.ndarray
        Height field array with shape matching grid.nb_elements.
    """
    generators = {
        "flat": _generate_flat,
        "tip": _generate_tip,
        "sinusoid": _generate_sinusoid,
        "rough": _generate_rough,
        "pattern": _generate_pattern,
    }
    if shape not in generators:
        raise ValueError(f"Unknown surface shape: {shape}. "
                         f"Available: {list(generators.keys())}")
    return generators[shape](grid, params)


def _generate_flat(grid: Grid, params: dict[str, Any]) -> np.ndarray:
    """Generate a flat surface at constant height."""
    constant = float(params.get("constant", 0.0))
    return constant * np.ones(grid.nb_elements)


def _generate_tip(grid: Grid, params: dict[str, Any]) -> np.ndarray:
    """Generate a spherical tip (paraboloid approximation)."""
    R = float(params["radius"])
    [lx, ly] = grid.lengths
    x_center = 0.5 * lx
    y_center = 0.5 * ly
    [x, y] = grid.form_nodal_mesh()
    height = -np.sqrt(np.clip(R**2 - (x - x_center)**2 - (y - y_center)**2, 0, None))
    # Set lowest point to zero
    height += np.amax(abs(height))
    return height


def _generate_sinusoid(grid: Grid, params: dict[str, Any]) -> np.ndarray:
    """Generate a sinusoidal surface."""
    wave_num = float(params["wavenumber"])
    wave_amp = float(params["amplitude"])
    [x, y] = grid.form_nodal_mesh()
    [qx, qy] = grid.form_spectral_mesh()
    height = wave_amp * np.cos(qx * x) * np.cos(qy * y)
    return height


def _generate_rough(grid: Grid, params: dict[str, Any]) -> np.ndarray:
    """Generate a self-affine rough surface from PSD."""
    C0 = float(params["prefactor"])
    nR = float(params["rolloff_wavelength_pixels"])
    qR = (2 * np.pi) / (grid.element_sizes[0] * nR)  # roll-off wave vector
    nS = float(params["cutoff_wavelength_pixels"])
    qS = (2 * np.pi) / (grid.element_sizes[0] * nS)  # cut-off wave vector
    H = float(params["hurst_exponent"])

    roughness = SelfAffineRoughness(C0, qR, qS, H)
    q_2D = grid.form_spectral_mesh()
    [_, C_2D] = roughness.mapto_isotropic_psd(q_2D)

    # Get or generate the seed
    seed = params.get("seed")
    if seed is not None:
        seed = int(seed)
    seq = random.SeedSequence(seed)

    # Generate rough surface from PSD
    rng = random.default_rng(seq)
    height = psd_to_height(C_2D, rng=rng)
    return height.squeeze(axis=0)


def _generate_pattern(grid: Grid, params: dict[str, Any]) -> np.ndarray:
    """Generate a multi-scale wave pattern surface."""
    [xm, ym] = grid.form_nodal_mesh()
    x0 = float(params.get("tip_center_x", 0.0))
    y0 = float(params.get("tip_center_y", 0.0))

    height = np.zeros_like(xm)

    # Large wavelength component
    if "wave_len_L" in params and "wave_amp_L" in params:
        wave_len_L = float(params["wave_len_L"])
        wave_amp_L = float(params["wave_amp_L"])
        height += wave_amp_L * np.cos(2 * np.pi / wave_len_L * (xm - x0)) * \
                  np.cos(2 * np.pi / wave_len_L * (ym - y0))

    # Medium wavelength component
    if "wave_len_M" in params and "wave_amp_M" in params:
        wave_len_M = float(params["wave_len_M"])
        wave_amp_M = float(params["wave_amp_M"])
        height += wave_amp_M * np.cos(2 * np.pi / wave_len_M * (xm - x0)) * \
                  np.cos(2 * np.pi / wave_len_M * (ym - y0))

    # Small wavelength component
    if "wave_len_S" in params and "wave_amp_S" in params:
        wave_len_S = float(params["wave_len_S"])
        wave_amp_S = float(params["wave_amp_S"])
        height += wave_amp_S * np.cos(2 * np.pi / wave_len_S * (xm - x0)) * \
                  np.cos(2 * np.pi / wave_len_S * (ym - y0))

    return np.atleast_2d(height)
