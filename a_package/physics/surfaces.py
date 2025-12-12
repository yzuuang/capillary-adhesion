"""
Surface geometry generators.

Each generator creates a height field on a given grid based on surface configuration.
"""

from dataclasses import asdict

import numpy as np
import numpy.random as random

from a_package.grid import Grid
from a_package.config.schema import (
    SurfaceConfig,
    FlatSurface,
    TipSurface,
    SinusoidSurface,
    RoughSurface,
    PatternSurface,
)
from a_package.physics.models import SelfAffineRoughness, psd_to_height


def generate_surface(grid: Grid, surface: SurfaceConfig) -> np.ndarray:
    """
    Generate a surface height field based on surface configuration.

    Parameters
    ----------
    grid : Grid
        The computational grid.
    surface : SurfaceConfig
        Surface configuration (FlatSurface, TipSurface, etc.).

    Returns
    -------
    np.ndarray
        Height field array with shape matching grid.nb_elements.
    """
    if isinstance(surface, FlatSurface):
        return _generate_flat(grid, surface)
    elif isinstance(surface, TipSurface):
        return _generate_tip(grid, surface)
    elif isinstance(surface, SinusoidSurface):
        return _generate_sinusoid(grid, surface)
    elif isinstance(surface, RoughSurface):
        return _generate_rough(grid, surface)
    elif isinstance(surface, PatternSurface):
        return _generate_pattern(grid, surface)
    else:
        raise ValueError(f"Unknown surface type: {type(surface).__name__}")


def _generate_flat(grid: Grid, surface: FlatSurface) -> np.ndarray:
    """Generate a flat surface at constant height."""
    return surface.constant * np.ones(grid.nb_elements)


def _generate_tip(grid: Grid, surface: TipSurface) -> np.ndarray:
    """Generate a spherical tip (paraboloid approximation)."""
    R = surface.radius
    [lx, ly] = grid.lengths
    x_center = 0.5 * lx
    y_center = 0.5 * ly
    [x, y] = grid.form_nodal_mesh()
    height = -np.sqrt(np.clip(R**2 - (x - x_center)**2 - (y - y_center)**2, 0, None))
    # Set lowest point to zero
    height += np.amax(abs(height))
    return height


def _generate_sinusoid(grid: Grid, surface: SinusoidSurface) -> np.ndarray:
    """Generate a sinusoidal surface."""
    [x, y] = grid.form_nodal_mesh()
    [qx, qy] = grid.form_spectral_mesh()
    height = surface.amplitude * np.cos(qx * x) * np.cos(qy * y)
    return height


def _generate_rough(grid: Grid, surface: RoughSurface) -> np.ndarray:
    """Generate a self-affine rough surface from PSD."""
    qR = (2 * np.pi) / (grid.element_sizes[0] * surface.rolloff_wavelength_pixels)
    qS = (2 * np.pi) / (grid.element_sizes[0] * surface.cutoff_wavelength_pixels)

    roughness = SelfAffineRoughness(
        surface.prefactor, qR, qS, surface.hurst_exponent
    )
    q_2D = grid.form_spectral_mesh()
    [_, C_2D] = roughness.mapto_isotropic_psd(q_2D)

    # Get or generate the seed
    seed = surface.seed
    seq = random.SeedSequence(seed)

    # Generate rough surface from PSD
    rng = random.default_rng(seq)
    height = psd_to_height(C_2D, rng=rng)
    return height.squeeze(axis=0)


def _generate_pattern(grid: Grid, surface: PatternSurface) -> np.ndarray:
    """Generate a multi-scale wave pattern surface."""
    [xm, ym] = grid.form_nodal_mesh()
    x0 = surface.tip_center_x
    y0 = surface.tip_center_y

    height = np.zeros_like(xm)

    # Large wavelength component
    if surface.wave_len_L is not None and surface.wave_amp_L is not None:
        height += surface.wave_amp_L * \
                  np.cos(2 * np.pi / surface.wave_len_L * (xm - x0)) * \
                  np.cos(2 * np.pi / surface.wave_len_L * (ym - y0))

    # Medium wavelength component
    if surface.wave_len_M is not None and surface.wave_amp_M is not None:
        height += surface.wave_amp_M * \
                  np.cos(2 * np.pi / surface.wave_len_M * (xm - x0)) * \
                  np.cos(2 * np.pi / surface.wave_len_M * (ym - y0))

    # Small wavelength component
    if surface.wave_len_S is not None and surface.wave_amp_S is not None:
        height += surface.wave_amp_S * \
                  np.cos(2 * np.pi / surface.wave_len_S * (xm - x0)) * \
                  np.cos(2 * np.pi / surface.wave_len_S * (ym - y0))

    return np.atleast_2d(height)
