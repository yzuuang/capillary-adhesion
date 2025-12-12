"""
Configuration schema using dataclasses.

TOML structure mirrors physical domains:
- [geometry]: grid and surface definitions
- [physics]: capillary bridge parameters
- [simulation]: trajectory and solver settings
- [[sweep]]: parameter sweep specifications (optional)
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GridConfig:
    """Grid discretization parameters."""
    pixel_size: float
    nb_pixels: int


# Surface type dataclasses
@dataclass
class FlatSurface:
    """Flat surface at constant height."""
    constant: float = 0.0


@dataclass
class TipSurface:
    """Spherical tip surface."""
    radius: float


@dataclass
class SinusoidSurface:
    """Sinusoidal surface."""
    wavenumber: float
    amplitude: float


@dataclass
class RoughSurface:
    """Self-affine rough surface from PSD."""
    prefactor: float
    rolloff_wavelength_pixels: float
    cutoff_wavelength_pixels: float
    hurst_exponent: float
    seed: int | None = None


@dataclass
class PatternSurface:
    """Multi-scale wave pattern surface."""
    tip_center_x: float = 0.0
    tip_center_y: float = 0.0
    wave_len_L: float | None = None
    wave_amp_L: float | None = None
    wave_len_M: float | None = None
    wave_amp_M: float | None = None
    wave_len_S: float | None = None
    wave_amp_S: float | None = None


SurfaceConfig = FlatSurface | TipSurface | SinusoidSurface | RoughSurface | PatternSurface


@dataclass
class GeometryConfig:
    """Geometry configuration grouping grid and surfaces."""
    grid: GridConfig
    upper: SurfaceConfig
    lower: SurfaceConfig


@dataclass
class CapillaryConfig:
    """Capillary bridge physical parameters."""
    interface_thickness: float
    contact_angle_degree: float
    liquid_volume_percent: float


@dataclass
class PhysicsConfig:
    """Physics configuration."""
    capillary: CapillaryConfig


@dataclass
class TrajectoryConfig:
    """Approach/retraction trajectory parameters."""
    min_separation: float
    max_separation: float
    step_size: float


@dataclass
class SolverConfig:
    """Solver parameters for the augmented Lagrangian method."""
    max_nb_iters: int
    max_nb_loops: int
    tol_convergence: float
    tol_constraints: float
    init_penalty_weight: float


@dataclass
class SimulationConfig:
    """Simulation settings."""
    trajectory: TrajectoryConfig
    solver: SolverConfig


@dataclass
class SweepConfig:
    """Parameter sweep specification."""
    path: str  # Dot-notation path, e.g., "physics.capillary.liquid_volume_percent"
    # One of the following must be specified:
    linspace: list[float] | None = None   # [start, stop, num]
    logspace: list[float] | None = None   # [start, stop, num] (powers of 10)
    values: list[Any] | None = None       # Explicit list of values


@dataclass
class Config:
    """Top-level configuration."""
    geometry: GeometryConfig
    physics: PhysicsConfig
    simulation: SimulationConfig
    sweeps: list[SweepConfig] = field(default_factory=list)
