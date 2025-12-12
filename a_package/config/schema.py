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


@dataclass
class SurfaceConfig:
    """
    Generic surface configuration.

    Surface types are identified by shape string, with type-specific
    parameters stored in params dict. This keeps config decoupled from
    physics implementation.

    Supported shapes and their parameters:
    - "flat": constant (default: 0.0)
    - "tip": radius
    - "sinusoid": wavenumber, amplitude
    - "rough": prefactor, rolloff_wavelength_pixels, cutoff_wavelength_pixels,
               hurst_exponent, seed (optional)
    - "pattern": tip_center_x, tip_center_y, wave_len_L, wave_amp_L, etc.
    """
    shape: str
    params: dict[str, Any] = field(default_factory=dict)


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
