"""
Configuration module for TOML-based simulation parameters.

Provides:
- Schema dataclasses for typed configuration
- TOML loading and saving
- Parameter sweep expansion for parametric exploration
"""

from .schema import (
    Config,
    GeometryConfig,
    GridConfig,
    SurfaceConfig,
    PhysicsConfig,
    CapillaryConfig,
    SimulationConfig,
    TrajectoryConfig,
    SolverConfig,
    SweepConfig,
)

from .loader import load_config, save_config

from .sweep import expand_sweeps, count_sweep_combinations

__all__ = [
    # Schema classes
    "Config",
    "GeometryConfig",
    "GridConfig",
    "SurfaceConfig",
    "PhysicsConfig",
    "CapillaryConfig",
    "SimulationConfig",
    "TrajectoryConfig",
    "SolverConfig",
    "SweepConfig",
    # Loader functions
    "load_config",
    "save_config",
    # Sweep functions
    "expand_sweeps",
    "count_sweep_combinations",
]
