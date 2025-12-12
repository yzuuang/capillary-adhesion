"""
TOML configuration loading and saving.

Uses tomllib (Python 3.11+) or tomli (backport) for reading,
and tomli_w for writing.
"""

import sys
from dataclasses import fields, asdict
from pathlib import Path
from typing import Any

# Import tomllib or backport
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import tomli_w

from .schema import (
    Config,
    GeometryConfig,
    GridConfig,
    SurfaceConfig,
    FlatSurface,
    TipSurface,
    SinusoidSurface,
    RoughSurface,
    PatternSurface,
    PhysicsConfig,
    CapillaryConfig,
    SimulationConfig,
    TrajectoryConfig,
    SolverConfig,
    SweepConfig,
)


# Mapping from shape name to surface dataclass
_surface_classes = {
    "flat": FlatSurface,
    "tip": TipSurface,
    "sinusoid": SinusoidSurface,
    "rough": RoughSurface,
    "pattern": PatternSurface,
}

# Reverse mapping from dataclass to shape name
_surface_names = {cls: name for name, cls in _surface_classes.items()}


def load_config(path: str | Path) -> Config:
    """Load a TOML configuration file and return a Config object."""
    path = Path(path)
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return _dict_to_config(data)


def save_config(config: Config, path: str | Path) -> None:
    """Save a Config object to a TOML file."""
    path = Path(path)
    data = _config_to_dict(config)
    with open(path, "wb") as f:
        tomli_w.dump(data, f)


def _dict_to_config(data: dict[str, Any]) -> Config:
    """Convert a dictionary (from TOML) to a Config object."""
    # Parse geometry section
    geom = data["geometry"]
    grid_config = GridConfig(
        pixel_size=geom["grid"]["pixel_size"],
        nb_pixels=geom["grid"]["nb_pixels"],
    )
    upper_config = _parse_surface(geom["upper"])
    lower_config = _parse_surface(geom["lower"])
    geometry = GeometryConfig(grid=grid_config, upper=upper_config, lower=lower_config)

    # Parse physics section
    phys = data["physics"]
    capillary_config = CapillaryConfig(
        interface_thickness=phys["capillary"]["interface_thickness"],
        contact_angle_degree=phys["capillary"]["contact_angle_degree"],
        liquid_volume_percent=phys["capillary"]["liquid_volume_percent"],
    )
    physics = PhysicsConfig(capillary=capillary_config)

    # Parse simulation section
    sim = data["simulation"]
    trajectory_config = TrajectoryConfig(
        min_separation=sim["trajectory"]["min_separation"],
        max_separation=sim["trajectory"]["max_separation"],
        step_size=sim["trajectory"]["step_size"],
    )
    solver_config = SolverConfig(
        max_nb_iters=sim["solver"]["max_nb_iters"],
        max_nb_loops=sim["solver"]["max_nb_loops"],
        tol_convergence=sim["solver"]["tol_convergence"],
        tol_constraints=sim["solver"]["tol_constraints"],
        init_penalty_weight=sim["solver"]["init_penalty_weight"],
    )
    simulation = SimulationConfig(trajectory=trajectory_config, solver=solver_config)

    # Parse sweeps (optional)
    sweeps = []
    if "sweep" in data:
        for sweep_data in data["sweep"]:
            sweeps.append(SweepConfig(
                path=sweep_data["path"],
                linspace=sweep_data.get("linspace"),
                logspace=sweep_data.get("logspace"),
                values=sweep_data.get("values"),
            ))

    return Config(geometry=geometry, physics=physics, simulation=simulation, sweeps=sweeps)


def _parse_surface(surface_data: dict[str, Any]) -> SurfaceConfig:
    """Parse surface configuration by dispatching to the correct dataclass."""
    # Make a copy to avoid modifying the original
    data = dict(surface_data)
    shape = data.pop("shape")

    if shape not in _surface_classes:
        raise ValueError(f"Unknown surface shape: {shape}. "
                         f"Available: {list(_surface_classes.keys())}")

    cls = _surface_classes[shape]
    return cls(**data)


def _surface_to_dict(surface: SurfaceConfig) -> dict[str, Any]:
    """Convert a surface config to a dict with shape field."""
    shape = _surface_names[type(surface)]
    data = asdict(surface)
    # Remove None values for cleaner TOML output
    data = {k: v for k, v in data.items() if v is not None}
    return {"shape": shape, **data}


def _config_to_dict(config: Config) -> dict[str, Any]:
    """Convert a Config object to a dictionary (for TOML serialization)."""
    data = {
        "geometry": {
            "grid": {
                "pixel_size": config.geometry.grid.pixel_size,
                "nb_pixels": config.geometry.grid.nb_pixels,
            },
            "upper": _surface_to_dict(config.geometry.upper),
            "lower": _surface_to_dict(config.geometry.lower),
        },
        "physics": {
            "capillary": {
                "interface_thickness": config.physics.capillary.interface_thickness,
                "contact_angle_degree": config.physics.capillary.contact_angle_degree,
                "liquid_volume_percent": config.physics.capillary.liquid_volume_percent,
            }
        },
        "simulation": {
            "trajectory": {
                "min_separation": config.simulation.trajectory.min_separation,
                "max_separation": config.simulation.trajectory.max_separation,
                "step_size": config.simulation.trajectory.step_size,
            },
            "solver": {
                "max_nb_iters": config.simulation.solver.max_nb_iters,
                "max_nb_loops": config.simulation.solver.max_nb_loops,
                "tol_convergence": config.simulation.solver.tol_convergence,
                "tol_constraints": config.simulation.solver.tol_constraints,
                "init_penalty_weight": config.simulation.solver.init_penalty_weight,
            },
        },
    }

    # Add sweeps if present
    if config.sweeps:
        data["sweep"] = []
        for sweep in config.sweeps:
            sweep_dict: dict[str, Any] = {"path": sweep.path}
            if sweep.linspace is not None:
                sweep_dict["linspace"] = sweep.linspace
            if sweep.logspace is not None:
                sweep_dict["logspace"] = sweep.logspace
            if sweep.values is not None:
                sweep_dict["values"] = sweep.values
            data["sweep"].append(sweep_dict)

    return data


def get_surface_shape(surface: SurfaceConfig) -> str:
    """Get the shape name for a surface config object."""
    return _surface_names[type(surface)]
