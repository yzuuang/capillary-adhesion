"""
TOML configuration loading and saving.

Uses tomllib (Python 3.11+) or tomli (backport) for reading,
and tomli_w for writing.
"""

import copy
import sys
from pathlib import Path
from typing import Any

# Import tomllib or backport
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import tomli_w

from .schema import Config


def load_config(path: str | Path) -> Config:
    """Load a TOML configuration file and return a Config object."""
    path = Path(path)
    with open(path, "rb") as f:
        data = tomllib.load(f)

    # Extract sweeps (TOML uses [[sweep]] array syntax)
    sweeps = data.pop("sweep", [])

    return Config(
        domain=data["domain"],
        problem=data["problem"],
        solver=data["solver"],
        simulation=data["simulation"],
        sweeps=sweeps,
    )


def save_config(config: Config, path: str | Path) -> None:
    """Save a Config object to a TOML file."""
    path = Path(path)
    data: dict[str, Any] = {}

    if config.domain:
        data["domain"] = config.domain
    if config.problem:
        data["problem"] = config.problem
    if config.solver:
        data["solver"] = config.solver
    if config.simulation:
        data["simulation"] = config.simulation
    if config.sweeps:
        data["sweep"] = config.sweeps

    with open(path, "wb") as f:
        tomli_w.dump(data, f)


def get_surface_shape(config: Config, which: str) -> str:
    """
    Get the shape name for a surface.

    Parameters
    ----------
    config : Config
        The configuration object.
    which : str
        Either "upper" or "lower".

    Returns
    -------
    str
        The surface shape name.
    """
    return config.problem[which]["shape"]
