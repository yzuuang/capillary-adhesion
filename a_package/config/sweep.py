"""
Parameter sweep expansion for parametric exploration.

Expands a config with sweep specifications into multiple configs,
one for each parameter combination.
"""

import copy
import itertools
from dataclasses import replace
from typing import Any, Iterator

import numpy as np

from .schema import Config, SweepConfig


def expand_sweeps(config: Config) -> Iterator[Config]:
    """
    Expand a configuration with sweeps into individual configurations.

    If no sweeps are defined, yields the original config once.

    Parameters
    ----------
    config : Config
        Configuration potentially containing sweep specifications.

    Yields
    ------
    Config
        Individual configurations with sweep parameters resolved.
    """
    if not config.sweeps:
        # No sweeps defined, return the original config
        yield replace(config, sweeps=[])
        return

    # Expand each sweep into values
    sweep_values: list[tuple[str, list[Any]]] = []
    for sweep in config.sweeps:
        values = _expand_sweep_values(sweep)
        sweep_values.append((sweep.path, values))

    # Compute the Cartesian product of all sweep parameters
    paths = [path for path, _ in sweep_values]
    value_lists = [values for _, values in sweep_values]

    for combo in itertools.product(*value_lists):
        # Create a new config with sweep values applied
        new_config = _apply_values(config, paths, combo)
        # Remove sweep specifications from the expanded config
        new_config = replace(new_config, sweeps=[])
        yield new_config


def _expand_sweep_values(sweep: SweepConfig) -> list[Any]:
    """Convert sweep specification to list of values."""
    if sweep.linspace is not None:
        start, stop, num = sweep.linspace
        return np.linspace(start, stop, int(num)).tolist()
    elif sweep.logspace is not None:
        start, stop, num = sweep.logspace
        return np.logspace(start, stop, int(num)).tolist()
    elif sweep.values is not None:
        return list(sweep.values)
    else:
        raise ValueError(f"Sweep at path '{sweep.path}' has no values specified. "
                         "Use linspace, logspace, or values.")


def _apply_values(config: Config, paths: list[str], values: tuple[Any, ...]) -> Config:
    """Apply sweep values to a config by modifying the nested attributes."""
    # Work on a deep copy to avoid mutating the original
    new_config = copy.deepcopy(config)

    for path, value in zip(paths, values):
        _set_nested_attr(new_config, path, value)

    return new_config


def _set_nested_attr(obj: Any, path: str, value: Any) -> None:
    """
    Set a nested attribute using dot notation.

    Example: path="physics.capillary.contact_angle_degree" sets
             obj.physics.capillary.contact_angle_degree = value
    """
    parts = path.split(".")
    # Navigate to the parent object
    for part in parts[:-1]:
        obj = getattr(obj, part)
    # Set the final attribute
    setattr(obj, parts[-1], value)


def _get_nested_attr(obj: Any, path: str) -> Any:
    """
    Get a nested attribute using dot notation.

    Example: path="physics.capillary.contact_angle_degree" returns
             obj.physics.capillary.contact_angle_degree
    """
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def count_sweep_combinations(config: Config) -> int:
    """
    Count the total number of configurations that would be generated.

    Returns 1 if no sweeps are defined.
    """
    if not config.sweeps:
        return 1

    total = 1
    for sweep in config.sweeps:
        values = _expand_sweep_values(sweep)
        total *= len(values)
    return total
