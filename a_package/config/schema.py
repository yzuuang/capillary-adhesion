"""
Configuration schema.

Minimal schema that mirrors subpackage organization:
- domain: grid and surface definitions
- physics: capillary bridge parameters
- numerics: solver settings
- simulation: trajectory settings
- sweeps: parameter sweep specifications

Each section is a raw dict - semantic knowledge lives in the consuming code
(simulation/runner.py), not here. This avoids duplication between config
schema and physics/numerics classes.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Config:
    """
    Top-level configuration.

    First-level keys match subpackage names for clarity.
    All sections are raw dicts to avoid schema duplication.
    """
    domain: dict[str, Any]
    physics: dict[str, Any]
    numerics: dict[str, Any]
    simulation: dict[str, Any]
    sweeps: list[dict[str, Any]] = field(default_factory=list)
