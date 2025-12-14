"""
Contact solvers.

Computes the gap field between surfaces at a given separation.
Future: elastic deformation, adhesive contact, etc.
"""

import numpy as np

from a_package.domain import adapt_shape


class RigidContactSolver:
    """Computes the gap field between two surfaces at a given separation."""

    def __init__(self, upper: np.ndarray, lower: np.ndarray):
        self.upper = adapt_shape(upper)
        self.lower = adapt_shape(lower)

    def solve_gap(self, separation: float):
        return np.clip(separation + self.upper - self.lower, 0, None)
