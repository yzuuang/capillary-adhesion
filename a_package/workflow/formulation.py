"""
modelling and computing are coupled here.
"""

import logging
import dataclasses as dc

import numpy as np

from a_package.models import CapillaryBridge
from a_package.numeric import Grid, FirstOrderElement, NumOptEq


logger = logging.getLogger(__name__)


@dc.dataclass(init=True)
class Formulation:
    """All necessary methods to formulate into a optimisation problem."""

    grid: Grid
    upper: np.ndarray
    lower: np.ndarray
    capi: CapillaryBridge

    def __post_init__(self):
        self.fem = FirstOrderElement(self.grid)
        self.element_area = 0.5 * self.grid.dx * self.grid.dy
        self.at_contact = None
        # these values are for quadrature points
        self._quad_phase = None
        self._quad_phase_grad = None

    def get_gap(self, z1: float):
        """For the sake of post-processing."""
        return np.clip(self.upper + z1 - self.lower, 0, None) 

    def update_gap(self, z1: float):
        height_diff = self.upper + z1 - self.lower
        # match the shape as FEM expects a vector
        height_diff = np.ravel(height_diff)

        self.at_contact = np.nonzero(height_diff < 0)
        # ideal plastic contact, material interpenetration
        gap = np.clip(height_diff, 0, None)
        # map to quadrature points & match the shape as modelling expects components as the first dimension
        self.capi.gap = np.vstack([self.fem.K_centroid @ gap])

    def update_phase_field(self, nodal_phase: np.ndarray):
        nodal_phase = np.ravel(nodal_phase)
        # Clean the phase-field where the solid bodies contact
        nodal_phase[self.at_contact] = 0.0
        # map to quadrature points & match the shape as modelling expects components as the first dimension
        self._quad_phase = np.vstack([self.fem.K_centroid @ nodal_phase])
        self._quad_phase_grad = np.vstack([self.fem.Dx @ nodal_phase, self.fem.Dy @ nodal_phase])

    def validate_phase_field(self, nodal_phase: np.ndarray):
        # check phase field < 0
        if np.any(nodal_phase < 0):
            outlier = np.where(nodal_phase < 0, nodal_phase, np.nan)
            count = np.count_nonzero(~np.isnan(outlier))
            extreme = np.nanmin(outlier)
            logger.warning(f"Notice: phase field has {count} values < 0, min at {extreme:.2e}")
        # check phase field > 1
        if np.any(nodal_phase > 1):
            outlier = np.where(nodal_phase > 1, nodal_phase, np.nan)
            count = np.count_nonzero(~np.isnan(outlier))
            extreme = np.nanmax(outlier)
            logger.warning(f"Notice: phase field has {count} values > 1, max at 1.0+{extreme - 1:.2e}.")

    def get_energy(self):
        integrand = self.capi.compute_energy(self._quad_phase, self._quad_phase_grad)
        return self.element_area * np.sum(integrand)

    def get_volume(self):
        integrand = self.capi.compute_volume(self._quad_phase)
        return self.element_area * np.sum(integrand)

    def get_perimeter(self):
        integrand = self.capi.compute_perimeter(self._quad_phase, self._quad_phase_grad)
        return self.element_area * np.sum(integrand)

    def create_numopt_with_constant_volume(self, volume: float):

        def objective(x: np.ndarray):
            self.update_phase_field(x)
            return self.get_energy()

        def objective_jacobian(x: np.ndarray):
            self.update_phase_field(x)
            [energy_D_phase, energy_D_phase_grad] = self.capi.compute_energy_jacobian(
                self._quad_phase, self._quad_phase_grad
            )
            return (
                self.element_area
                * (
                    # map from quadrature points back to the nodes
                    energy_D_phase @ self.fem.K_centroid
                    + energy_D_phase_grad[0] @ self.fem.Dx
                    + energy_D_phase_grad[1] @ self.fem.Dy
                ).ravel()
            )

        def constraint(x: np.ndarray):
            self.update_phase_field(x)
            return self.get_volume() - volume

        def constraint_jacobian(x: np.ndarray):
            self.update_phase_field(x)
            [volume_D_phase] = self.capi.compute_volume_jacobian(self._quad_phase)
            return (
                self.element_area
                * (
                    # map from quadrature points back to the nodes
                    volume_D_phase
                    @ self.fem.K_centroid
                ).ravel()
            )

        return NumOptEq(objective, objective_jacobian, constraint, constraint_jacobian)
