"""
Simulation routines: modelling, computing and minimising are coupled here.
"""

import logging
import dataclasses as dc

import numpy as np

from a_package.modelling import CapillaryBridge
from a_package.computing import Grid, FirstOrderElement
from a_package.minimising import NumOptEq


logger = logging.getLogger(__name__)


@dc.dataclass(init=True)
class Formulation:
    """All necessary parameters for visualizing a capillary bridge.

    Data attributes: field is stored with nodal values as 2D array.
    1 refer to the solid top; 2 refer to the solid base.
    """

    grid: Grid
    upper: np.ndarray
    lower: np.ndarray
    capi: CapillaryBridge

    def __post_init__(self):
        self.fem = FirstOrderElement(self.grid)
        self.element_area = 0.5 * self.grid.lx * self.grid.ly
        self.at_contact = None
        self.phase = None
        self.phase_grad = None

    def update_gap(self, z1: float):
        height_diff = self.upper + z1 - self.lower
        self.at_contact = np.nonzero(height_diff < 0)
        # ideal plastic contact, material interpenetration
        gap = np.clip(height_diff, 0, None)
        # map to quadrature points
        self.capi.gap = self.fem.K_centroid @ gap

    def update_phase_field(self, phase: np.ndarray):
        # Clean the phase-field where the solid bodies contact
        phase[self.at_contact] = 0.0
        # map to quadrature points
        self.phase = self.fem.K_centroid @ phase
        self.phase_grad = np.stack([self.fem.Dx @ phase, self.fem.Dy @ phase], axis=0)

    def validate_phase_field(self, phase: np.ndarray):
        # check phase field < 0
        if np.any(phase < 0):
            outlier = np.where(phase < 0, phase, np.nan)
            count = np.count_nonzero(~np.isnan(outlier))
            extreme = np.nanmin(outlier)
            logger.warning(f"Notice: phase field has {count} values < 0, min at {extreme:.2e}")
        # check phase field > 1
        if np.any(phase > 1):
            outlier = np.where(phase > 1, phase, np.nan)
            count = np.count_nonzero(~np.isnan(outlier))
            extreme = np.nanmax(outlier)
            logger.warning(f"Notice: phase field has {count} values > 1, max at 1.0+{extreme - 1:.2e}.")

    def get_volume(self):
        integrand = self.capi.compute_volume(self.phase)
        spatial_axes = np.arange(1, np.ndim(integrand))
        return self.element_area * np.sum(integrand, spatial_axes)

    def formulate_with_constant_volume(self, volume: float):

        def objective(x: np.ndarray):
            self.update_phase_field(x)
            integrand = self.capi.compute_energy(self.phase, self.phase_grad)
            spatial_axes = np.arange(1, np.ndim(integrand))
            return self.element_area * np.sum(integrand, spatial_axes)

        def objective_jacobian(x: np.ndarray):
            self.update_phase_field(x)
            [energy_D_phase, energy_D_phase_grad] = self.capi.compute_energy_jacobian(self.phase, self.phase_grad)
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
            integrand = self.capi.compute_volume(self.phase)
            spatial_axes = np.arange(1, np.ndim(integrand))
            return self.element_area * np.sum(integrand, spatial_axes) - volume

        def constraint_jacobian(x: np.ndarray):
            self.update_phase_field(x)
            [volume_D_phase] = self.capi.compute_volume_jacobian(self.phase)
            return (
                self.element_area
                * (
                    # map from quadrature points back to the nodes
                    volume_D_phase
                    @ self.fem.K_centroid
                ).ravel()
            )

        return NumOptEq(objective, objective_jacobian, constraint, constraint_jacobian)
