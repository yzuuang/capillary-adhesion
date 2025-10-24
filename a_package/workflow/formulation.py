"""
modelling and computing are coupled here.
"""

import logging
import dataclasses as dc
import types

import numpy as np

from a_package.field import Field
from a_package.models import CapillaryBridge
from a_package.numeric.compute import FirstOrderElement
from a_package.grid import Grid


logger = logging.getLogger(__name__)


@dc.dataclass(init=True)
class Formulation:
    """All necessary methods to formulate into a optimisation problem."""

    grid: Grid
    upper: Field
    lower: Field
    capi: CapillaryBridge

    def __post_init__(self):
        self.fem = FirstOrderElement(self.grid)
        self.element_area = 0.5 * self.grid.element_area
        # these values are for quadrature points
        self.at_contact = None
        nb_nodes = 1
        self.nodal_phase = np.zeros((1, nb_nodes, *self.grid.nb_elements))
        nb_quad_pts = 2
        self._quad_gap = np.zeros((1, nb_quad_pts, *self.grid.nb_elements))
        self._quad_phase = np.zeros((1, nb_quad_pts, *self.grid.nb_elements))
        self._quad_phase_grad = np.zeros((2, nb_quad_pts, *self.grid.nb_elements))

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
        self.capi.gap = np.stack([self.fem.interpolate_value(gap)], axis=0) 

    def get_phase_field(self):
        return self.nodal_phase

    def update_phase_field(self, nodal_phase: Field):
        self.nodal_phase = np.ravel(nodal_phase)
        # Clean the phase-field where the solid bodies contact
        self.nodal_phase[self.at_contact] = 0.0
        # map to quadrature points & match the shape as modelling expects components as the first dimension
        self._quad_phase = np.stack([self.fem.interpolate_value(nodal_phase)], axis=0)
        # self._quad_phase_grad = np.stack(
        #     [self.fem.interpolate_gradient_x(nodal_phase), self.fem.interpolate_gradient_y(nodal_phase)],
        #     axis=0
        # )
        self._quad_phase_grad = self.fem.interpolate_gradient(nodal_phase)

    def validate_phase_field(self, nodal_phase: Field):
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

    def get_energy_jacobian(self):
        [energy_D_phase, energy_D_phase_grad] = self.capi.compute_energy_jacobian(
            self._quad_phase, self._quad_phase_grad
        )
        raw = self.element_area * (
            # map from quadrature points back to the nodes
            self.fem.propag_sens_value(energy_D_phase)
            # + self.fem.propag_sens_gradient_x(energy_D_phase_grad[0])
            # + self.fem.propag_sens_gradient_y(energy_D_phase_grad[1])
            + self.fem.propag_sens_gradient(energy_D_phase_grad)
        )
        shape = raw.shape
        raw = raw.ravel()
        raw[self.at_contact] = 0
        return raw.reshape(shape)

    def get_volume(self):
        integrand = self.capi.compute_volume(self._quad_phase)
        return self.element_area * np.sum(integrand)

    def get_volume_jacobian(self):
        [volume_D_phase] = self.capi.compute_volume_jacobian(self._quad_phase)
        raw = (
            self.element_area
            * (
                # map from quadrature points back to the nodes
                self.fem.propag_sens_value(volume_D_phase)
            )
        )
        shape = raw.shape
        raw = raw.ravel()
        raw[self.at_contact] = 0
        return raw.reshape(shape)

    def get_perimeter(self):
        integrand = self.capi.compute_perimeter(self._quad_phase, self._quad_phase_grad)
        return self.element_area * np.sum(integrand)

    def create_numopt_with_constant_volume(self, volume: float):

        def volume_constraint():
            return self.get_volume() - volume

        return types.SimpleNamespace(
            get_x=self.get_phase_field,
            set_x=self.update_phase_field,
            get_f=self.get_energy,
            get_f_Dx=self.get_energy_jacobian,
            get_g=volume_constraint,
            get_g_Dx=self.get_volume_jacobian,
            x_lb=self.capi.phase_vapour,
            x_ub=self.capi.phase_liquid,
        )
