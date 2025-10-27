"""
modelling and computing are coupled here.
"""

import logging
import types

import numpy as np

from a_package.grid import Grid
from a_package.field import Field, adapt_shape
from a_package.models import CapillaryBridge
from a_package.numeric.fem import FirstOrderElement
from a_package.numeric.quadrature import Quadrature, centroid_gaussian_quadrature


logger = logging.getLogger(__name__)


class NodalFormCapillary:
    """All necessary methods to formulate into a optimisation problem."""

    nodal_shape: list[int]
    nodal_gap: Field
    nodal_phase: Field
    quad_phase: Field
    quad_phase_gradient: Field
    bridge: CapillaryBridge
    fem: FirstOrderElement
    quadrature: Quadrature
    phase_lb: float
    phase_ub: float

    def __init__(self, grid: Grid, capillary_args: dict, quadrature: Quadrature = centroid_gaussian_quadrature):
        self.grid = grid
        self.nodal_shape = [1, 1, *grid.nb_elements]
        quadr_shape = [1, quadrature.nb_quad_pts, *grid.nb_elements]

        self.nodal_gap = np.zeros(self.nodal_shape)
        self.nodal_phase = np.zeros(self.nodal_shape)
        self.quad_phase = np.zeros(quadr_shape)
        self.quad_phase_gradient = np.zeros(quadr_shape)

        self.bridge = CapillaryBridge(**capillary_args)
        self.fem = FirstOrderElement(grid, quadrature.quad_pt_coords)
        self.quadrature = quadrature

    def get_gap(self):
        return self.nodal_gap

    def set_gap(self, value):
        self.nodal_gap = adapt_shape(value)
        # map to quadrature points
        self.bridge.gap = self.fem.interpolate_value(self.nodal_gap)

    @property
    def quadr_gap(self):
        return self.bridge.gap

    @property
    def in_contact_part(self):
        return self.nodal_gap == 0

    def get_phase(self):
        return self.nodal_phase

    @property
    def phase_lb(self):
        return self.bridge.phase_vapour

    @property
    def phase_ub(self):
        return self.bridge.phase_liquid

    def set_phase(self, value):
        # Clean the phase-field where the solid bodies contact
        value[self.in_contact_part] = 0.
        self.nodal_phase = value
        # map to quadrature points
        self.quadr_phase = self.fem.interpolate_value(self.nodal_phase)
        # self._quad_phase_grad = np.stack(
        #     [self.fem.interpolate_gradient_x(nodal_phase), self.fem.interpolate_gradient_y(nodal_phase)],
        #     axis=0
        # )
        self.quadr_phase_grad = self.fem.interpolate_gradient(self.nodal_phase)

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
        integrand = self.bridge.compute_energy(self.quadr_phase, self.quadr_phase_grad)
        return self.quadrature.integrate(self.grid, integrand).item()

    def get_energy_jacobian(self):
        [energy_D_phase, energy_D_phase_grad] = self.bridge.compute_energy_jacobian(
            self.quadr_phase, self.quadr_phase_grad)
        jacobian = self.fem.propag_sens_value(self.quadrature.propag_integral_weight(
            self.grid, energy_D_phase)) + self.fem.propag_sens_gradient(self.quadrature.propag_integral_weight(
            self.grid, energy_D_phase_grad))
        jacobian[self.in_contact_part] = 0
        return jacobian

    def get_volume(self):
        integrand = self.bridge.compute_volume(self.quadr_phase)
        return self.quadrature.integrate(self.grid, integrand).item()

    def get_volume_jacobian(self):
        [volume_D_phase] = self.bridge.compute_volume_jacobian(self.quadr_phase)
        jacobian = self.fem.propag_sens_value(self.quadrature.propag_integral_weight(self.grid, volume_D_phase))
        jacobian[self.in_contact_part] = 0
        return jacobian

    def get_perimeter(self):
        integrand = self.bridge.compute_perimeter(self.quadr_phase, self.quadr_phase_grad)
        return self.quadrature.integrate(self.grid, integrand).item()

    def create_numopt_with_constant_volume(self, volume: float):

        def volume_constraint():
            return self.get_volume() - volume

        return types.SimpleNamespace(
            get_x=self.get_phase,
            set_x=self.set_phase,
            get_f=self.get_energy,
            get_f_Dx=self.get_energy_jacobian,
            get_g=volume_constraint,
            get_g_Dx=self.get_volume_jacobian,
            x_lb=self.phase_lb,
            x_ub=self.phase_ub,
        )
