"""Capillary bridge physics model.

The model is agnostic to discretization details, so functions in this file are
limited to Field(s) -> Field(s) relations. Operators such as integral and differential
are implemented in 'numerics'. Only modify the component axis; keep the axis even if
it can be squeezed, i.e., #components == 1
"""

import dataclasses as dc

import numpy as np

from a_package.domain import Field, field_component_ax


@dc.dataclass(init=True)
class CapillaryBridge:
    eta: float
    """interface thickness"""
    theta: float
    """contact angle"""

    def __post_init__(self):
        # convention
        self.phase_vapour = 0.
        self.phase_liquid = 1.
        # According to Modica-Mortola's theorem, the perimeter of liquid-vapour interface is propotional to its energy.
        # That propotion equals to the integral of the square root of the double-well penalty, on the interval connected
        # by two phases. Therefore, we have to set a prefactor equal to the inverse of that propotion, then that value
        # would exactly be the perimeter
        self.perimeter_prefactor = 3.

        # more parameters
        self.curv = 0.5 * (abs(np.sin(self.theta)) + np.asin(np.cos(self.theta)) / np.cos(self.theta))
        self.gamma = -np.cos(self.theta)

        # To save gap heights in quadrature points
        self.gap: Field = None
        """gap between two rigid bodies"""

    def compute_perimeter(self, phase: Field, phase_grad: Field):
        """
        - phase: phase-field
        - phase_grad: gradient of phase-field
        """
        return self.perimeter_prefactor * (
            (1 / self.eta) * self.double_well_penalty(phase) + self.eta * self.square_penalty(phase_grad)
        )

    def compute_energy(self, phase: Field, phase_grad: Field):
        """
        - phase: phase-field
        - phase_grad: gradient of phase-field
        """
        liquid_vapour = (
            self.perimeter_prefactor
            * ((1 / self.eta) * self.double_well_penalty(phase) + self.eta * self.square_penalty(phase_grad))
            * self.gap
            * self.curv
        )
        # upper and lower surface, so the 2. (height gradient squareis one order higher and omitted)
        liquid_solid = 2.0 * phase
        return liquid_vapour + self.gamma * liquid_solid

    @staticmethod
    def double_well_penalty(x):
        return np.sum(x**2 * (1 - x) ** 2, axis=field_component_ax, keepdims=True)

    @staticmethod
    def square_penalty(x):
        return np.sum(x**2, axis=field_component_ax, keepdims=True)

    # def compute_force(self, phase: Field, phase_grad: Field):
    #     # the common part of 3 components
    #     liquid_vapour_D_gap = (
    #         self.perimeter_prefactor
    #         * ((1 / self.eta) * self.double_well_penalty(phase) + self.eta * self.square_penalty(phase_grad))
    #         * self.curv
    #     )

    #     # the different part of 3 components
    #     f_x = (liquid_vapour_D_gap * self.dg_dx).sum()
    #     f_y = (liquid_vapour_D_gap * self.dg_dy).sum()
    #     f_z = liquid_vapour_D_gap.sum()  # dg_dz = 1

    #     return f_x, f_y, f_z

    def compute_energy_jacobian(self, phase: Field, phase_grad: Field):
        """
        - phase: phase-field
        - phase_grad: gradient of phase-field
        """
        liquid_vapour_D_phase = (
            self.perimeter_prefactor
            * ((1 / self.eta) * self.double_well_penalty_derivative(phase))
            * self.gap
            * self.curv
        )
        liquid_vapour_D_phase_grad = (
            self.perimeter_prefactor * (self.eta * self.square_penalty_derivatie(phase_grad)) * self.gap * self.curv
        )

        liquid_solid_D_phase = 2.0
        # liquid_vapour_D_phase_grad = 0.

        return liquid_vapour_D_phase + self.gamma * liquid_solid_D_phase, liquid_vapour_D_phase_grad

    @staticmethod
    def double_well_penalty_derivative(x):
        return 2 * x * (1 - x) * (1 - 2 * x)

    @staticmethod
    def square_penalty_derivatie(x):
        return 2 * x

    def compute_volume(self, phase: Field):
        return phase * self.gap

    def compute_volume_jacobian(self, phase: Field):
        return (self.gap,)
