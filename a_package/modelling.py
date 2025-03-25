"""This file addresses the physical perspectives.

- Roughness of the solid surface
- The "gap" formed between two solid surface with displacement
- Capillary bridge

NOTE: The model should be agonistic to discretization details, so functions in this file should
be limited to Field(s) -> Field(s) relations. Operators such as integral and differential are thus
implemented in "computing".
"""

import dataclasses as dc
import typing as _t

import numpy as np
import numpy.linalg as la


Components_t: _t.TypeAlias = np.ndarray
"""An array with the first dimension always corresponds to the number of components. One should 
consider other dimensions unknown (ravelled) because they are related to discretization.
"""


@dc.dataclass(init=True)
class SelfAffineRoughness:
    C0: float
    qR: float
    qS: float
    H: float

    def isotropic_spectrum(self, wave_numbers: Components_t):
        # Isotropic, so it only depends on its magnitude
        magnitude = la.norm(wave_numbers, ord=2, axis=0)

        # Find three regimes
        constant = magnitude < self.qR
        self_affine = (magnitude >= self.qR) & (magnitude < self.qS)
        omitted = magnitude >= self.qS

        # Evaluate accordingly
        psd = np.empty_like(magnitude)
        psd[constant] = self.C0 * self.qR ** (2 - 2 * self.H)
        psd[self_affine] = self.C0 * magnitude[self_affine] ** (-2 - 2 * self.H)
        psd[omitted] = 0

        # Return both in convenience of plotting
        return magnitude, psd


@dc.dataclass(init=True)
class SolidSolidContact:
    """Contact between two rough solid planes."""

    mean_plane_separation: float
    height_lower: np.ndarray

    def gap_height(self, height_upper: np.ndarray):
        """The gap between two rough surface where a thrid matter can exist.

        Assume interpenaltration, so clip all negative values to zero.
        """
        return np.clip(
            height_upper - self.height_lower + self.mean_plane_separation, a_min=0, a_max=None
        )


@dc.dataclass(init=True)
class CapillaryVapourLiquid:

    interfacial_width: float
    surface_tension_ratio: float
    heterogeneous_height: np.ndarray

    @property
    def solid_solid_contact(self):
        return np.nonzero(self.heterogeneous_height == 0)

    def energy_density(self, phi: Components_t, d_phi: Components_t):
        area_water_vapour = self.heterogeneous_height * (
            # double well penalty on phi
            (1 / self.interfacial_width) * self.double_well_penalty(phi)
            # square penalty on d_phi
            + self.interfacial_width * self.square_penalty(d_phi)
        )

        # FIXME: add the slope contribution.
        area_water_solid = 2 * phi

        return area_water_vapour - self.surface_tension_ratio * area_water_solid

    @staticmethod
    def double_well_penalty(x):
        return np.squeeze(9 * x**2 * (1 - x) ** 2, axis=0)

    @staticmethod
    def square_penalty(x):
        return np.sum(x**2, axis=0)

    def energy_density_sensitivity(self, phi: Components_t, d_phi: Components_t):
        area_water_vapour_sens_phi = self.heterogeneous_height * (
            # derivative of double well penalty w.r.t. phi
            (1 / self.interfacial_width)
            * self.double_well_penalty_derivative(phi)
        )
        # FIXME: add the slope contribution.
        area_water_solid_sens_phi = 2

        energy_density_sens_phi = (
            area_water_vapour_sens_phi - self.surface_tension_ratio * area_water_solid_sens_phi
        )

        energy_density_sens_d_phi = self.heterogeneous_height * (
            # derivative of square penalty w.r.t. d_phi
            self.interfacial_width
            * self.square_penalty_derivatie(d_phi)
        )

        return energy_density_sens_phi, energy_density_sens_d_phi

    @staticmethod
    def double_well_penalty_derivative(x):
        return 18 * x * (1 - x) * (1 - 2 * x)

    @staticmethod
    def square_penalty_derivatie(x):
        return 2 * x

    def liquid_height(self, phi: Components_t):
        return self.heterogeneous_height * phi

    def liquid_height_sensitivity(self, phi: Components_t):
        return (self.heterogeneous_height[np.newaxis,...], )

    def adhesive_force(self, phi: Components_t, d_phi: Components_t):
        return (
            # double well penalty on phi
            (1 / self.interfacial_width) * self.double_well_penalty(phi)
            # square penalty on d_phi
            + self.interfacial_width * self.square_penalty(d_phi)
        )
