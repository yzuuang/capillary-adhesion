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
import numpy.fft as fft
import numpy.random as random


components_t: _t.TypeAlias = np.ndarray
"""An array with the first dimension always corresponds to the number of components. One should 
consider other dimensions unknown (ravelled) because they are related to discretization.

**always keep the axis of components even if #components == 1.**
"""


@dc.dataclass(init=True)
class SelfAffineRoughness:
    C0: float
    """Prefactor"""
    qR: float
    """Roll-off (angular) wavenumber"""
    qS: float
    """Cut-off (angular) wavenumber"""
    H: float
    """Hurst exponent"""

    def mapto_isotropic_psd(self, q: components_t):
        """
        Get the isotropic power spectral density (psd) of a given wavenumber
        - q: angular wavenumber, i.e. 2 pi over wavelength
        """
        magnitude = la.norm(q, ord=2, axis=0, keepdims=True)

        # Find three regimes
        constant = magnitude < self.qR
        self_affine = (magnitude >= self.qR) & (magnitude < self.qS)
        omitted = magnitude >= self.qS

        # Evaluate accordingly
        psd = np.full_like(magnitude, np.nan)
        psd[constant] = self.C0 * self.qR ** (2 - 2 * self.H)
        psd[self_affine] = self.C0 * magnitude[self_affine] ** (-2 - 2 * self.H)
        psd[omitted] = 0

        # Return both in convenience of plotting
        return magnitude, psd


def psd_to_height(psd: components_t, rng=None, seed=None):
    # <h^2> corresponding to <PSD>, thus, take the square-root to match overall amplitude
    h_norm = np.sqrt(psd)

    # impose some random phase angle following uniform distribution
    if rng is None:
        rng = random.default_rng(seed)
    phase_angle = np.exp(1j * rng.uniform(0, 2 * np.pi, psd.shape))

    # drop the imaginary part as that should be close to zeros
    return fft.ifftn(h_norm * phase_angle).real


class CapillaryBridge:

    def __init__(self, theta: float, eta: float, gap: components_t):
        """
        - theta: contact angle
        - eta: interface thickness
        - gap: gap between two rigid bodies
        """
        self.curv = 0.5 * (abs(np.sin(theta)) + np.asin(np.cos(theta)) / np.cos(theta))
        self.gamma = -np.cos(theta)
        self.eta = eta
        self.gap = gap

        # According to Modica-Mortola's theorem
        self.perimeter_prefactor = 3.0

    def compute_perimeter(self, phase: components_t, phase_grad: components_t) -> components_t:
        """
        - phase: phase-field
        - phase_grad: gradient of phase-field
        """
        return self.perimeter_prefactor * (
            (1 / self.eta) * self.double_well_penalty(phase) + self.eta * self.square_penalty(phase_grad)
        )

    def compute_energy(self, phase: components_t, phase_grad: components_t) -> components_t:
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
        return np.sum(x**2 * (1 - x) ** 2, axis=0, keepdims=True)

    @staticmethod
    def square_penalty(x):
        return np.sum(x**2, axis=0, keepdims=True)

    # def compute_force(self, phase: components_t, phase_grad: components_t) -> components_t:
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

    def compute_energy_jacobian(
        self, phase: components_t, phase_grad: components_t
    ) -> tuple[components_t, components_t]:
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

    def compute_volume(self, phase: components_t) -> components_t:
        return phase * self.gap

    def compute_volume_jacobian(self, phase: components_t) -> tuple[components_t]:
        return self.gap
