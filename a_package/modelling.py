"""This file addresses the physical perspectives.

- Roughness of the solid surface
- The "gap" formed between two solid surface with displacement
- Capillary bridge
"""

import math
import dataclasses as dc

import numpy as np
import numpy.linalg as la
import numpy.fft as fft
from scipy.interpolate import RegularGridInterpolator

from a_package.computing import Region, CentroidQuadrature


# FIXME: an idea of more "separation". Remove all discretization calls (except Region?)
#
# - Roughness & Plates:
#   not aware that it computes global and then index the local. It should be some scheme that can
# affect a functions input & output.
#
# - CapillaryBridge:
#   - not aware that it starts local, and in the end, calls region to do something over global.
#   - not aware of the difference between discrete nodal values & quadratrue field. Then
# most of its call would accept [phi, d_phi] as parameters.
#
# - some imagined class that uses Fourier series.
#
# A common thing, it seems in the model it remains function relations. And in computing, it does
# something to the input and also possibly something to the output. And this "does something" may
# be implemented in "Simulation", where modelling, computing, solving are stitched together.


@dc.dataclass(init=True)
class SelfAffineRoughness:
    C0: float
    qR: float
    qS: float
    H: float

    def get_isotropic_spectrum(self, region: Region):
        # Wave numbers
        wave_numbers_in_each_dim = (
            (2 * np.pi) * fft.fftfreq(nb_pts, d=region.grid_spacing)
            for nb_pts in region.nb_domain_grid_pts
        )
        wave_numbers_in_each_dim = np.meshgrid(*wave_numbers_in_each_dim)
        wave_numbers = np.stack(wave_numbers_in_each_dim, axis=0)

        # Find three regimes
        magnitude = la.norm(wave_numbers, ord=2, axis=0)
        constant = magnitude < self.qR
        self_affine = (magnitude >= self.qR) & (magnitude < self.qS)
        omitted = magnitude >= self.qS

        # Evaluate accordingly
        psd = np.empty_like(magnitude)
        psd[constant] = self.C0 * self.qR ** (2 - 2 * self.H)
        psd[self_affine] = self.C0 * magnitude[self_affine] ** (-2 - 2 * self.H)
        psd[omitted] = 0

        # Return both for convenience of plotting
        return magnitude, psd

    def generate_height_profile(self, region: Region, rng):
        [_, psd] = self.get_isotropic_spectrum(region)

        # <h^2> = <C^2>, take the square root
        rms_height = np.sqrt(psd)

        # impose some angular variation on the magnitude
        # amplitude_variation = rng.normal()

        # impose some random phase angle
        phase_angle = np.exp(1j * rng.uniform(0, 2 * np.pi, np.shape(psd)))

        # Generate the global height, and slice to get the subdomain
        # FIXME: use muFFT when the parallel is updated
        height = fft.ifftn(rms_height * phase_angle).real

        return height


@dc.dataclass(init=True)
class RoughPlane:
    """Solid plane with a rough surface specified by its height profile."""

    region: Region
    height_original: np.ndarray
    displacement: np.ndarray = None

    def __post_init__(self):
        if self.displacement is None:
            self.displacement = np.zeros(3)

        # Save indices for the local subdomain.
        self.indices = self.region.collection.uint_field(
            "indices_top", self.region.spatial_dims, "pixel"
        )
        self.indices.p = self.region.global_coords

        # Interpolate the original height with periodic boundary
        pad_left = 1
        pad_right = 1
        axis_pt_locs = tuple(
            # Extend the range on both sides due to periodic boundary
            np.arange(-pad_left, nb_pts + pad_right)
            for nb_pts in self.region.nb_domain_grid_pts
        )
        self.interpolate_height = RegularGridInterpolator(
            axis_pt_locs,
            # Pad at both sides due to periodic boundary
            np.pad(self.height_original, (pad_left, pad_right), mode="wrap"),
            method="cubic",
            # Then there shall be no out-of-boundary pixels
            bounds_error=True,
        )

        # To record the shear movement in unit of pixels
        self.nb_sheared_pixels = np.zeros(2)

    @property
    def height(self):
        # Get components
        [x, y, z] = self.displacement

        # The tangential part, split into integer and fractional parts. The integer part is the
        # handled by rolling, the fractional part is handled by interpolation.
        frac_shear = np.zeros(2)
        for axis, value in enumerate([x, y]):
            # Normaliz such that grid_spacing == 1.0
            frac, nb_pixels = math.modf(value / self.region.grid_spacing)
            if nb_pixels >= 1.0:
                self.region.roll(
                    self.indices, int(nb_pixels - self.nb_sheared_pixels[axis]), axis=axis
                )
                self.nb_sheared_pixels[axis] = int(nb_pixels)
            frac_shear[axis] = frac

        # The normal part is simply added
        return z + self.interpolate_height(
            np.moveaxis(self.indices.p + frac_shear[..., np.newaxis, np.newaxis], 0, -1)
        )


@dc.dataclass(init=True)
class SolidSolidContact:
    """Contact between two rough solid planes."""

    height_base: np.ndarray
    height_top: np.ndarray

    @property
    def gap_height(self):
        # Assume interpenaltration at solid-solid contact
        return np.clip(self.height_top - self.height_base, a_min=0, a_max=None)


@dc.dataclass(init=True)
class CapillaryBridge:

    region: Region
    interfacial_width: float
    surface_tension_ratio: float

    def __post_init__(self):
        self.quadrature = CentroidQuadrature("quadrature_for_capillary", self.region)
        self.gap_height_var = self.quadrature.discrete_variable("gap_height", 1)
        self.phase_field_var = self.quadrature.discrete_variable("phase_field", 1)

    @property
    def gap_height(self):
        return self.gap_height.s

    @gap_height.setter
    def gap_height(self, value):
        self.gap_height_var.s = value
        self.region.update(self.gap_height_var.name)
        [height_in_integrand] = self.quadrature.apply_operators(
            self.gap_height_var, self.quadrature.op_interpolation
        )
        self.gap_height_in_integrand = height_in_integrand.s

    @property
    def solid_solid_contact(self):
        return np.nonzero(self.gap_height == 0)

    @property
    def phase_field(self):
        return self.phase_field_var.s

    @phase_field.setter
    def phase_field(self, value):
        value[self.solid_solid_contact] = 0
        self.phase_field_var.s = value
        self.region.update(self.phase_field_var.name)

    @property
    def energy(self):
        [phi, d_phi] = self.quadrature.apply_operators(
            self.phase_field_var,
            [self.quadrature.op_interpolation, self.quadrature.op_gradient],
        )
        energy_density = self.quadrature.integrand_field("energy_density", 1)

        area_water_vapour = self.gap_height_in_integrand * (
            # double well penalty on phi
            (1 / self.interface_width) * self.double_well_penalty(phi.s)
            # square penalty on d_phi
            + self.interface_width * self.square_penalty(d_phi.s)
        )

        # FIXME: add the slope contribution.
        area_water_solid = 2 * phi.s

        energy_density.s = area_water_vapour - self.surface_tension_ratio * area_water_solid
        return self.quadrature.field_integral(energy_density)

    @staticmethod
    def double_well_penalty(x):
        return 9 * x**2 * (1 - x) ** 2

    @staticmethod
    def square_penalty(x):
        return sum(x_i**2 for x_i in x)

    @property
    def energy_sensitivity(self):
        [phi, d_phi] = self.quadrature.apply_operators(
            self.phase_field_var,
            [self.quadrature.op_interpolation, self.quadrature.op_gradient],
        )

        energy_density_sens_phi = self.quadrature.integrand_field("energy_density_sens_phi", 1)
        area_water_vapour_sens_phi = self.gap_height_in_integrand * (
            # derivative of double well penalty w.r.t. phi
            (1 / self.interface_width)
            * self.double_well_penalty_derivative(phi.s)
        )
        # FIXME: add the slope contribution.
        area_water_solid_sens_phi = 2
        energy_density_sens_phi.s = (
            area_water_vapour_sens_phi - self.surface_tension_ratio * area_water_solid_sens_phi
        )

        energy_density_sens_d_phi = self.quadrature.integrand_field("energy_density_sens_d_phi", 2)
        energy_density_sens_d_phi.s = self.gap_height_in_integrand * (
            # derivative of square penalty w.r.t. d_phi
            self.interface_width
            * self.square_penalty_derivatie(d_phi.s)
        )

        return self.quadrature.field_sensitivity(
            [energy_density_sens_phi, energy_density_sens_d_phi],
            [self.quadrature.op_interpolation, self.quadrature.op_gradient],
        )

    @staticmethod
    def double_well_penalty_derivative(x):
        return 18 * x * (1 - x) * (1 - 2 * x)

    @staticmethod
    def square_penalty_derivatie(x):
        return 2 * x

    @property
    def volume(self):
        [phi] = self.quadrature.apply_operators(
            self.phase_field_var, [self.quadrature.op_interpolation]
        )
        bridge_height = self.quadrature.integrand_field("bridge_height", 1)
        bridge_height.s = self.gap_height_in_integrand * phi.s
        return self.quadrature.field_integral(bridge_height)

    @property
    def volume_sensitivity(self):
        bridge_height_sens_phi = self.quadrature.integrand_field("bridge_height_sens_phi", 1)
        bridge_height_sens_phi.s = self.gap_height_in_integrand
        return self.quadrature.field_sensitivity(
            [bridge_height_sens_phi], [self.quadrature.op_interpolation]
        )

    @property
    def adhesive_force(self):
        [phi, d_phi] = self.quadrature.apply_operators(
            self.phase_field_var,
            [self.quadrature.op_interpolation, self.quadrature.op_gradient],
        )
        interface_perimeter = self.quadrature.integrand_field("interface_perimeter", 1)
        interface_perimeter.s = (
            # double well penalty on phi
            (1 / self.interface_width) * self.double_well_penalty(phi.s)
            # square penalty on d_phi
            + self.interface_width * self.square_penalty(d_phi.s)
        )
        return self.quadrature.field_integral(interface_perimeter)
