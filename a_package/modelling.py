"""This file addresses the physical perspectives.

- Roughness of the solid surface
- The "gap" formed between two solid surface with displacement
- Capillary bridge
"""

import dataclasses as dc

import numpy as np
import numpy.linalg as la
import numpy.fft as fft

from a_package.computing import Region, Bicubic, CentroidQuadrature


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

        return height[region.in_local_subdomain]


@dc.dataclass(init=True)
class CapillaryPlanes:
    """Deal with two solids and their displacement"""

    region: Region
    height_base: np.ndarray
    height_top: np.ndarray

    def __post_init__(self):
        # The solid at base doesn't move, get the height in local subdomain
        self.h_base = self.height_base[self.region.in_local_subdomain]

        # The solid at top moves, interpolate with bicubic spline
        self.interpolate_height_top = Bicubic(self.height_top, periodic=True)

        # Get the global coordinates of grid points in local subdomain, normalize to [0, 1]
        [self.x_grid, self.y_grid] = self.region.global_coords / self.region.nb_domain_grid_pts

        # Get the size of the whole domain
        [self.domain_size_x, self.domain_size_y] = (
            self.region.nb_domain_grid_pts * self.region.grid_spacing
        )

    def gap_height(self, displacement: np.ndarray):
        [x, y, z] = displacement
        h_top = self.interpolate_height_top(
            self.x_grid - x / self.domain_size_x, self.y_grid - y / self.domain_size_y
        )
        return np.clip(z + h_top[self.region.in_local_subdomain] - self.h_base, a_min=0)


@dc.dataclass(init=True)
class CapillaryBridge:

    region: Region
    interfacial_width: float
    surface_tension_ratio: float

    def __post_init__(self):
        self.quadrature = CentroidQuadrature("quadrature_for_capillary", self.region)
        self.gap_height_var = self.quadrature.discrete_variable("gap_height", 1)
        self.phase_field_var = self.quadrature.discrete_variable("phase_field", 1)
        self.gap_height_in_integrand = None

    @property
    def gap_height(self):
        return self.gap_height.s

    @gap_height.setter
    def gap_height(self, value: np.ndarray):
        self.gap_height_var.s = value
        self.region.update(self.gap_height_var.name)
        [height_in_integrand] = self.quadrature.apply_operators(
            self.gap_height_var, self.quadrature.op_interpolation
        )
        self.gap_height_in_integrand = height_in_integrand.s

    @property
    def solid_solid_contact(self):
        return np.nonzero(self.gap_height == 0)

    def update_phase_field(self, value: np.ndarray):
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
