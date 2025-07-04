"""This file addresses the "orchestration" of simulation

- Stitching together different pieces to model, compute, solve, and localize; especially
- Post-processing
- RNG (Random number generator) seeding
"""

import dataclasses as dc
import math
import typing as _t

import numpy as np
import numpy.fft as fft
import numpy.random as random

from a_package.modelling import *
from a_package.computing import *
from a_package.solving import *
from a_package.storing import FilesToReadWrite


def get_rng(grid: Grid, seed: _t.Optional[int]):
    """Get the RNG with a given seed. When no seed is given, it generates a random seed at rank 0
    and broadcasts it.
    """
    if seed is None:
        communicator = grid.get_world_communicator()

        # Generate a random seed at the root process
        if communicator.rank == 0:
            seed = random.SeedSequence().entropy
            print(f"Seed is generated: {seed}")

        # Seed is a 128-bit integer. To broadcast it, we need to save into multiple integers.
        bit_len_seed = 128
        bit_len_buffer = 16
        buffer = np.zeros(bit_len_seed // bit_len_buffer, dtype=np.uint16)

        # Save the seed into buffer
        if communicator.rank == 0:
            for i in range(len(buffer)):
                buffer[i] = seed % (1 << bit_len_buffer)
                seed >>= bit_len_buffer

        # Broadcast the seed to every process
        for i in range(len(buffer)):
            buffer[i] = communicator.bcast(buffer[i], root=0)

        # Recover the seed from buffer
        seed = 0
        for i in range(len(buffer)):
            # NOTE: One must recover Python integer from NumPy array to avoid implicit conversion.
            seed += buffer[i].item() << (i * bit_len_buffer)

        # Debug information
        if communicator.rank == 0:
            print(f"Seed is received: {seed}")

    # Every process usees the same RNG.
    return random.default_rng(seed)


def generate_wave_numbers(grid: Grid):
    # Wave number axes
    wave_numbers_in_each_dim = (
        (2 * np.pi) * fft.fftfreq(n, d=l) for n, l in zip(grid.nb_pixels, grid.pixel_length)
    )
    # Wave number grids
    wave_numbers_in_each_dim = np.meshgrid(*wave_numbers_in_each_dim)
    # Stack them into a single array
    return np.stack(wave_numbers_in_each_dim, axis=0)


def generate_height_profile(grid: Grid, roughness: SelfAffineRoughness, rng):
    # Get the power spectrum
    wave_numbers = generate_wave_numbers(grid)
    [_, psd] = roughness.isotropic_spectrum(wave_numbers)

    # psd:=h^2(q), <h^2(q)> = <h^2(r)>
    # psd:=h^2(q), <h^2(q)> = <h^2(r)>
    rms_height = np.sqrt(psd)

    # impose some random phase angle
    phase_angle = np.exp(1j * rng.uniform(0, 2 * np.pi, np.shape(psd)))

    # FIXME: parallelise it with muFFT
    height = fft.ifftn(rms_height * phase_angle).real
    # Get the local section
    return height[(np.newaxis, ..., *grid.pixel_indices_in_section)]


def random_initial_guess(grid: Grid, rng):
    return rng.random([1, 1, *tuple(grid.nb_pixels_in_section)])


@dc.dataclass(init=True)
class CapillaryBridge:
    """The simulator.

    Naming convention for the data attributes related to field data:
    - Ending with "_field" means a Field object, otherwise the raw value (NumPy array)
    - "aaa_D_bbb" means the derivative of aaa w.r.t. bbb
    - Starting with "evaluate_" means a non-parameter function
    """

    def __init__(
        self,
        grid: Grid,
        solid_solid: SolidSolidContact,
        vapour_liquid: CapillaryVapourLiquid,
        solver: AugmentedLagrangian,
        lower_height_nodal: np.ndarray,
        upper_height_nodal: np.ndarray,
        phase_nodal: np.ndarray,
        quadrature: Quadrature = centroid_quadrature,
    ):
        self.grid = grid
        self.solid_solid = solid_solid
        self.vapour_liquid = vapour_liquid
        self.solver = solver

        # Register the sub-point scheme of quadrature
        self.quadrature = quadrature
        self.grid.add_sub_pt_scheme(self.quadrature.tag, self.quadrature.nb_quad_pts)

        # The field to save height at nodal points
        self.lower_height_nodal_field = self.grid.real_field("lower_height_nodal", 1, "pixel")
        self.lower_height_nodal_field.data = lower_height_nodal
        self.upper_height_nodal_field = self.grid.real_field("upper_height_nodal", 1, "pixel")
        self.upper_height_nodal_field.data = upper_height_nodal

        # get the height at quad pionts
        spline = CubicSpline(self.grid)
        spline.sample(self.lower_height_nodal_field)
        self.solid_solid.lower_height = spline.interpolate(self.quadrature.quad_pt_offset)
        spline.sample(self.upper_height_nodal_field)
        self.solid_solid.upper_height = spline.interpolate(self.quadrature.quad_pt_offset)

        # The field to save phase at nodal points
        self.phase_nodal_field = self.grid.real_field("phase_nodal", 1, "pixel")
        self.phase_nodal_field.data = phase_nodal
        self.original_shape = np.shape(phase_nodal)

        # The "Convolution Operator" will be used
        fem = Linear2DFiniteElementInPixel()
        value_op = fem.create_field_value_approximation(self.quadrature.quad_pt_offset)
        gradient_op = fem.create_field_gradient_approximation(
            self.quadrature.quad_pt_offset, *grid.pixel_length
        )

        # Construct various fields and methods for phase field computation
        phase_quad_field = self.grid.real_field("phase_quad", 1, self.quadrature.tag)
        self.evaluate_phase_quad = self.phase_nodal_field.bind_mapping(value_op, phase_quad_field)

        phase_gradient_quad_field = self.grid.real_field(
            "phase_gradient_quad", self.grid.nb_dims, self.quadrature.tag
        )
        self.evaluate_phase_gradient_quad = self.phase_nodal_field.bind_mapping(
            gradient_op, phase_gradient_quad_field
        )

        self.energy_D_phase_quad_field = self.grid.real_field(
            "energy_D_phase_quad", 1, self.quadrature.tag
        )
        energy_D_phase_quad_D_phase_nodal_field = self.grid.real_field(
            "energy_D_phase_quad_D_phase_nodal", 1, "pixel"
        )
        self.evaluate_energy_D_phase_quad_D_phase_nodal = (
            self.energy_D_phase_quad_field.bind_mapping_sensitivity(
                value_op, energy_D_phase_quad_D_phase_nodal_field
            )
        )

        self.energy_D_phase_gradient_quad_field = self.grid.real_field(
            "energy_D_phase_gradient_quad", self.grid.nb_dims, self.quadrature.tag
        )
        energy_D_phase_gradient_quad_D_phase_nodal_field = self.grid.real_field(
            "energy_D_phase_quad_D_phase_nodal", 1, "pixel"
        )
        self.evaluate_energy_D_phase_gradient_quad_D_phase_nodal = (
            self.energy_D_phase_gradient_quad_field.bind_mapping_sensitivity(
                gradient_op, energy_D_phase_gradient_quad_D_phase_nodal_field
            )
        )

        self.volume_D_phase_quad_field = self.grid.real_field(
            "volume_D_phase_quad", 1, self.quadrature.tag
        )
        volume_D_phase_quad_D_phase_nodal_field = self.grid.real_field(
            "volume_D_phase_quad_D_phase_nodal", 1, "pixel"
        )
        self.evaluate_volume_D_phase_quad_D_phase_nodal = (
            self.volume_D_phase_quad_field.bind_mapping_sensitivity(
                value_op, volume_D_phase_quad_D_phase_nodal_field
            )
        )

        # FIXME: roll the upper height for sliding simulation
        # self.rolling_height_field = self.grid.real_field("rolling_height", 1, "pixel")
        # self.nb_rolled_pixels = np.zeros(self.grid.nb_dims)

    def relocate_solid_plane(self, displacement: np.ndarray):
        # Get components
        [x, y, z] = displacement

        # The normal part, simply set the quantity
        self.solid_solid.mean_plane_separation = z

        # FIXME: roll the upper height for sliding simulation
        # # The tangential part, split into integer and fractional parts. The integer part is the
        # # handled by rolling, the fractional part becomes coordinate deviation in interpolation.
        # frac_shear = np.empty(2)
        # for axis, value in enumerate([x, y]):
        #     # Normaliz such that grid_spacing == 1.0
        #     [frac, nb_pixels] = math.modf(value / self.grid.pixel_length[axis])
        #     if nb_pixels >= 1.0:
        #         self.region.roll(
        #             self.rolling_height_field,
        #             int(nb_pixels - self.nb_rolled_pixels[axis]),
        #             axis=axis,
        #         )
        #         self.nb_rolled_pixels[axis] = nb_pixels
        #     frac_shear[axis] = frac
        # spline = CubicSpline(self.grid)
        # spline.sample(self.upper_height_nodal_field)
        # self.solid_solid.upper_height = spline.interpolate(self.quadrature.quad_pt_offset)

        # Update the gap height
        self.vapour_liquid.heterogeneous_height = self.solid_solid.gap_height()

    def compute_energy(self, nodal_values: np.ndarray):
        # update nodal values
        self.phase_nodal_field.data = nodal_values
        # interpolate values at quadrature points
        phase_quad = self.evaluate_phase_quad()
        phase_gradient_quad = self.evaluate_phase_gradient_quad()
        # Evaluate the integrand and the integral
        return self.quadrature.integrate(self.vapour_liquid.energy_density(phase_quad, phase_gradient_quad), self.grid)

    def compute_volume(self, nodal_values: np.ndarray):
        # update nodal values
        self.phase_nodal_field.data = nodal_values
        # interpolate values at quadrature points
        phi_quad = self.evaluate_phase_quad()
        # Evaluate the integrand and the integral
        return self.quadrature.integrate(self.vapour_liquid.liquid_height(phi_quad), self.grid)

    def compute_energy_jacobian(self, nodal_values: np.ndarray):
        # update nodal values
        self.phase_nodal_field.data = nodal_values
        # interpolate values at quadrature points
        phase_quad = self.evaluate_phase_quad()
        phase_gradient_quad = self.evaluate_phase_gradient_quad()
        # Get the model functions derivative w.r.t. the interpolated values
        [self.energy_D_phase_quad_field.data, self.energy_D_phase_gradient_quad_field.data] = (
            self.vapour_liquid.energy_density_sensitivity(phase_quad, phase_gradient_quad)
        )
        # Multiply the last results with the derivative of interpolation w.r.t. the nodal values
        energy_D_phase_quad_D_phase_nodal = self.evaluate_energy_D_phase_quad_D_phase_nodal()
        energy_D_phase_gradient_quad_D_phase_nodal = (
            self.evaluate_energy_D_phase_gradient_quad_D_phase_nodal()
        )
        # Sum up all contributing terms as in a total derivative
        return 0.5*self.grid.pixel_area * (
            energy_D_phase_quad_D_phase_nodal + energy_D_phase_gradient_quad_D_phase_nodal
        )

    def compute_volume_jacobian(self, nodal_values: np.ndarray):
        # update nodal values
        self.phase_nodal_field.data = nodal_values
        # interpolate values at quadrature points
        phase_quad = self.evaluate_phase_quad()
        # Get the model functions derivative w.r.t. the interpolated values
        [self.volume_D_phase_quad_field.data] = self.vapour_liquid.liquid_height_sensitivity(
            phase_quad
        )
        # Multiply the last results with the derivative of interpolation w.r.t. the nodal values
        volume_D_phase_quad_D_phase_nodal = self.evaluate_volume_D_phase_quad_D_phase_nodal()
        # Sum up all contributing terms as in a total derivative
        # FIXME: try to move this triangle area thing to where integral happens
        return 0.5*self.grid.pixel_area * volume_D_phase_quad_D_phase_nodal

    def formulate_with_constant_volume(
        self,
        liquid_volume: float,
    ):
        def f(x: np.ndarray) -> float:
            x = x.reshape(self.original_shape)
            return self.compute_energy(x)

        self.solver.f = f
        # self.solver.f = lambda x: self.compute_energy(x)

        def g(x: np.ndarray) -> float:
            x = x.reshape(self.original_shape)
            return self.compute_volume(x) - liquid_volume

        self.solver.g = g
        # self.solver.g = lambda x: self.compute_volume(x) - liquid_volume

        def l(x: np.ndarray, lam: float, c: float) -> float:
            # print(f"In l(x), x has datatype {x.dtype}")
            # phi = np.pad(x, 1, mode="constant", constant_values=-1)
            # self.phase_nodal_field.sample(np.expand_dims(phi, axis=0))
            # [phi_interp, phi_grad] = self.phase_nodal_field.apply_operators(
            #     [self.phase_nodal_field.interpolation, self.phase_nodal_field.gradient]
            # )
            # f = self.quadrature.integrate(self.liquid_vapour.energy_density(phi_interp, phi_grad))
            # g = (
            #     self.quadrature.integrate(self.liquid_vapour.liquid_height(phi_interp))
            #     - liquid_volume
            # )
            # val = f + lam * g + 0.5 * c * g**2
            # print(f"In l(x), l has value {val}\n")
            # return val
            # FIXME: avoid updating field two times
            x = x.reshape(self.original_shape)
            f = self.compute_energy(x)
            g = self.compute_volume(x)
            return f + lam * g + 0.5 * c * g**2

        self.solver.l = l

        def l_D_x(x: np.ndarray, lam: float, c: float) -> np.ndarray:
            # print(f"In dl(x), x has datatype {x.dtype}")
            # phi = np.pad(x, 1, mode="constant", constant_values=-1)
            # self.phase_nodal_field.sample(np.expand_dims(phi, axis=0))
            # [phi_interp, phi_grad] = self.phase_nodal_field.apply_operators(
            #     [self.phase_nodal_field.interpolation, self.phase_nodal_field.gradient]
            # )
            # dx_f = self.quadrature.pixel_area * self.phase_nodal_field.apply_transposed_to_values(
            #     self.liquid_vapour.energy_density_sensitivity(phi_interp, phi_grad),
            #     [self.phase_nodal_field.interpolation, self.phase_nodal_field.gradient],
            # )
            # g = (
            #     self.quadrature.integrate(self.liquid_vapour.liquid_height(phi_interp))
            #     - liquid_volume
            # )
            # dx_g = self.quadrature.pixel_area * self.phase_nodal_field.apply_transposed_to_values(
            #     self.liquid_vapour.liquid_height_sensitivity(phi_interp),
            #     [self.phase_nodal_field.interpolation],
            # )
            # val = (dx_f + (lam + c * g) * dx_g)[self.grid.coordinator.non_ghost]
            # print(f"In dx_l(x), dx_l has shape {val.shape}, has values\n {val}\n")
            # return val
            # FIXME: avoid updating field three times
            x = x.reshape(self.original_shape)
            f_D_x = self.compute_energy_jacobian(x)
            g = self.compute_volume(x)
            g_D_x = self.compute_volume_jacobian(x)
            return f_D_x + (lam + c * g) * g_D_x

        self.solver.dx_l = l_D_x

    def simulate_with_trajectory(
        self,
        trajectory: np.ndarray,
        init_guess: np.ndarray,
        store: FilesToReadWrite,
    ):
        """
        trajectory: 2D array, with first axis size = #steps, last axis size = 3 (spatial dims).
        """
        # inform
        print(f"Simulating for all {len(trajectory)} displacements.")

        # Only non-ghost pixels are decision variables
        x = init_guess

        # Simulation
        steps = []
        for index in range(np.size(trajectory, axis=0)):
            # update the parameter
            print(f"Displacement={trajectory[index]}")

            # solve the problem
            self.relocate_solid_plane(trajectory[index])
            [x, t_exec, lam] = self.solver.find_minimizer(x)

            # save the results
            store.save("Simulation", f"steps---{index}", x)
            steps.append(f"steps---{index}.json")

            # Check the bounds on phase field
            self.validate_phase_field(x)

        # Save simulation results
        store.save("Simulation", "result", steps)

    def validate_phase_field(values: np.ndarray):
        """Check the bounds on the phase field. Which is not enforced in optimization process."""

        # phase field < 0
        if np.any(values < 0):
            outlier = np.where(values < 0, values, np.nan)
            count = np.count_nonzero(~np.isnan(outlier))
            extreme = np.nanmin(outlier)
            print(f"Notice: phase field has {count} values < 0, min at {extreme:.2e}")

        # phase field > 1
        if np.any(values > 1):
            outlier = np.where(values > 1, values, np.nan)
            count = np.count_nonzero(~np.isnan(outlier))
            extreme = np.nanmax(outlier)
            print(f"Notice: phase field has {count} values > 1, max at 1.0+{extreme - 1:.2e}.")


@dc.dataclass
class SimulationStep:
    m: tuple[int, int]
    d: float
    t_exec: float
    phi: np.ndarray
    lam: np.ndarray


@dc.dataclass
class SimulationResult:
    modelling: CapillaryVapourLiquid
    solving: AugmentedLagrangian
    steps: list[SimulationStep]


def post_process(res: SimulationResult):
    # allocate memory
    n_step = len(res.steps)
    n_dimension = 3
    t = np.empty(n_step)
    g = []
    phi = []
    r = np.empty((n_step, n_dimension))
    F = np.empty((n_step, n_dimension))
    E = np.empty((n_step))
    p = np.empty((n_step))
    V = np.empty((n_step))

    # use the model for computing extra quantities
    capi = res.modelling

    # Convert data "rows" to "columns"
    for i, step in enumerate(res.steps):
        t[i] = step.t_exec

        capi.ix1_iy1 = step.m
        capi.z1 = step.d
        capi.update_gap()
        g.append(capi.g)

        capi.phi = step.phi
        capi.update_phase_field()
        phi.append(capi.phi)

        r[i] = capi.displacement
        F[i] = capi.force

        E[i] = capi.energy_density
        p[i] = step.lam
        V[i] = capi.liquid_height

    # pack in an object
    evo = Evolution(t, g, phi, r, F, E, p, V)
    return ProcessedResult(res.modelling, res.solving, evo)


@dc.dataclass
class Evolution:
    t_exec: np.ndarray
    g: list[np.ndarray]
    phi: list[np.ndarray]
    r: np.ndarray
    F: np.ndarray
    E: np.ndarray
    p: np.ndarray
    V: np.ndarray


@dc.dataclass
class ProcessedResult:
    modelling: CapillaryVapourLiquid
    solving: AugmentedLagrangian
    evolution: Evolution
