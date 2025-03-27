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
        communicator = grid.coordinator.communicator

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
        (2 * np.pi) * fft.fftfreq(n, d=grid.spacing) for n in grid.nb_pixels
    )
    # Wave number grids
    wave_numbers_in_each_dim = np.meshgrid(*wave_numbers_in_each_dim)
    # Stack them into a single array
    return np.stack(wave_numbers_in_each_dim, axis=0)


def generate_height_profile(grid: Grid, roughness: SelfAffineRoughness, rng):
    # Get the power spectrum
    wave_numbers = generate_wave_numbers(grid)
    [_, psd] = roughness.isotropic_spectrum(wave_numbers)

    # <h^2> = <C^2>, take the square root
    rms_height = np.sqrt(psd)

    # impose some random phase angle
    phase_angle = np.exp(1j * rng.uniform(0, 2 * np.pi, np.shape(psd)))

    # FIXME: use muFFT to have it parallelized
    height = fft.ifftn(rms_height * phase_angle).real
    return height[(..., *grid.section.global_coords)]


def random_initial_guess(grid: Grid, rng):
    return rng.random(tuple(grid.section.nb_pixels))


@dc.dataclass(init=True)
class CapillaryBridge:
    """The simulator."""

    grid: Grid
    contact: SolidSolidContact
    height_lower_solid: CubicSpline
    height_upper_solid: CubicSpline
    vapour_liquid: CapillaryVapourLiquid
    phase_field: LinearFiniteElement
    quadrature: Quadrature
    solver: AugmentedLagrangian

    def __post_init__(self):
        collection = self.grid.section.pixel_collection
        self.rolling_height_field = collection.real_field("rolling_height", 1, "pixel")
        self.pixels_rolled = np.zeros(2)

        self.phase_field.setup_operators(self.quadrature)

        self.original_shape = (1, *self.grid.section.nb_pixels)

    def relocate_solid_plane(self, displacement: np.ndarray):
        # Get components
        [x, y, z] = displacement

        # The normal part, simply set the quantity
        self.contact.mean_plane_separation = z

        # The tangential part, split into integer and fractional parts. The integer part is the
        # handled by rolling, the fractional part becomes coordinate deviation in interpolation.
        frac_shear = np.empty(2)
        for axis, value in enumerate([x, y]):
            # Normaliz such that grid_spacing == 1.0
            [frac, nb_pixels] = math.modf(value / self.grid.spacing)
            if nb_pixels >= 1.0:
                self.region.roll(
                    self.rolling_height_field,
                    int(nb_pixels - self.pixels_rolled[axis]),
                    axis=axis,
                )
                self.pixels_rolled[axis] = nb_pixels
            frac_shear[axis] = frac
        self.height_upper_solid.sample(self.rolling_height_field.p)
        h1 = self.height_upper_solid.interpolate(self.quadrature.quad_pt_local_coords - frac_shear)
        self.vapour_liquid.heterogeneous_height = self.contact.gap_height(h1)

    def compute_energy(self, phase: np.ndarray):
        self.phase_field.sample(phase)
        [phi_interp, phi_grad] = self.phase_field.apply_operators(
            [self.phase_field.interpolation, self.phase_field.gradient]
        )
        return self.quadrature.integrate2D(self.vapour_liquid.energy_density(phi_interp, phi_grad))

    def compute_volume(self, phase: np.ndarray):
        self.phase_field.sample(phase)
        [phi_interp, phi_grad] = self.phase_field.apply_operators(
            [self.phase_field.interpolation, self.phase_field.gradient]
        )
        return self.quadrature.integrate2D(self.vapour_liquid.energy_density(phi_interp, phi_grad))

    def formulate_with_constant_volume(
        self,
        liquid_volume: float,
    ):
        def f(x: np.ndarray) -> float:
            phi = np.pad(x, 1, mode="constant", constant_values=-1)
            return self.compute_energy(phi)

        self.solver.f = f

        def g(x: np.ndarray) -> float:
            phi = np.pad(x, 1, mode="constant", constant_values=-1)
            return self.compute_volume(phi) - liquid_volume

        self.solver.g = g

        def l(x: np.ndarray, lam: float, c: float) -> float:
            print(f"In l(x), x has datatype {x.dtype}")
            phi = np.pad(x, 1, mode="constant", constant_values=-1)
            self.phase_field.sample(phi)
            [phi_interp, phi_grad] = self.phase_field.apply_operators(
                [self.phase_field.interpolation, self.phase_field.gradient]
            )
            f = self.quadrature.integrate2D(self.vapour_liquid.energy_density(phi_interp, phi_grad))
            g = (
                self.quadrature.integrate2D(self.vapour_liquid.liquid_height(phi_interp))
                - liquid_volume
            )
            val = f + lam * g + 0.5 * c * g**2
            print(f"In l(x), l has value {val}\n")
            return val

        self.solver.l = l

        def dx_l(x: np.ndarray, lam: float, c: float) -> np.ndarray:
            print(f"In dl(x), x has datatype {x.dtype}")
            phi = np.pad(x, 1, mode="constant", constant_values=-1)
            self.phase_field.sample(phi)
            [phi_interp, phi_grad] = self.phase_field.apply_operators(
                [self.phase_field.interpolation, self.phase_field.gradient]
            )
            dx_f = self.quadrature.pixel_area * self.phase_field.apply_transposed_to_values(
                self.vapour_liquid.energy_density_sensitivity(phi_interp, phi_grad),
                [self.phase_field.interpolation, self.phase_field.gradient],
            )
            g = (
                self.quadrature.integrate2D(self.vapour_liquid.liquid_height(phi_interp))
                - liquid_volume
            )
            dx_g = self.quadrature.pixel_area * self.phase_field.apply_transposed_to_values(
                self.vapour_liquid.liquid_height_sensitivity(phi_interp),
                [self.phase_field.interpolation],
            )
            val = (dx_f + (lam + c * g) * dx_g)[self.grid.coordinator.non_ghost]
            print(f"In dx_l(x), dx_l has shape {val.shape}, has values\n {val}\n")
            return val

        self.solver.dx_l = dx_l

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
        x = init_guess[self.grid.coordinator.non_ghost]

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
