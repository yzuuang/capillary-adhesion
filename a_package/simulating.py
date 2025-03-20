"""This file addresses the "orchestration" of simulation

- Setups that are neither physics nor numerics
- Formulating optimization problem from model functions
- Post-processing
- RNG (Random number generator) seeding
"""

import dataclasses as dc
import math
import typing as _t

import numpy as np
import numpy.random as random

from a_package.modelling import *
from a_package.computing import Region, communicator
from a_package.solving import AugmentedLagrangian
from a_package.storing import FilesToReadWrite


@dc.dataclass
class SimulationStep:
    m: tuple[int, int]
    d: float
    t_exec: float
    phi: np.ndarray
    lam: np.ndarray


@dc.dataclass
class SimulationResult:
    modelling: CapillaryBridge
    solving: AugmentedLagrangian
    steps: list[SimulationStep]


def formulate_with_constant_volume(
    region: Region,
    capillary: CapillaryBridge,
    water_volume: float,
    augm_lagr: AugmentedLagrangian,
):

    def f(x: np.ndarray) -> float:
        capillary.phase_field = x.reshape(region.nb_subdomain_grid_pts)
        return capillary.energy

    augm_lagr.f = f

    def g(x: np.ndarray) -> float:
        capillary.phase_field = x.reshape(region.nb_subdomain_grid_pts)
        return capillary.volume - water_volume

    augm_lagr.g = g

    def l(x: np.ndarray, lam: float, c: float) -> float:
        capillary.phase_field = x.reshape(region.nb_subdomain_grid_pts)
        return (
            capillary.energy
            + lam * (capillary.volume - water_volume)
            + 0.5 * c * (capillary.volume - water_volume) ** 2
        )

    augm_lagr.l = l

    def dx_l(x: np.ndarray, lam: float, c: float) -> np.ndarray:
        capillary.phase_field = x.reshape(region.nb_subdomain_grid_pts)
        return (
            capillary.energy_sensitivity
            + lam * capillary.volume_sensitivity
            + c * (capillary.volume - water_volume) * capillary.volume_sensitivity
        )

    augm_lagr.dx_l = dx_l


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


def simulate_with_trajectory(
    store: FilesToReadWrite,
    region: Region,
    solid_planes: tuple[RoughPlane, RoughPlane],
    solid_contact: SolidSolidContact,
    capillary: CapillaryBridge,
    trajectory: np.ndarray,
    augm_lagr: AugmentedLagrangian,
    init_guess: np.ndarray,
):
    # save the configurations
    if communicator.rank == 0:
        store.save("Simulation", "solving", augm_lagr)

    # inform
    print(f"Simulating for all {len(trajectory)} ")

    # initial guess
    x = np.ravel(init_guess)

    # Simulation
    steps = []
    [_, plane_move] = solid_planes
    for index, displacement in enumerate(trajectory):
        # update the parameter
        print(f"" f"Parameter of interest: mean distance={trajectory[index]}")

        # solve the problem
        plane_move.displacement = displacement
        solid_contact.height_top = plane_move.height
        capillary.gap_height = solid_contact.gap_height
        [x, t_exec, lam] = augm_lagr.find_minimizer(x)

        # save the results
        phi = x.reshape(region.nb_subdomain_grid_pts)
        store.save("Simulation", f"steps---{index}", region.gather(phi))
        steps.append(f"steps---{index}.json")

        # Check the bounds on phase field
        validate_phase_field(phi)

    # Save simulation results
    store.save("Simulation", "result", steps)


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

        E[i] = capi.energy
        p[i] = step.lam
        V[i] = capi.volume

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
    modelling: CapillaryBridge
    solving: AugmentedLagrangian
    evolution: Evolution


def get_rng(seed: _t.Optional[int] = None):
    """Get the RNG with a given seed. When no seed is given, it generates a random seed at rank 0
    and broadcasts it.
    """
    if seed is None:
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
