"""This file addresses the "orchestration" of simulation

- Setups that are neither physics nor numerics
- Formulating optimization problem from model functions
- Post-processing
- RNG (Random number generator) seeding
"""

import dataclasses as dc
import math

import numpy as np
import numpy.random as random

from a_package.modelling import CapillaryBridge
from a_package.computing import communicator
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


def simulate_quasi_static_pull_push(
    store: FilesToReadWrite,
    capi: CapillaryBridge,
    solver: AugmentedLagrangian,
    V: float,
    d_min: float,
    d_max: float,
    d_step: float,
):
    # save the configurations
    store.save("Simulation", "modelling", capi)
    store.save("Simulation", "solving", solver)
    sim = SimulationResult("modelling.json", "solving.json", [])

    # generate all `d` (mean distances) values
    num_d = math.floor((d_max - d_min) / d_step) + 1
    d_departing = d_min + d_step * np.arange(num_d)
    d_approaching = d_max - d_step * np.arange(num_d)
    all_d = np.concatenate((d_approaching, d_departing[1:]))

    # Truncate to remove floating point errors
    # NOTE: assume 'd_step' < 0
    n_decimals = 6
    all_d = np.round(all_d, n_decimals)

    # inform
    print(
        f"Problem size: {capi.region.nx}x{capi.region.ny}. "
        f"Simulating for all {len(all_d)} mean distance values in...\n{all_d}"
    )

    # simulate
    x = capi.phi.ravel()
    for index, d in enumerate(all_d):
        # update the parameter
        print(f"" f"Parameter of interest: mean distance={d}")
        capi.z1 = d
        capi.update_gap()
        capi.update_phase_field()

        # solve the problem
        numopt = capi.formulate_with_constant_volume(V)
        x, t_exec, lam = solver.solve_minimisation(numopt, x)

        # save the results
        capi.phi = x.reshape(capi.region.nx, capi.region.ny)
        data = SimulationStep([0, 0], d, t_exec, capi.phi, lam)
        store.save("Simulation", f"steps---{index}", data)
        sim.steps.append(f"steps---{index}.json")

        # Check the bounds on phase field
        capi.validate_phase_field()

    # Save simulation results
    store.save("Simulation", "result", sim)

    # Load again to get all data (because they were saved part by part)
    sim = store.load("Simulation", "result", SimulationResult)
    # Post-process & save
    p_sim = post_process(sim)
    store.save("Processed", "result", p_sim)


def simulate_quasi_static_slide(
    store: FilesToReadWrite,
    capi: CapillaryBridge,
    solver: AugmentedLagrangian,
    V: float,
    slide_by_indices: list[tuple[int, int]],
):
    # save the configurations
    store.save("Simulation", "modelling", capi)
    store.save("Simulation", "solving", solver)
    sim = SimulationResult("modelling.json", "solving.json", [])

    # inform
    print(
        f"Problem size: {capi.region.nx}x{capi.region.ny}. "
        f"Simulating for all {len(slide_by_indices)} mean distance values in...\n{slide_by_indices}"
    )

    # simulate
    x = capi.phi.ravel()
    for index, m in enumerate(slide_by_indices):
        # update the parameter
        print(f"" f"Parameter of interest: slide by indices={m}")
        capi.ix1_iy1 = m
        capi.update_gap()

        # solve the problem
        numopt = capi.formulate_with_constant_volume(V)
        x, t_exec, lam = solver.solve_minimisation(numopt, x)

        # save the result
        capi.phi = x.reshape(capi.region.nx, capi.region.ny)
        data = SimulationStep(m, capi.z1, t_exec, capi.phi, lam)
        store.save("Simulation", f"steps---{index}", data)
        sim.steps.append(f"steps---{index}.json")

        # Check the bounds on phase field
        capi.validate_phase_field()

    # Save simulation results
    store.save("Simulation", "result", sim)

    # Load again to get all data (because they were saved part by part)
    sim = store.load("Simulation", "result", SimulationResult)
    # Post-process & save
    p_sim = post_process(sim)
    store.save("Processed", "result", p_sim)


def get_rng(seed: _t.Optional[int] = None):
    """Get the RNG with a given seed. The seed is broadcasted to all processes."""
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
