"""
Simulation routines: modelling, solving and post-processing.
"""

import dataclasses as dc
import math

import numpy as np

from a_package.modelling import CapillaryBridge
from a_package.solving import AugmentedLagrangian
from a_package.storing import FilesToReadWrite


@dc.dataclass
class SimulationStep:
    m: tuple[int, int]
    d: float
    phi: np.ndarray
    t_exec: float


@dc.dataclass
class SimulationResult:
    modelling: CapillaryBridge
    solving: AugmentedLagrangian
    steps: list[SimulationStep]


def post_process(res: SimulationResult):
    # allocate memory
    num_steps = len(res.steps)
    t = np.empty(num_steps)
    phi = []
    m = []
    z = np.empty(num_steps)
    V = np.empty(num_steps)
    E = np.empty(num_steps)
    F = np.empty((num_steps, 3))

    # model for computing extra quantities
    capi = res.modelling

    # data "rows" to "columns"
    for i, step in enumerate(res.steps):
        t[i] = step.t_exec
        phi.append(step.phi)
        m.append(step.m)
        z[i] = step.d

        capi.phi = step.phi
        capi.update_displacement(m1=m[i], z1=z[i])
        V[i] = capi.compute_volume()
        E[i] = capi.compute_energy()
        F[i] = capi.compute_force()

    # pack in an object
    evo = Evolution(t, phi, m, z, V, E, F)
    return ProcessedResult(res.modelling, res.solving, evo)


@dc.dataclass
class Evolution:
    t_exec: np.ndarray
    phi: list[np.ndarray]
    m1: list[tuple[int, int]]
    z1: np.ndarray
    V: np.ndarray
    E: np.ndarray
    F: np.ndarray


@dc.dataclass
class ProcessedResult:
    modelling: CapillaryBridge
    solving: AugmentedLagrangian
    evolution: Evolution


def simulate_quasi_static_pull_push(store: FilesToReadWrite, capi: CapillaryBridge, solver: AugmentedLagrangian,
                                    V: float, d_min: float, d_max: float, d_step: float):
    # save the configurations
    store.save("Simulation", "modelling", capi)
    store.save("Simulation", "solving", solver)
    sim = SimulationResult("modelling.json", "solving.json", [])

    # generate all `d` (mean distances) values
    num_d = math.floor((d_max - d_min) / d_step) + 1
    d_departing = d_min + d_step * np.arange(num_d)
    d_approaching = d_max - d_step * np.arange(num_d)
    all_d = np.concatenate((d_departing, d_approaching))

    # inform
    print(f"Problem size: {capi.region.nx}x{capi.region.ny}. "
          f"Simulating for all {len(all_d)} mean distance values in...\n{all_d}")

    # simulate
    x = capi.phi.ravel()
    for index, d in enumerate(all_d):
        # update the parameter
        print(f""
              f"Parameter of interest: mean distance={d}")
        capi.update_displacement(d)

        # solve the problem
        numopt = capi.formulate_with_constant_volume(V)
        x, t_exec = solver.solve_minimisation(numopt, x)

        # save the result
        data = SimulationStep([0,0], d, capi.phi, t_exec)
        store.save("Simulation", f"steps---{index}", data)
        sim.steps.append(f"steps---{index}.json")

    store.save("Simulation", "result", sim)

    # reload to get all data in memory and post-process
    sim = store.load("Simulation", "result", SimulationResult)
    p_sim = post_process(sim)
    store.save("Processed", "result", p_sim)


def simulate_quasi_static_slide(store: FilesToReadWrite, capi: CapillaryBridge, solver: AugmentedLagrangian,
                                V: float, slide_by_indices: list[tuple[int, int]]):
    # # save the configurations
    # store.save("Simulation", "modelling", capi)
    # store.save("Simulation", "solving", solver)
    # sim = SimulationResult("modelling.json", "solving.json", [])
    #
    # # inform
    # print(f"Problem size: {capi.region.nx}x{capi.region.ny}. "
    #       f"Simulating for all {len(slide_by_indices)} mean distance values in...\n{slide_by_indices}")
    #
    # # simulate
    # x = capi.phi.ravel()
    # for index, m in enumerate(slide_by_indices):
    #     # update the parameter
    #     print(f""
    #           f"Parameter of interest: slide by indices={m}")
    #     capi.update_displacement(m1=m)
    #
    #     # solve the problem
    #     numopt = capi.formulate_with_constant_volume(V)
    #     x, t_exec = solver.solve_minimisation(numopt, x)
    #
    #     # save the result
    #     data = SimulationStep(m, 0., capi.phi, t_exec)
    #     store.save("Simulation", f"steps---{index}", data)
    #     sim.steps.append(f"steps---{index}.json")
    #
    # store.save("Simulation", "result", sim)

    # reload to get all data in memory and post-process
    sim = store.load("Simulation", "result", SimulationResult)
    p_sim = post_process(sim)
    store.save("Processed", "result", p_sim)
