"""
Simulation routines: modelling, solving and post-processing.
"""

import dataclasses as dc
import logging
import math

import numpy as np

from a_package.modelling import CapillaryBridge
from a_package.solving import AugmentedLagrangian
from a_package.storing import FilesToReadWrite


logger = logging.getLogger(__name__)


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
    V = np.empty((n_step))               # volume
    P = np.empty((n_step))               # perimeter

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
        P[i] = capi.perimeter

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


def simulate_quasi_static_pull_push(store: FilesToReadWrite, capi: CapillaryBridge, solver: AugmentedLagrangian,
                                    V: float, d_min: float, d_max: float, d_step: float):
    # save the configurations
    store.save("modelling", capi)
    store.save("solving", solver)
    sim = SimulationResult("modelling.json", "solving.json", [])

    # generate all `d` (mean distances) values
    # FIXME: specify nb_steps may be a better choice
    num_d = math.ceil((d_max - d_min) / d_step) + 1
    d_approaching = d_max - d_step * np.arange(num_d)
    d_departing = np.flip(d_approaching)
    all_d = np.concatenate((d_approaching, d_departing[1:]))

    # Truncate to remove floating point errors
    # NOTE: assume 'd_step' < 0
    n_decimals = 6
    all_d = np.round(all_d, n_decimals)

    # inform
    logger.info(
        f"Problem size: {capi.region.nx}x{capi.region.ny}. "
        f"Simulating for all {len(all_d)} mean distance values in...\n{all_d}"
    )
    report = {
        'not_converged': [],
        'iter_limit': [],
        'abnormal_stop': [],
    }

    # simulate
    x = capi.phi.ravel()
    lam = 0.0
    for index, d in enumerate(all_d):
        # update the parameter
        logger.info(f"Parameter of interest: mean distance={d}")
        capi.z1 = d
        capi.update_gap()
        capi.update_phase_field()

        # solve the problem
        numopt = capi.formulate_with_constant_volume(V)
        [x, lam, t_exec, *flags] = solver.solve_minimisation(numopt, x, lam, 0, 1)
        if not flags[0]:
            report["not_converged"].append(index)
        if flags[1]:
            report["iter_limit"].append(index)
        if flags[2]:
            report["abnormal_stop"].append(index)

        # save the results
        capi.phi = x.reshape(capi.region.nx, capi.region.ny)
        # capi.phi = capi.squashing(x.reshape(capi.region.nx, capi.region.ny))
        data = SimulationStep([0,0], d, t_exec, capi.phi, lam)
        store.save(f"steps---{index}", data)
        sim.steps.append(f"steps---{index}.json")

        # Check the bounds on phase field
        capi.validate_phase_field()

    # report
    if all(not len(v) for v in report.values()):
        logger.info("Congrats! All simulation steps went well.")
    else:
        logger.warning(f"The following steps may have problems:\n {report}")

    # Save simulation results
    store.save("result", sim)

    # Load again to get all data (because they were saved part by part)
    sim = store.load("result", SimulationResult)
    return sim


def simulate_quasi_static_slide(store: FilesToReadWrite, capi: CapillaryBridge, solver: AugmentedLagrangian,
                                V: float, slide_by_indices: list[tuple[int, int]]):
    # save the configurations
    store.save("modelling", capi)
    store.save("solving", solver)
    sim = SimulationResult("modelling.json", "solving.json", [])

    # inform
    logger.info(
        f"Problem size: {capi.region.nx}x{capi.region.ny}. "
        f"Simulating for all {len(slide_by_indices)} mean distance values in...\n{slide_by_indices}"
    )
    report = {
        'not_converged': [],
        'iter_limit': [],
        'abnormal_stop': [],
    }

    # simulate
    x = capi.phi.ravel()
    lam = 0.0
    for index, m in enumerate(slide_by_indices):
        # update the parameter
        logger.info(f"Parameter of interest: slide by indices={m}")
        capi.ix1_iy1 = m
        capi.update_gap()

        # solve the problem
        numopt = capi.formulate_with_constant_volume(V)
        [x, lam, t_exec, *flags] = solver.solve_minimisation(numopt, x, lam, 0, 1)
        if not flags[0]:
            report["not_converged"].append(index)
        if flags[1]:
            report["iter_limit"].append(index)
        if flags[2]:
            report["abnormal_stop"].append(index)

        # save the result
        capi.phi = x.reshape(capi.region.nx, capi.region.ny)
        data = SimulationStep(m, capi.z1, t_exec, capi.phi, lam)
        store.save(f"steps---{index}", data)
        sim.steps.append(f"steps---{index}.json")

        # Check the bounds on phase field
        capi.validate_phase_field()

    # report
    if all(not len(v) for v in report.values()):
        logger.info("Congrats! All simulation steps went well.")
    else:
        logger.warning(f"The following steps may have problems:\n {report}")

    # Save simulation results
    store.save("result", sim)

    # Load again to get all data (because they were saved part by part)
    sim = store.load("result", SimulationResult)
    return sim
