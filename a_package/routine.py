"""
Simulation routines: modelling, solving and post-processing.
"""

import dataclasses as dc
import logging

import numpy as np

from a_package.modelling import CapillaryBridge
from a_package.computing import Grid
from a_package.minimising import AugmentedLagrangian
from a_package.formulating import Formulation
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
    E = np.empty((n_step))               
    p = np.empty((n_step))               
    V = np.empty((n_step))               
    P = np.empty((n_step))               

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

        E[i] = capi.energy
        p[i] = step.lam
        V[i] = capi.volume
        P[i] = capi.perimeter

    # get normal force by numerical differences of energy
    Fz = -(E[1:] - E[:-1]) / (r[1:, 2] - r[:-1, 2])

    # pack in an object
    # HACK: skip this "troublemaking" value
    # evo = Evolution(t, g, phi, r, E, p, V, P, Fz)
    evo = Evolution(t, g, phi, r, E, V, P, Fz)
    return ProcessedResult(res.modelling, res.solving, evo)


@dc.dataclass
class Evolution:
    t_exec: np.ndarray
    g: list[np.ndarray]
    phi: list[np.ndarray]
    r: np.ndarray       # relative displacement
    E: np.ndarray       # energy
    # Hack: skip this "troublemaking" value
    # p: np.ndarray       # presure
    V: np.ndarray       # volume
    P: np.ndarray       # perimeter
    Fz: np.ndarray      # normal force


@dc.dataclass
class ProcessedResult:
    modelling: CapillaryBridge
    solving: AugmentedLagrangian
    evolution: Evolution


logger = logging.getLogger(__name__)


def simulate_quasi_static_pull_push(
    store: FilesToReadWrite,
    grid: Grid,
    upper: np.ndarray,
    lower: np.ndarray,
    capillary: CapillaryBridge,
    phase: np.ndarray,
    volume: float,
    minimiser: AugmentedLagrangian,
    trajectory: list[float],
    round_trip: bool = True,
):
    # save the configurations
    store.save("modelling", capillary)
    store.save("solving", minimiser)
    sim = SimulationResult("modelling.json", "solving.json", [])

    formulation = Formulation(grid, upper, lower, capillary)

    trajectory = np.array(trajectory)
    if round_trip:
        trajectory = np.concatenate((trajectory, np.flip(trajectory)[1:]))
    # Truncate to remove floating point errors
    nb_decimals = 6
    trajectory = np.round(trajectory, nb_decimals)

    # inform
    logger.info(
        f"Problem size: {grid.nx}x{grid.ny}. "
        f"Simulating for all {len(trajectory)} mean distance values in...\n{trajectory}"
    )
    report = {
        "not_converged": [],
        "iter_limit": [],
        "abnormal_stop": [],
    }

    # simulate
    x = np.ravel(phase)
    lam = 0.0
    for index, delta in enumerate(trajectory):
        # update the parameter
        logger.info(f"Parameter of interest: mean distance={delta}")
        formulation.update_gap(delta)
        formulation.update_phase_field(x)

        # solve the problem
        numopt = formulation.formulate_with_constant_volume(volume)
        [x, lam, t_exec, *flags] = minimiser.solve_minimisation(numopt, x, lam, 0, 1)
        if not flags[0]:
            report["not_converged"].append(index)
        if flags[1]:
            report["iter_limit"].append(index)
        if flags[2]:
            report["abnormal_stop"].append(index)

        # save the results
        formulation.phase = x.reshape(grid.nx, grid.ny)
        # data = SimulationStep([0, 0], delta, t_exec, formulation.phase, lam)
        # store.save(f"steps---{index}", data)
        # sim.steps.append(f"steps---{index}.json")

        # Check the bounds on phase field
        formulation.validate_phase_field()

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


def simulate_quasi_static_slide(
    store: FilesToReadWrite,
    grid: Grid,
    upper: np.ndarray,
    lower: np.ndarray,
    capillary: CapillaryBridge,
    phase: np.ndarray,
    volume: float,
    minimiser: AugmentedLagrangian,
    slide_by_indices: list[tuple[int, int]],
):
    # save the configurations
    store.save("modelling", capillary)
    store.save("solving", minimiser)
    sim = SimulationResult("modelling.json", "solving.json", [])

    formulation = Formulation(grid, upper, lower, capillary)

    # inform
    logger.info(
        f"Problem size: {grid.nx}x{grid.ny}. "
        f"Simulating for all {len(slide_by_indices)} mean distance values in...\n{slide_by_indices}"
    )
    report = {
        "not_converged": [],
        "iter_limit": [],
        "abnormal_stop": [],
    }

    # simulate
    x = np.ravel(phase)
    lam = 0.0
    for index, coords in enumerate(slide_by_indices):
        # update the parameter
        logger.info(f"Parameter of interest: displacement={coords}")
        # FIXME: sliding
        formulation.update_phase_field(x)

        # solve the problem
        numopt = formulation.formulate_with_constant_volume(volume)
        [x, lam, t_exec, *flags] = minimiser.solve_minimisation(numopt, x, lam, 0, 1)
        if not flags[0]:
            report["not_converged"].append(index)
        if flags[1]:
            report["iter_limit"].append(index)
        if flags[2]:
            report["abnormal_stop"].append(index)

        # save the results
        formulation.phase = x.reshape(grid.nx, grid.ny)
        # data = SimulationStep([0, 0], delta, t_exec, formulation.phase, lam)
        # store.save(f"steps---{index}", data)
        # sim.steps.append(f"steps---{index}.json")

        # Check the bounds on phase field
        formulation.validate_phase_field()

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
