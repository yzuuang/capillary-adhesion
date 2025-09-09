
import dataclasses as dc
import logging

import numpy as np

from a_package.workflow.formulation import Formulation
from a_package.numeric import AugmentedLagrangian
from a_package.utils import FilesToReadWrite


logger = logging.getLogger(__name__)


@dc.dataclass(init=True)
class SimulationStep:
    m: tuple[int, int]
    d: float
    t_exec: float
    phi: np.ndarray
    lam: float


@dc.dataclass(init=True)
class SimulationResult:
    formulating: Formulation
    minimising: AugmentedLagrangian
    steps: list[SimulationStep]


def simulate_quasi_static_pull_push(
    store: FilesToReadWrite,
    formulation: Formulation,
    minimiser: AugmentedLagrangian,
    volume: float,
    phase_init: np.ndarray,
    trajectory: list[float],
    round_trip: bool = True,
):
    # save the configurations
    store.save("simulation---formulating", formulation)
    store.save("simulation---minimising", minimiser)
    result = SimulationResult("simulation---formulating.json", "simulation---minimising.json", [])

    trajectory = np.array(trajectory)
    if round_trip:
        trajectory = np.concatenate((trajectory, np.flip(trajectory)[1:]))
    # Truncate to remove floating point errors
    nb_decimals = 6
    trajectory = np.round(trajectory, nb_decimals)

    # inform
    logger.info(
        f"Problem size: {'x'.join(str(dim) for dim in phase_init.shape)}. "
        f"Simulating for all {len(trajectory)} mean distance values in...\n{trajectory}"
    )
    report = {
        "not_converged": [],
        "iter_limit": [],
        "abnormal_stop": [],
    }

    # simulate
    original_shape = np.shape(phase_init.squeeze())
    x = phase_init
    lam = 0.0
    for index, delta_z in enumerate(trajectory):
        # update the parameter
        logger.info(f"Parameter of interest: mean distance={delta_z}")
        formulation.update_gap(delta_z)
        formulation.update_phase_field(x)

        # solve the problem
        numopt = formulation.create_numopt_with_constant_volume(volume)
        solver_result = minimiser.solve_minimisation(numopt, x, lam, 0, 1)

        # check flags
        if not solver_result.is_converged:
            report["not_converged"].append(index)
        if solver_result.reached_iter_limit:
            report["iter_limit"].append(index)
        if solver_result.had_abnormal_stop:
            report["abnormal_stop"].append(index)

        # check bonds (feasibility)
        phase = np.reshape(solver_result.primal, original_shape)
        formulation.validate_phase_field(phase)

        # save the results
        store.save(
            f"simulation---steps---{index}",
            SimulationStep([0, 0], delta_z, solver_result.time, phase, solver_result.dual),
        )
        result.steps.append(f"simulation---steps---{index}.json")

        # update next iter
        x = phase
        lam = solver_result.dual

    # report
    if all(not len(v) for v in report.values()):
        logger.info("Congrats! All simulation steps went well.")
    else:
        logger.warning(f"The following steps may have problems:\n {report}")

    store.save("simulation", result)

    # reload so that every string represented objects are also loaded as objects
    return store.load("simulation", SimulationResult)


# FIXME: sliding
# def simulate_quasi_static_slide(
#     store: FilesToReadWrite,
#     formulation: Formulation,
#     minimiser: AugmentedLagrangian,
#     volume: float,
#     phase_init: np.ndarray,
#     slide_by_indices: list[tuple[int, int]],
# ):
#     # save the configurations
#     store.save("modelling", capillary)
#     store.save("solving", minimiser)
#     sim = SimulationResult("modelling.json", "solving.json", [])

#     formulation = Formulation(grid, upper, lower, capillary)

#     # inform
#     logger.info(
#         f"Problem size: {grid.nx}x{grid.ny}. "
#         f"Simulating for all {len(slide_by_indices)} mean distance values in...\n{slide_by_indices}"
#     )
#     report = {
#         "not_converged": [],
#         "iter_limit": [],
#         "abnormal_stop": [],
#     }

#     # simulate
#     x = np.ravel(phase)
#     lam = 0.0
#     for index, coords in enumerate(slide_by_indices):
#         # update the parameter
#         logger.info(f"Parameter of interest: displacement={coords}")
#         # FIXME: sliding
#         formulation.update_phase_field(x)

#         # solve the problem
#         numopt = formulation.formulate_with_constant_volume(volume)
#         [x, lam, t_exec, *flags] = minimiser.solve_minimisation(numopt, x, lam, 0, 1)
#         if not flags[0]:
#             report["not_converged"].append(index)
#         if flags[1]:
#             report["iter_limit"].append(index)
#         if flags[2]:
#             report["abnormal_stop"].append(index)

#         # save the results
#         formulation.phase = x.reshape(grid.nx, grid.ny)
#         # data = SimulationStep([0, 0], delta, t_exec, formulation.phase, lam)
#         # store.save(f"steps---{index}", data)
#         # sim.steps.append(f"steps---{index}.json")

#         # Check the bounds on phase field
#         formulation.validate_phase_field()

#     # report
#     if all(not len(v) for v in report.values()):
#         logger.info("Congrats! All simulation steps went well.")
#     else:
#         logger.warning(f"The following steps may have problems:\n {report}")

#     # Save simulation results
#     store.save("result", sim)

#     # Load again to get all data (because they were saved part by part)
#     sim = store.load("result", SimulationResult)
#     return sim
