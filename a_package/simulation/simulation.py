import logging
import pathlib
import types

import numpy as np

from a_package.field import adapt_shape
from a_package.grid import Grid
from a_package.numerics.optimizer import AugmentedLagrangian
from a_package.simulation.formulation import NodalFormCapillary
from a_package.simulation.io import SimulationIO, Term


logger = logging.getLogger(__name__)


class Simulation:

    def __init__(
            self, grid: Grid, store_dir: pathlib.Path | str, capillary_args: dict = {},
            optimizer_args: dict = {}) -> None:
        self.grid = grid
        self.store_dir = store_dir
        self.capillary_args = capillary_args
        self.optimizer_args = optimizer_args

    def simulate_approach_retraction_with_constant_volume(
            self, upper: np.ndarray, lower: np.ndarray, volume: float, separation_trajectory: np.ndarray,
            round_trip: bool = True, phase_init: np.ndarray | None = None, pressure_init: float = 0):

        assert upper.shape[-2:] == tuple(self.grid.nb_elements)
        assert lower.shape[-2:] == tuple(self.grid.nb_elements)
        assert volume > 0, "Liquid volume must be positive."

        # Truncate to remove trailing digits due to rounding errors
        nb_decimals = 6
        separation_trajectory = np.round(separation_trajectory, nb_decimals)

        # add return trip
        if round_trip:
            separation_trajectory = np.concatenate([separation_trajectory, np.flip(separation_trajectory)[1:]])


        # a random phase as the default initial guess
        if phase_init is None:
            phase_init = np.zeros(self.grid.nb_elements)

        # lower-level solvers
        contact_solver = ContactSolver(self.grid, upper, lower)
        phase_solver = PhaseSolver(self.grid, self.capillary_args, self.optimizer_args)

        # IO
        io = SimulationIO(self.grid, self.store_dir)
        # save initial guess
        io.save_constant(fields={Term.phase_init: phase_init}, single_values={Term.pressure_init: pressure_init})

        # report = {
        #     "not_converged": [],
        #     "iter_limit": [],
        #     "abnormal_stop": [],
        # }

        # inform
        logger.info(
            f"Problem size: {'x'.join(str(dim) for dim in phase_init.shape)}. "
            f"Simulating for all {len(separation_trajectory)} mean distance values in...\n{separation_trajectory}"
        )

        # initial guess
        phase = np.asarray(phase_init)
        pressure = pressure_init

        # simulate
        for [index, separation] in enumerate(separation_trajectory):
            # update the parameter
            logger.info(f"Parameter of interest: mean distance={separation}")
            gap = contact_solver.solve_gap(separation)
            [phase, pressure] = phase_solver.solve_with_constant_volume(gap, phase, pressure, volume)

            # check flags
            # if not solver_result.is_converged:
            #     report["not_converged"].append(index)
            # if solver_result.reached_iter_limit:
            #     report["iter_limit"].append(index)
            # if solver_result.had_abnormal_stop:
            #     report["abnormal_stop"].append(index)

            # check bonds (feasibility)
            phase_solver.check_phase_feasibility(phase)

            # save this iteration to storage
            io.save_step(
                index,
                fields={Term.upper_solid: contact_solver.upper, Term.lower_solid: contact_solver.lower, Term.gap: gap,
                        Term.phase: phase},
                single_values={Term.separation: separation, Term.pressure: pressure, 
                               Term.energy: phase_solver.formulation.get_energy()})

        # report
        # if all(not len(v) for v in report.values()):
        #     logger.info("Congrats! All simulation steps went well.")
        # else:
        #     logger.warning(f"The following steps may have problems:\n {report}")

        # FIXME: need to return something?
        return io


class ContactSolver:

    def __init__(self, grid: Grid, upper: np.ndarray, lower: np.ndarray):
        self.upper = adapt_shape(upper)
        self.lower = adapt_shape(lower)

    def solve_gap(self, separation: float):
        return np.clip(separation + self.upper - self.lower, 0, None)


class PhaseSolver:

    formulation: NodalFormCapillary
    optimizer: AugmentedLagrangian

    def __init__(self, grid: Grid, capillary_args, optimizer_args):
        self.formulation = NodalFormCapillary(grid, capillary_args)
        self.optimizer = AugmentedLagrangian(**optimizer_args)

    def solve_with_constant_volume(self, gap, phase, pressure, volume):
        # gap are simply parameters of the optimization problem
        self.formulation.set_gap(gap)

        # save the original shape
        original_shape = phase.shape

        # wrap volume constraint into a function
        def volume_constraint():
            return self.formulation.get_volume() - volume

        # pack into an object matching the "NumOptEqB" protocol typing
        problem = types.SimpleNamespace(
            get_x=self.formulation.get_phase, set_x=self.formulation.set_phase, get_f=self.formulation.get_energy,
            get_f_Dx=self.formulation.get_energy_jacobian, get_g=volume_constraint,
            get_g_Dx=self.formulation.get_volume_jacobian, x_lb=self.formulation.phase_lb,
            x_ub=self.formulation.phase_ub)

        # call the minimizer
        res = self.optimizer.solve_minimisation(problem, x0=phase, lam0=pressure)
        phase = np.reshape(res.primal, original_shape)
        pressure = res.dual
        return phase, pressure

    def check_phase_feasibility(self, phase: np.ndarray):
        # check lower bound
        if np.any(phase < self.formulation.phase_lb):
            outlier = np.where(phase < self.formulation.phase_lb, phase, np.nan)
            count = np.count_nonzero(~np.isnan(outlier))
            extreme = np.nanmin(outlier)
            logger.warning(
                f"Notice: phase field has {count} values exceeding lower bound, min at {self.formulation.phase_lb:.1f}-{-extreme:.2e}.")
        # check upper bound
        if np.any(phase > self.formulation.phase_ub):
            outlier = np.where(phase > self.formulation.phase_ub, phase, np.nan)
            count = np.count_nonzero(~np.isnan(outlier))
            extreme = np.nanmax(outlier)
            logger.warning(
                f"Notice: phase field has {count} values exceeding upper bound, max at  {self.formulation.phase_ub:.1f}+{extreme - 1:.2e}.")


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
