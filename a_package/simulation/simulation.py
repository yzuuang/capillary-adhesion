"""
Simulation orchestration.

Provides the main Simulation class that coordinates contact solving,
phase field solving, and result storage.
"""

import logging
import pathlib
import types

import numpy as np

from a_package.domain import Grid, adapt_shape
from a_package.numerics.optimizer import AugmentedLagrangian
from a_package.physics.capillary import NodalFormCapillary
from a_package.simulation.io import SimulationIO, Term


logger = logging.getLogger(__name__)


class ContactSolver:
    """Computes the gap field between two surfaces at a given separation."""

    def __init__(self, grid: Grid, upper: np.ndarray, lower: np.ndarray):
        self.upper = adapt_shape(upper)
        self.lower = adapt_shape(lower)

    def solve_gap(self, separation: float):
        return np.clip(separation + self.upper - self.lower, 0, None)


class PhaseSolver:
    """
    Solves for the equilibrium phase field under various constraints.

    Owns the formulation and optimizer, provides methods for different
    constraint types.
    """

    formulation: NodalFormCapillary
    optimizer: AugmentedLagrangian

    def __init__(self, grid: Grid, capillary_args: dict, optimizer_args: dict):
        self.formulation = NodalFormCapillary(grid, capillary_args)
        self.optimizer = AugmentedLagrangian(**optimizer_args)

    def solve_with_constant_volume(self, gap, phase, pressure, volume):
        """
        Solve for equilibrium phase field with constant volume constraint.

        Parameters
        ----------
        gap : np.ndarray
            Gap field between surfaces.
        phase : np.ndarray
            Initial guess for phase field.
        pressure : float
            Initial guess for pressure (Lagrange multiplier).
        volume : float
            Target liquid volume.

        Returns
        -------
        tuple[np.ndarray, float]
            (phase, pressure) - optimized phase field and pressure.
        """
        # gap is simply a parameter of the optimization problem
        self.formulation.set_gap(gap)

        # save the original shape
        original_shape = phase.shape

        # wrap volume constraint into a function
        def volume_constraint():
            return self.formulation.get_volume() - volume

        # pack into an object matching the "NumOptEqB" protocol typing
        problem = types.SimpleNamespace(
            get_x=self.formulation.get_phase,
            set_x=self.formulation.set_phase,
            get_f=self.formulation.get_energy,
            get_f_Dx=self.formulation.get_energy_jacobian,
            get_g=volume_constraint,
            get_g_Dx=self.formulation.get_volume_jacobian,
            x_lb=self.formulation.phase_lb,
            x_ub=self.formulation.phase_ub,
        )

        # call the minimizer
        res = self.optimizer.solve_minimisation(problem, x0=phase, lam0=pressure)
        phase = np.reshape(res.primal, original_shape)
        pressure = res.dual
        return phase, pressure

    def check_phase_feasibility(self, phase: np.ndarray):
        """Check and warn if phase field exceeds bounds."""
        # check lower bound
        if np.any(phase < self.formulation.phase_lb):
            outlier = np.where(phase < self.formulation.phase_lb, phase, np.nan)
            count = np.count_nonzero(~np.isnan(outlier))
            extreme = np.nanmin(outlier)
            logger.warning(
                f"Notice: phase field has {count} values exceeding lower bound, "
                f"min at {self.formulation.phase_lb:.1f}-{-extreme:.2e}."
            )
        # check upper bound
        if np.any(phase > self.formulation.phase_ub):
            outlier = np.where(phase > self.formulation.phase_ub, phase, np.nan)
            count = np.count_nonzero(~np.isnan(outlier))
            extreme = np.nanmax(outlier)
            logger.warning(
                f"Notice: phase field has {count} values exceeding upper bound, "
                f"max at {self.formulation.phase_ub:.1f}+{extreme - 1:.2e}."
            )


class Simulation:
    """
    Main simulation orchestrator.

    Coordinates contact solving, phase solving, and IO for different
    simulation types.
    """

    def __init__(
        self,
        grid: Grid,
        capillary_args: dict,
        optimizer_args: dict,
    ):
        self.grid = grid
        self.capillary_args = capillary_args
        self.optimizer_args = optimizer_args

    def run_with_constant_volume(
        self,
        upper: np.ndarray,
        lower: np.ndarray,
        trajectory: np.ndarray,
        volume: float,
        store_dir: pathlib.Path | str,
        phase_init: np.ndarray | None = None,
        pressure_init: float = 0,
    ) -> SimulationIO:
        """
        Run simulation with constant liquid volume constraint.

        Parameters
        ----------
        upper : np.ndarray
            Upper surface height field.
        lower : np.ndarray
            Lower surface height field.
        trajectory : np.ndarray
            Array of separation values to simulate.
        volume : float
            Target liquid volume to maintain.
        store_dir : Path | str
            Directory to store simulation results.
        phase_init : np.ndarray, optional
            Initial guess for phase field. Zeros if not provided.
        pressure_init : float, optional
            Initial guess for pressure (Lagrange multiplier).

        Returns
        -------
        SimulationIO
            IO object for accessing saved results.
        """
        assert upper.shape[-2:] == tuple(self.grid.nb_elements)
        assert lower.shape[-2:] == tuple(self.grid.nb_elements)
        assert volume > 0, "Liquid volume must be positive."

        # Truncate to remove trailing digits due to rounding errors
        trajectory = np.round(trajectory, 6)

        # Default initial phase field
        if phase_init is None:
            phase_init = np.zeros(self.grid.nb_elements)

        # Create solvers
        contact_solver = ContactSolver(self.grid, upper, lower)
        phase_solver = PhaseSolver(self.grid, self.capillary_args, self.optimizer_args)

        # IO
        io = SimulationIO(self.grid, store_dir)
        io.save_constant(
            fields={Term.phase_init: phase_init},
            single_values={Term.pressure_init: pressure_init},
        )

        # Log info
        logger.info(
            f"Problem size: {'x'.join(str(dim) for dim in phase_init.shape)}. "
            f"Simulating for {len(trajectory)} separation values..."
        )

        # Initial guess
        phase = np.asarray(phase_init)
        pressure = pressure_init

        # Simulation loop
        for index, separation in enumerate(trajectory):
            logger.info(f"Step {index}: separation={separation}")
            gap = contact_solver.solve_gap(separation)
            phase, pressure = phase_solver.solve_with_constant_volume(
                gap, phase, pressure, volume
            )

            # Check feasibility
            phase_solver.check_phase_feasibility(phase)

            # Save step
            io.save_step(
                index,
                fields={
                    Term.upper_solid: contact_solver.upper,
                    Term.lower_solid: contact_solver.lower,
                    Term.gap: gap,
                    Term.phase: phase,
                },
                single_values={
                    Term.separation: separation,
                    Term.pressure: pressure,
                    Term.energy: phase_solver.formulation.get_energy(),
                },
            )

        return io
