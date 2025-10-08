"""
Solving the numerical optimization problem. No physics meaning in this file.
"""

import dataclasses as dc
import logging
import timeit
import typing as t_

import numpy as np
import scipy.optimize as optimize


logger = logging.getLogger(__name__)


class NumOptEqB(t_.Protocol):
    """Numerical optimization problem with equality constraints and simple bounds.

    x* = arg min f(x)

    s.t. g(x) = 0, x_lb <= x <= x_ub
    """
    def get_x(self) -> np.ndarray: ...

    def set_x(self, x: np.ndarray): ...

    def get_f(self) -> float: ...
    
    def get_f_Dx(self) -> np.ndarray: ...

    def get_g(self) -> float: ...

    def get_g_Dx(self) -> np.ndarray: ...

    @property
    def x_lb(self) -> float: ...

    @property
    def x_ub(self) -> float: ...


@dc.dataclass
class AugmentedLagrangian:
    max_inner_iter: int
    max_outer_loop: int
    tol_convergence: float
    tol_constraint: float
    init_penalty_weight: float

    def __post_init__(self):
        # terminate if creeping is detected (in unit of machine precision)
        # only used by inner solver
        self.tol_creeping = 1e1
        # parameters deciding how to grow the penalty weight
        self.sufficient_constr_dec = 1e-2
        self.penalty_weight_growth = 3e0

    def solve_minimisation(
        self, numopt: NumOptEqB, x0: np.ndarray, lam0: float = 0
    ):
        # print headers
        nabla = "\u2207"
        delta = "\u0394"
        tabel_headers_line1 = [
            "Loop",
            "f",
            f"|Pr({nabla}L)|",
            "|g|",
            f"|{delta} lam|",
            "c"
        ]
        tabel_headers_line2 = [
            "Iter",
            f"|res {nabla}|",
            "Message",
        ]
        separator = "  "
        logger.info(
            separator.join(
                "{:<4}".format(col_name) if col_name in ["Loop"] else "{:<8}".format(col_name)
                for col_name in tabel_headers_line1
            )
        )
        logger.info(
            separator.join(
                "{:<4}".format(col_name) if col_name in ["Iter"] else "{:<8}".format(col_name)
                for col_name in tabel_headers_line2
            )
        )
        logger.info("="*50)

        # initial values
        t_exec = 0
        x_shape = x0.shape
        x_plus = x0.ravel()
        lam_plus = lam0
        c_plus = self.init_penalty_weight
        is_converged = False
        reached_iter_limit = False
        had_abnormal_stop = False
        lam = lam0  # only for the purpose of printing delta-lam at count=0

        for count in range(self.max_outer_loop):
            # compute values that must be evaluated before update
            norm_delta_lam = abs(lam_plus - lam)

            # update primal, dual and penalty parameter
            # for primal, clip the solution to fit within the feasible region
            x = np.clip(x_plus, numopt.x_lb, numopt.x_ub)
            lam = lam_plus
            c = c_plus

            # derive augmented lagrangian
            def l(x: np.ndarray):
                """Augmented Lagrangian."""
                x = x.reshape(x_shape)
                numopt.set_x(x)
                g = numopt.get_g()
                return numopt.get_f() + lam * g + (0.5 * c) * g**2

            def l_grad(x: np.ndarray):
                """Gradient of the Augmented Lagrangian."""
                x = x.reshape(x_shape)
                numopt.set_x(x)
                l_Dx = numopt.get_f_Dx() + (lam + c * numopt.get_g()) * numopt.get_g_Dx()
                # projecting to the feasible range
                l_Dx[(x <= numopt.x_lb) & (l_Dx > 0)] = 0
                l_Dx[(x >= numopt.x_ub) & (l_Dx < 0)] = 0
                return l_Dx.ravel()

            # print status before calling inner solver
            numopt.set_x(x)
            obj_value = numopt.get_f()
            norm_lagr_gradient = np.max(abs(l_grad(x)))
            constr_violation = abs(numopt.get_g())
            padded_literals = [
                f"{count:>4d}",
                f"{obj_value:>8.1e}",
                f"{norm_lagr_gradient:>8.1e}",
                f"{constr_violation:>8.1e}",
                f"{norm_delta_lam:>8.1e}",
                f"{c:>8.1e}",
            ]
            logger.info(separator.join(padded_literals))

            # convergence criteria
            if norm_lagr_gradient < self.tol_convergence and constr_violation < self.tol_constraint:
                is_converged = True
                logger.info(f"Notice: achieving required tolerance at iter#{count}")
                break

            # solve minimisation of augmented lagrangian
            t_exec_sub = -timeit.default_timer()
            [x_plus, _, info] = optimize.fmin_l_bfgs_b(
                l,
                x,  # old solution as new initial guess
                fprime=l_grad,
                # relative decreasing of `f`, in units of `eps`; if set it to 0,
                # the test will stop the algorithm only if the objective function remains unchanged after one iteration
                factr=self.tol_creeping,
                # this `pg` should be zero at exactly a local minimizer
                pgtol=self.tol_convergence,
                maxiter=self.max_inner_iter,
            )
            t_exec_sub += timeit.default_timer()
            t_exec += t_exec_sub

            # print inner solver progress
            res_norm_grad = np.max(abs(info["grad"]))
            padded_literals = [f"{info['nit']:>4d}", f"{res_norm_grad:>8.1e}", info["task"]]
            logger.info(separator.join(padded_literals))
            logger.info("-"*50)

            # if reaches maximal iterations, simply do another loop to run more iterations
            # only the "x" is updated
            if info["warnflag"] == 1:
                reached_iter_limit = True
                continue
            # else if it stops due to some "abnomral" reason
            elif info["warnflag"] == 2:
                had_abnormal_stop = True
                break
            # else, it achieves convergence
            numopt.set_x(x_plus)
            constr_violation_plus = numopt.get_g()
            # update Lagrangian multiplier following the formula
            lam_plus = lam + c * constr_violation_plus
            # increase penalty weight if the constraint didn't improve enough
            if not (abs(constr_violation_plus) < self.tol_constraint) and not (
                abs(constr_violation_plus) < self.sufficient_constr_dec * abs(constr_violation)
            ):
                c_plus = c * self.penalty_weight_growth

        # show a warning under necessary conditions
        if not is_converged:
            logger.warning("WARNING: not converged.")
        if reached_iter_limit:
            logger.warning("WARNING: reached inner iteration limit.")
        if had_abnormal_stop:
            logger.warning("WARNING: had abnormal stop.")

        # more prints
        logger.info(f"Total time for inner solver: {t_exec:.1e} seconds.")
        logger.info(f"Ends with dual variable lambda={lam:.6f}")

        return SolverResult(x_plus.reshape(x_shape), lam, t_exec, is_converged, reached_iter_limit, had_abnormal_stop)


class SolverResult(t_.NamedTuple):
    primal: np.ndarray
    dual: float
    time: float
    is_converged: bool
    reached_iter_limit: bool
    had_abnormal_stop: bool
