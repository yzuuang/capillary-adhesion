"""
Solving the numerical optimization problem. No physics meaning in this file.
"""

import dataclasses as dc
import timeit
import typing as t_

import numpy as np
import scipy.optimize as optimize


@dc.dataclass
class NumOptEq:
    """Numerical optimization problem with equality constraints.

    x* = arg min f(x)

    s.t. g(x) = 0
    """

    f: t_.Callable[[np.ndarray], float]
    """
    def f(x: np.ndarray) -> float: ...
    """

    f_grad: t_.Callable[[np.ndarray], np.ndarray]
    """
    def f_grad(x: np.ndarray) -> np.ndarray: ...
    """

    g: t_.Callable[[np.ndarray], float]
    """
    def g(x: np.ndarray) -> float: ...
    """

    g_grad: t_.Callable[[np.ndarray], np.ndarray]
    """
    def g_grad(x: np.ndarray) -> np.ndarray: ...
    """


@dc.dataclass
class AugmentedLagrangian:
    max_inner_iter: int
    max_outer_loop: int
    tol_convergence: float
    tol_constraint: float
    init_penalty_weight: float

    def __post_init__(self):
        # parameters related to convergence, used only by inner solver
        self.tol_creeping = 1e1
        # parameters deciding how to grow the penalty weight
        self.sufficient_constr_dec = 1e-2
        self.penalty_weight_growth = 3e0

    def solve_minimisation(
        self, numopt: NumOptEq, x0: np.ndarray, x_lb: float = -np.inf, x_ub: float = +np.inf
    ):
        # print headers
        nabla = "\u2207"
        tabel_headers = ["Loop", "f", f"|Pr({nabla}L)|", "|g|", "lam", "c", "Iter", f"|res {nabla}|", "Message"]
        separator = "  "
        print(
            *[
                "{:<4}".format(col_name) if col_name in ["Loop", "Iter"] else "{:<8}".format(col_name)
                for col_name in tabel_headers
            ],
            sep=separator,
        )
        t_exec = 0

        # initial values
        x_plus = x0
        lam_plus = 0.0
        c_plus = self.init_penalty_weight
        is_converged = False

        for count in range(self.max_outer_loop):
            # update primal, dual and parameter
            # FIXME: the solver shouldn't know the exact values (0 and 1)
            x = np.clip(x_plus, x_lb, x_ub)
            lam = lam_plus
            c = c_plus

            # derive augmented lagrangian
            def l(x: np.ndarray):
                """Augmented Lagrangian."""
                g_x = numopt.g(x)
                return numopt.f(x) + lam * g_x + (0.5 * c) * g_x**2

            def l_grad(x: np.ndarray):
                """Gradient of the Augmented Lagrangian."""
                g_D_x = numopt.g_grad(x)
                l_D_x = numopt.f_grad(x) + lam * g_D_x + c * numopt.g(x) * g_D_x
                # projecting to the [0, 1] feasible range
                l_D_x[(x <= x_lb) & (l_D_x > 0)] = 0
                l_D_x[(x >= x_ub) & (l_D_x < 0)] = 0
                return l_D_x

            # print status before calling inner solver
            obj_value = numopt.f(x)
            norm_lagr_gradient = max(abs(l_grad(x)))
            constr_violation = abs(numopt.g(x))
            padded_literals = [
                f"{count:>4d}",
                f"{obj_value:>8.1e}",
                f"{norm_lagr_gradient:>8.1e}",
                f"{constr_violation:>8.1e}",
                f"{lam:>8.1e}",
                f"{c:>8.1e}",
            ]
            print(separator.join(padded_literals), end=separator, flush=True)

            # convergence criteria
            if norm_lagr_gradient < self.tol_convergence and constr_violation < self.tol_constraint:
                is_converged = True
                print(f"Notice: achieving required tolerance at iter #{count}")
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
            res_norm_grad = max(abs(info["grad"]))
            padded_literals = [f"{info['nit']:>4d}", f"{res_norm_grad:>8.1e}", info["task"]]
            print(separator.join(padded_literals))

            # if reaches maximal iterations, simply do another loop to run more iterations
            # only the "x" is updated
            if info["warnflag"] == 1:
                continue
            # else, it achieves convergence
            constr_violation_plus = numopt.g(x_plus)
            # update Lagrangian multiplier following the formula
            lam_plus = lam + c * constr_violation_plus
            # increase penalty weight if the constraint didn't improve enough
            if not (abs(constr_violation_plus) < self.sufficient_constr_dec * abs(constr_violation)):
                c_plus = c * self.penalty_weight_growth

        # check if not solved
        if not is_converged:
            print(f"WARNING: NOT CONVERGED.")

        print(f"Total time for inner solver: {t_exec:.1e} seconds.")

        return x_plus, t_exec, lam
