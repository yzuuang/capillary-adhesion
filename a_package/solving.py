"""This file addreses solving optimization problem.

- Approximate a constrained optimization with an unconstrained problem using augmented Lagrangian 
method.
- Iterative approaches to find numerically the minimizer of an unconstrained problem.
"""

import dataclasses as dc
import typing as _t
import timeit

import numpy as np
from NuMPI.Optimization import l_bfgs


@dc.dataclass
class AugmentedLagrangian:
    inner_max_iter: int
    tol_convergence: float
    tol_constraint: float
    c_init: float
    c_upper_bound: float
    beta: float

    @staticmethod
    def f(x: np.ndarray):
        """To be specified by the simulation. Should return the objective
        function value at x.
        """
        pass

    @staticmethod
    def g(x: np.ndarray):
        """To be specified by the simulation. Should return the constraint
        function value at x.
        """
        pass

    # NOTE: define `l` and `dx_l` explicitly to avoid repeated updating of phase-field.

    @staticmethod
    def l(x: np.ndarray, lam: float, c: float):
        """To be specified by the simulation. Should return values equal to
        f(x) + lam * g(x) + (0.5 * c) * g(x)**2
        """
        pass

    @staticmethod
    def dx_l(x: np.ndarray, lam: float, c: float):
        """To be specified by the simulation. Should return values equal to
        dx_f(x) + lam * dx_g(x) + c * g(x) * dx_g(x)
        """
        pass

    def find_minimizer(self, x0: np.ndarray):
        # compute all possible `                                                                                                                                                                 c` values, i.e. for(c=c_init; c<c_upper_bound; c*=beta)
        num_iter = int(np.log(self.c_upper_bound / self.c_init) / np.log(self.beta)) + 1
        cc = self.c_init * np.pow(self.beta, np.arange(num_iter))

        # initial setup
        x_plus = x0
        lam = 0
        t_exec = 0

        # inform
        tabel_header = [
            "Iter",
            "T_exec",
            "Objective",
            "lambda\t",
            "Lagrangian",
            "c\t",
            "Augm. Lagr.",
            "max grad",
        ]
        print(*tabel_header, sep="\t")

        for k, c in enumerate(cc):
            # solve minimization problem
            t_exec_sub = -timeit.default_timer()
            res = l_bfgs(
                self.l,
                # old solution as new initial guess
                x_plus,
                # bounds=[(0, 1)] * len(x_plus),
                jac=self.dx_l,
                args=(lam, c),
                ftol=1e1,  # for extremely high accuracy
                gtol=self.tol_convergence,
                maxiter=self.inner_max_iter,
            )
            t_exec_sub += timeit.default_timer()
            t_exec += t_exec_sub

            # inform
            x_plus = res['x']
            l_plus = res['fun']
            l_grad = res['jac']
            f_plus = self.f(x_plus)
            error_g_x = self.g(x_plus)
            lagr1 = f_plus + lam * error_g_x

            tabel_entry = [
                f"#{k}",
                f"{round(t_exec_sub, 2):.2f}s",
                f"{f_plus:.2e}",
                f"{lam:.2e}",
                f"{lagr1:.2e}",
                f"{c:.2e}",
                f"{l_plus:.2e}",
                f"{np.amax[np.absolute(l_grad)]:.2e}",
            ]
            print("\t".join(tabel_entry))

            # convergence criteria
            if abs(error_g_x) < self.tol_constraint:
                print(f"Notice: achieving required tolerance at iter #{k}")
                break

            # update the estimate of Lagrangian multiplier
            lam += c * error_g_x

            # check if not solved
            if k + 1 == num_iter:
                print(f"Warning: maximal AL iter #{num_iter}")

        print(f"Total time for inner solver: {t_exec:.1e} seconds.")

        return x_plus, t_exec, lam
