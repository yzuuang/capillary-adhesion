"""This file addreses solving optimization problem.

- Approximate a constrained optimization with an unconstrained problem using augmented Lagrangian 
method.
- Iterative approaches to find numerically the minimizer of an unconstrained problem.
"""

import dataclasses as dc
import timeit
import typing as _t

import numpy as np
from NuMPI.Optimization import l_bfgs


@dc.dataclass
class AugmentedLagrangian:
    tolerance_convergence: float
    tolerance_constraint: float
    max_iter: int
    c0: float
    beta: float
    k_max: int

    @staticmethod
    def f(x: np.ndarray):
        """To be specified by the simulation. Should return the objective
        function value at x.
        """
        raise NotImplementedError()

    @staticmethod
    def g(x: np.ndarray):
        """To be specified by the simulation. Should return the constraint
        function value at x.
        """
        raise NotImplementedError()

    @staticmethod
    def l(x: np.ndarray, lam: float, c: float):
        """To be specified by the simulation. Should return values equal to
        f(x) + lam * g(x) + (0.5 * c) * g(x)**2
        """
        raise NotImplementedError()

    @staticmethod
    def dx_l(x: np.ndarray, lam: float, c: float):
        """To be specified by the simulation. Should return values equal to
        dx_f(x) + lam * dx_g(x) + c * g(x) * dx_g(x)
        """
        raise NotImplementedError()

    def find_minimizer(self, x0: np.ndarray):
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

        # initial setup
        x_plus = x0
        lam = 0
        t_exec = 0
        solved = False

        cc = self.c0 * np.pow(self.beta, np.arange(self.k_max))
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
                gtol=self.tolerance_convergence,
                maxiter=self.max_iter,
            )
            t_exec_sub += timeit.default_timer()
            t_exec += t_exec_sub

            # inform
            x_plus = res['x']
            l_plus = res['fun']
            l_grad = res['jac']
            f_plus = self.f(x_plus)
            g_plus = self.g(x_plus)
            lagr1 = f_plus + lam * g_plus

            tabel_entry = [
                f"#{k}",
                f"{round(t_exec_sub, 2):.2f}s",
                f"{f_plus}",
                f"{lam}",
                f"{lagr1}",
                f"{c}",
                f"{l_plus}",
                f"{np.amax(np.absolute(l_grad))}",
            ]
            print("\t".join(tabel_entry))

            # convergence criteria
            if abs(g_plus) < self.tolerance_constraint:
                solved = True
                print(f"Achieved required tolerance at iter #{k}")
                break

            # update the estimate of Lagrangian multiplier
            lam += c * g_plus

        # check if not solved
        if not solved:
            print(f"WARNING: not solved after iter #{self.k_max}.")

        print(f"Total time for inner solver: {t_exec:.1e} seconds.")

        return x_plus, t_exec, lam
