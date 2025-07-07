"""This file addreses solving optimization problem.

- Approximate a constrained optimization with an unconstrained problem using augmented Lagrangian 
method.
- Iterative approaches to find numerically the minimizer of an unconstrained problem.
"""

import dataclasses as dc
import timeit
import typing as _t

import numpy as np
from mpi4py import MPI
from NuMPI.Optimization import l_bfgs
# from scipy.optimize import fmin_l_bfgs_b


@dc.dataclass
class AugmentedLagrangian:
    tolerance_convergence: float
    tolerance_constraint: float
    max_iterations: int
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
            "I.evol".format("%8s"),
            "N.iter".format("%8s"),
            "T.exec".format("%8s"),
            "f".format("%8s"),
            "f_grad".format("%8s"),
            "g".format("%8s"),
            "l".format("%8s"),
            "l_grad".format("%8s"),
        ]
        print(*tabel_header)

        # initial setup
        x_plus = x0
        lam = 0
        t_exec = -timeit.default_timer()
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
                # ftol=1e-3,
                gtol=self.tolerance_convergence,
                maxiter=self.max_iterations,
                comm=MPI.COMM_WORLD,
            )
            t_exec_sub += timeit.default_timer()
            nit = res['nit']
            l_plus = res['fun'].item()
            l_norm_grad = res['norm_grad']
            x_plus = res['x']

            # [x_plus, l_plus, res] = fmin_l_bfgs_b(
            #     self.l,
            #     # old solution as new initial guess
            #     x_plus,
            #     # bounds=[(0, 1)] * len(x_plus),
            #     fprime=self.dx_l,
            #     args=(lam, c),
            #     factr=1e1,
            #     pgtol=self.tolerance_convergence,
            #     maxiter=self.max_iter,
            # )
            # t_exec_sub += timeit.default_timer()
            # t_exec += t_exec_sub
            # l_grad = res['grad']


            f_plus = self.f(x_plus).item()
            f_grad = self.f_D_x(x_plus)
            f_norm_grad = np.sqrt(np.sum(f_grad**2))
            g_plus = self.g(x_plus).item()
            # lagr1 = f_plus - lam * g_plus

            # Print
            tabel_entry = [
                f"#{k:4d}",
                f"{nit:4d}",
                f"{round(t_exec_sub, 2):4.2f}s",
                f"{f_plus: 8.2e}",
                f"{f_norm_grad: 8.2e}",
                f"{g_plus: 8.2e}"
                f"{l_plus: 8.2e}",
                f"{l_norm_grad: 8.2e}",
            ]
            print(*tabel_entry, sep="  ")

            # convergence criteria
            if abs(g_plus) < self.tolerance_constraint:
                solved = True
                print(f"Achieved required tolerance at iter #{k}")
                break

            # update the estimate of Lagrangian multiplier
            lam -= c * g_plus

        print(res)

        # check if not solved
        if not solved:
            print(f"WARNING: not solved after iter #{self.k_max}.")

        t_exec += timeit.default_timer()
        print(f"Total time to solve: {t_exec:.1e} seconds.")

        return x_plus, t_exec, lam
