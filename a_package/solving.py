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
    inner_max_iter: int
    tol_convergence: float
    tol_constraint: float
    c_init: float
    c_upper_bound: float
    beta: float

    def solve_minimisation(self, numopt: NumOptEq, x0: np.ndarray):
        # compute all possible `c` values, i.e. for(c=c_init; c<c_upper_bound; c*=beta)
        num_iter = int(np.log(self.c_upper_bound / self.c_init) / np.log(self.beta)) + 1
        cc = self.c_init * np.pow(self.beta, np.arange(num_iter))

        # initial setup
        x_plus = x0
        lam = 0
        t_exec = 0

        # inform
        tabel_header = ['Iter', 'T_exec', 'Objective', 'lambda\t', 'Lagrangian', 'c\t', 'Augm. Lagr.',
                        'INFO']
        print(*tabel_header, sep='\t')

        for k, c in enumerate(cc):
            # derive augmented lagrangian
            def l(x: np.ndarray):
                """Augmented Lagrangian."""
                g_x = numopt.g(x)
                return numopt.f(x) + lam * g_x + (0.5 * c) * g_x ** 2

            def l_grad(x: np.ndarray):
                """Gradient of the Augmented Lagrangian."""
                return numopt.f_grad(x) + (lam + c * numopt.g(x)) * numopt.g_grad(x)

            # solve minimization problem
            t_exec_sub = -timeit.default_timer()
            [x_plus, l_plus, info] = optimize.fmin_l_bfgs_b(
                l,
                x_plus,  # old solution as new initial guess
                bounds=[(0, 1)] * len(x_plus),
                fprime=l_grad,
                factr=1e1,  # for extremely high accuracy
                pgtol=self.tol_convergence,
                maxiter=self.inner_max_iter,
            )
            t_exec_sub += timeit.default_timer()
            t_exec += t_exec_sub

            # inform
            f_plus = numopt.f(x_plus)
            error_g_x = numopt.g(x_plus)
            lagr1 = f_plus + lam * error_g_x

            info['max_grad'] = max(info['grad'])
            del info['grad']

            tabel_entry = [f"#{k}", f"{round(t_exec_sub, 2):.2f}s", f"{f_plus:.2e}", f"{lam:.2e}",
                           f"{lagr1:.2e}", f"{c:.2e}", f"{l_plus:.2e}", f"{info}"]
            print('\t'.join(tabel_entry))

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
