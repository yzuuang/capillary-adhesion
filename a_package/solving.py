import timeit
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from .droplet import CapillaryDroplet


tol_min = 1e-8
max_iter = 2000
tol_lam = 1e-4
beta = 3.0
c_max_over_mean = 1e5


# NOTE: keep the same formulation of volume constraint, h := V(phi, g) - V*
def solve_augmented_lagrangian(phi_init: np.ndarray, droplet: CapillaryDroplet, V: float):
    lam = 0.0
    c = 1e-2
    c_max = c_max_over_mean * c

    t_exec = 0 - timeit.default_timer()
    for i in range(20):
        # late termination when max penalty weight achieved
        if c > c_max:
            print("Notice: maximal penalty weight achieved!")
            break

        # solve minimization problem
        [x, f, d] = fmin_l_bfgs_b(
            augmented_lagrangian,
            phi_init.ravel(),
            args=(lam, c, droplet, V),
            fprime=augmented_lagrangian_jacobian,
            factr=1e1,
            pgtol=tol_min,
            maxiter=max_iter,
	        # disp=True,
        )
        print(f"lam={lam:.2e}, c={c:.2e}, {d['task']}")

        # earlier termination when the tolerance for lambda is achieved
        err_lam = c * (droplet.compute_volume() - V)
        if abs(err_lam) < tol_lam:
            print("Notice: required tol achieved!")
            break

        # update lagrangian multiplier
        lam += err_lam
        c *= beta

    # execution time
    t_exec += timeit.default_timer()
    print(f"Solver return after {i+1:d} iters, {t_exec:.2e} secs")

    # validate
    droplet.phi = np.reshape(x, (droplet.M, droplet.N))
    validate_constraints(droplet, V)
    print()

    return np.ravel(x)


def augmented_lagrangian(phi_flat: np.ndarray, lam: float, c: float, droplet: CapillaryDroplet, V: float):
    droplet.update_phase_field(phi_flat)
    h = droplet.compute_volume() - V
    return droplet.compute_energy() + lam*h + 0.5*c*(h**2)


def augmented_lagrangian_jacobian(phi_flat: np.ndarray, lam: float, c: float, droplet: CapillaryDroplet, V: float):
    droplet.update_phase_field(phi_flat)
    h = droplet.compute_volume() - V
    h_x = droplet.compute_volume_jacobian()
    return (droplet.compute_energy_jacobian() + (lam + c*h)*h_x).ravel()


def validate_constraints(droplet: CapillaryDroplet, V: float):
    # volume conservation

    new_V = droplet.compute_volume()
    if not np.isclose(new_V, V):
        print(f"Notice: volume is not conserved, with a diff of {new_V - V:+.2e}")

    # phase field < 0
    if np.any(droplet.phi < 0):
        outlier = np.where(droplet.phi < 0, droplet.phi, np.nan)
        count = np.count_nonzero(~np.isnan(outlier))
        extreme = np.nanmin(outlier)
        print(f"Notice: phase field has {count} values < 0, min at {extreme:.2e}")

    # phase field > 1
    if np.any(droplet.phi > 1):
        outlier = np.where(droplet.phi > 1, droplet.phi, np.nan)
        count = np.count_nonzero(~np.isnan(outlier))
        extreme = np.nanmax(outlier)
        print(f"Notice: phase field has {count} values > 1, max at 1.0+{extreme-1:.2e}.")
