import copy
import math
import numpy as np

from a_package.data_record import DropletData, Record
from a_package.droplet import QuadratureRoughDroplet
from a_package.solving import solve_augmented_lagrangian


def sim_quasi_static_pull_push(data: DropletData, phi_init: np.ndarray, d_min: float, d_max: float, d_step: float):
    # generate all d (mean distances)
    num_d = math.ceil((d_max - d_min) / d_step)
    d_departing = d_min + d_step * np.arange(num_d)
    d_approaching = d_max - d_step * np.arange(num_d)
    all_d = np.concatenate((d_departing, d_approaching))

    # prepare objects
    phi_flat = phi_init.copy().ravel()
    droplet = QuadratureRoughDroplet(phi_flat, data.h1, data.h2, all_d[0], data.eta, data.M, data.N, data.dx, data.dy)

    # inform
    print(f"Problem size: {data.M}x{data.N}. Simulating for all {len(all_d)} mean distance values in...\n{all_d}")
    print()

    # simulate
    results = []
    for d in all_d:
        print(f"Parameter of interest: mean distance={d}")
        # update the parameter (the contact will be checked)
        droplet.update_separation(d)
        # clean the phase field where the plates contact
        phi_flat[droplet.at_contact] = 0
        # solve the problem
        sol = solve_augmented_lagrangian(phi_flat, droplet, data.V)
        # save the result
        data.phi = np.reshape(sol, (data.M, data.N))
        data.d = d
        results.append(copy.deepcopy(data))
        # apply recursive initials
        phi_flat[:] = sol

    return Record(results, phi_init)
