"""
Simulation routines.
"""

import datetime
import math

import numpy as np

from a_package.modelling import CapillaryBridge
from a_package.solving import AugmentedLagrangian
from a_package.storing import save
from a_package.data_record import SimulationResult


def sim_quasi_static_pull_push(capi: CapillaryBridge, solver: AugmentedLagrangian, V: float, d_min: float, d_max: float, d_step: float):
    # generate all d (mean distances)
    num_d = math.floor((d_max - d_min) / d_step) + 1
    d_departing = d_min + d_step * np.arange(num_d)
    d_approaching = d_max - d_step * np.arange(num_d)
    all_d = np.concatenate((d_departing, d_approaching))

    # use timestamp to differ paths for storing data
    timestamp = datetime.datetime.now(tz=datetime.timezone.utc).isoformat(timespec="seconds")
    path = __file__.replace(".py", f"---{timestamp}")

    # inform
    print(f"Problem size: {capi.region.nx}x{capi.region.ny}. "
          f"Simulating for all {len(all_d)} mean distance values in...\n{all_d}")
    print()

    # simulate
    for d in all_d:
        # update the parameter (the contact will be checked)
        print(f"Parameter of interest: mean distance={d}")
        capi.update_separation(d)

        # clean the phase field where the solid bodies contact
        capi.phi[capi.at_contact] = 0.

        # solve the problem
        numopt = capi.formulate_with_constant_volume(V)
        capi.phi[:], t_exec = solver.solve_minimisation(numopt, capi.phi)
        capi.validate_phase_field()

        # save the result
        data = SimulationResult(capi.eta, capi.phi, t_exec)
        save(path, f"droplet---d={d}", data)

    # save configurations
    save(path, "solving", solver)
