import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cProfile
import timeit
import numpy as np
from threadpoolctl import threadpool_limits

from a_package.domain import Grid
from a_package.simulation.formulation import NodalFormCapillary
from a_package.numerics.quadrature import centroid_quadrature


domain_lengths = [1., 1.]
nb_elements = [64, 64]

grid = Grid(domain_lengths, nb_elements)
capillary_args = dict(eta=0.01, theta=np.pi * (45 / 180))
capi = NodalFormCapillary(grid, capillary_args, centroid_quadrature)

[x, y] = grid.form_nodal_mesh()
gap = 0.005 * ((x - 0.5 * domain_lengths[0])**2 + (y - 0.5 * domain_lengths[1])**2) 
capi.set_gap(gap)

phase_init = 0.25 * np.ones([1, 1, *nb_elements], dtype=float)
nb_loops = 2000

with cProfile.Profile() as pr:
    for _ in range(nb_loops):
        capi.set_phase(phase_init)
        energy = capi.get_energy()
    pr.print_stats(sort="cumulative")
print(f"E={energy}")


# t_exec = -timeit.default_timer()
# for _ in range(nb_loops):
#     capi.set_phase(phase_init)
#     capi.get_energy()
# t_exec += timeit.default_timer()
# print(f"Executing {nb_loops} loops takes {t_exec:.3f} seconds.")


# test_store_dir = os.path.join(os.path.dirname(__file__), "cache")
# optimizer_args = dict(max_inner_iter=1000, max_outer_loop=1)

# fem = FirstOrderElement(grid, centroid_quadrature.quad_pt_coords)
# volume = 1e-4

# separation_trajectory = np.array([0.1])