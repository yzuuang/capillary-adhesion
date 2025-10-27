import logging
import dataclasses as dc

import numpy as np

from a_package.workflow.formulation import NodalFormCapillary 
from a_package.workflow.simulation import SimulationResult
from a_package.numeric import AugmentedLagrangian


logger = logging.getLogger(__name__)


@dc.dataclass
class Evolution:
    nb_steps: int
    g: list[np.ndarray]
    phi: list[np.ndarray]
    r: np.ndarray       # relative displacement
    E: np.ndarray       # energy
    # HACK: skip this "troublemaking" value
    # p: np.ndarray       # presure
    V: np.ndarray       # volume
    P: np.ndarray       # perimeter
    Fz: np.ndarray      # normal force


@dc.dataclass
class ProcessedResult:
    formulating: NodalFormCapillary
    minimising: AugmentedLagrangian
    evolution: Evolution


def post_process(res: SimulationResult):
    # allocate memory
    nb_steps = len(res.steps)
    nb_spatial_dims = 3
    g = []
    phi = []
    r = np.empty((nb_steps, nb_spatial_dims))  
    E = np.empty((nb_steps))               
    p = np.empty((nb_steps))             
    V = np.empty((nb_steps))               
    P = np.empty((nb_steps))               

    # use the model for computing extra quantities
    fmltn = res.formulating

    # Convert data "rows" to "columns"
    for index, step in enumerate(res.steps):
        phi.append(step.phi)

        g.append(fmltn.get_gap(step.d).squeeze())

        fmltn.set_gap(step.d)
        fmltn.set_phase(step.phi)

        r[index] = [step.m[0], step.m[1], step.d]
        E[index] = fmltn.get_energy()
        p[index] = step.lam
        V[index] = fmltn.get_volume()
        P[index] = fmltn.get_perimeter()

    # get normal force by numerical differences of energy
    Fz = -(E[1:] - E[:-1]) / (r[1:, 2] - r[:-1, 2])

    # HACK: skip this "troublemaking" value
    # evo = Evolution(g, phi, r, E, p, V, P, Fz)
    # pack in an object
    evo = Evolution(nb_steps, g, phi, r, E, V, P, Fz)
    return ProcessedResult(res.formulating, res.minimising, evo)
