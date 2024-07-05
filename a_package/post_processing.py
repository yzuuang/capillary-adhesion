import numpy as np

from a_package.droplet import QuadratureRoughDroplet
from a_package.data_record import DropletData


def compute_energy(many_data: list[DropletData]):
    for data in many_data:
        droplet = QuadratureRoughDroplet(
            data.phi, data.h1, data.h2, data.d, data.eta, data.M, data.N, data.dx, data.dy
        )
        droplet.update_separation(data.d)
        droplet.update_phase_field(data.phi.ravel())
        # TODO: better data_record?
        data.E = droplet.compute_energy()


def compute_force(many_data: list[DropletData]):
    for data in many_data:
        droplet = QuadratureRoughDroplet(
            data.phi, data.h1, data.h2, data.d, data.eta, data.M, data.N, data.dx, data.dy
        )
        droplet.update_separation(data.d)
        droplet.update_phase_field(data.phi.ravel())
        # TODO: better data_record?
        data.F = droplet.compute_force()
