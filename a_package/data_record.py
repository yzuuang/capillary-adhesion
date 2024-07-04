import dataclasses as dc
import pickle
import uuid
import numpy as np


# WARN: modifying these data classes usually implies to update all the data records!
# TODO: how to overcome that? (dict? json?)
# TODO: how to include post-process values (e.g. forces) ?


@dc.dataclass
class DropletData:
    V: float         # volume of the droplet
    eta: float       # interfacial width
    L: float         # length of the plate
    M: int           # number of pixels along x-axis
    N: int           # number of pixels along y-axis
    phi: np.ndarray  # phase-field in 2D-array
    h1: np.ndarray   # roughness of the 1 plate in 2D-array
    h2: np.ndarray   # roughness of the 2 plate in 2D-array
    d: float         # displacement between the baselines of two plates
    x: np.ndarray    # grid locations along x-axis in 1D-array
    y: np.ndarray    # grid locations along y-axis in 1D-array
    dx: float        # size of pixels along x-axis
    dy: float        # size of pixels along y-axis


@dc.dataclass
class Record:
    data: list[DropletData]  # think as if taking snapshots
    init_guess: np.ndarray
    ID: str = dc.field(default_factory=uuid.uuid4)  # for the sake of a unique ID


def save_record(r: Record, filepath):
    with open(filepath, "wb") as fp:
        pickle.dump(r, fp)


def load_record(filepath):
    with open(filepath, "rb") as fp:
        r:Record = pickle.load(fp)
    return r