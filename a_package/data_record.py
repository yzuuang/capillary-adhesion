import dataclasses as dc
import typing as t_
import uuid
import numpy as np


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


# TODO: what to include in the record. The Dict returned from the inner solver shall also be dumped.
# TODO: how to include post-process values (e.g. forces) ?
# TODO: allow the data to be saved for each quasi-static state and possibly each solution to a inner problem of AL.

@dc.dataclass
class SimulationResult:
    eta: float
    phi: np.ndarray
    t_exec: float


# something make it easy for visualizing.plot_xxx() ?
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
