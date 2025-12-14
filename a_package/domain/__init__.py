"""
Domain module: spatial foundation and discretization.

Provides:
- Grid: spatial discretization (will support domain decomposition)
- Field: data living on the grid
- NpyIO: parallel-aware persistence (future)
- FEM: finite element interpolation
- Quadrature: integration rules
"""

from .grid import Grid
from .field import Field, adapt_shape, field_component_ax, field_element_axs
from .io import NpyIO
from .fem import FirstOrderElement
from .quadrature import Quadrature, centroid_quadrature
