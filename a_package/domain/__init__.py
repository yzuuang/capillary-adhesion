"""
Domain module: computational domain and data structures.

Provides:
- Grid: spatial discretization (will support domain decomposition)
- Field: data living on the grid
- NpyIO: parallel-aware persistence (future)
"""

from .grid import Grid
from .field import Field, adapt_shape, field_component_ax, field_element_axs
from .io import NpyIO
