"""Use NumPy array to store field data. The conventions on the array dimensions are specified.
"""

import typing

import numpy as np

Field: typing.TypeAlias = np.ndarray[tuple[int, int, int, int]]
field_component_ax = 0
field_sub_pt_ax = 1
field_element_axs = (2, 3)
