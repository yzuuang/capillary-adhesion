"""Use NumPy array to store field data. The conventions on the array dimensions are specified.
"""
import sys
if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

import numpy as np

Field: TypeAlias = np.ndarray[tuple[int, int, int, int]]
field_component_ax = 0
field_sub_pt_ax = 1
field_element_axs = (2, 3)


if sys.version_info >= (3, 10):
    def adapt_shape(array: np.ndarray) -> Field:
        match(np.ndim(array)):
            case 2:
                return np.expand_dims(array, axis=(field_component_ax, field_sub_pt_ax))
            case 3:
                return np.expand_dims(array, axis=field_sub_pt_ax)
            case 4:
                return array
            case _:
                raise ValueError(f"Support array of 2/3/4D")
else:
    def adapt_shape(array: np.ndarray):
        ndim = np.ndim(array)
        if ndim == 2:
            return np.expand_dims(array, axis=(field_component_ax, field_sub_pt_ax))
        if ndim == 3:
            return np.expand_dims(array, axis=field_sub_pt_ax)
        if ndim == 4:
            return array
        else:
            raise ValueError(f"Support array of 2/3/4D")
