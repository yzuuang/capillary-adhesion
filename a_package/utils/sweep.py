import functools
import itertools
import operator

import numpy as np


class Sweep:
    """Sweeping through a range of values for parameters"""

    def __init__(self, sweep_specs: dict[tuple[str, str], np.ndarray]):
        """
        - sweep_specs: Each mapping represents a sweep in a certain parameter. The "key" is 
        a pair of names to access the parameter value in the config; and the "mapped value" is 
        an array of all values one wants to vary in that specific parameter.
        """
        self._specs = sweep_specs

    def __len__(self):
        return (
            0
            if not len(self._specs)
            else functools.reduce(operator.mul, (len(vals) for vals in self._specs.values()), 1)
        )

    def iter_config(self, config: dict[str, dict[str, str]]):
        """
        - config: a nested dict, exactly how one can access the values after using python's 
        ConfigParser to read parameters from files.
        """
        for updates in self._iter_combos():
            for [key_pair, value] in updates:
                config[key_pair[0]][key_pair[1]] = str(value)
            yield config

    def _iter_combos(self):
        keys = list(self._specs.keys())
        # get the combinations by outer product
        values_combos = itertools.product(*(self._specs.values()))
        for values_combo in values_combos:
            yield zip(keys, values_combo)
