import dataclasses as dc
import numpy as np
import numpy.random as random

from a_package.data_store import *


_rng = random.default_rng()


@dc.dataclass
class SomeDataclass:
    literal: str
    numeric: float
    array: np.ndarray

    @classmethod
    def random(cls):
        size = 100
        return cls(str(_rng.random()), _rng.random(), _rng.random((size, size)))


@dc.dataclass
class NestedDataclass:
    record: list[SomeDataclass]


def assert_equal(loaded, saved):

    for field in dc.fields(saved):
        loaded_value = getattr(loaded, field.name)
        saved_value = getattr(saved, field.name)

        # nested dataclass
        if dc.is_dataclass(saved_value):
            assert_equal(loaded_value, saved_value)
        # list of nested dataclass
        elif isinstance(saved_value, list):
            for index, saved_element in enumerate(saved_value):
                if dc.is_dataclass(saved_element):
                    assert_equal(loaded_value[index], saved_element)
        # flat or NumPy array
        else:
            assert np.all(loaded_value == saved_value), f"The loaded {field.name} data is not the same as the saved."


_folder = __file__.replace(".py", "_folder")
_label = "this is created by test"


def test_save_load_flat():
    test_data = SomeDataclass.random()
    save(_folder, f"{_label} flat", test_data)
    loaded_test_data = load(_folder, f"{_label} flat", SomeDataclass)
    assert_equal(loaded_test_data, test_data)


def test_save_load_nested():
    test_data = NestedDataclass([SomeDataclass.random() for _ in range(11)])
    save(_folder, f"{_label} nested", test_data)
    loaded_test_data = load(_folder, f"{_label} nested", NestedDataclass)
    assert_equal(loaded_test_data, test_data)
