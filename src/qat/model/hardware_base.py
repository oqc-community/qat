from __future__ import annotations

from typing import Annotated, TypeVar, get_args

import numpy as np
from annotated_types import Predicate
from pydantic import NonNegativeInt, RootModel

# This base file is used to implement classes/methods common to all hardware.

QubitId = NonNegativeInt


def validate_calibratable_positive_float(v: CalibratablePositiveFloat):
    if not np.isnan(v) and v < 0.0:
        raise ValueError(f"Given value {v} must be >=0.")
    return v


CalibratablePositiveFloat = Annotated[
    float,
    Predicate(validate_calibratable_positive_float),
    validate_calibratable_positive_float,
]


def validate_calibratable_unit_interval(v: CalibratableUnitInterval):
    if not np.isnan(v):
        if v < 0.0 or v > 1.0:
            raise ValueError("Given value {v} must be in the interval [0, 1].")
    return v


CalibratableUnitInterval = Annotated[
    float,
    Predicate(validate_calibratable_unit_interval),
    validate_calibratable_unit_interval,
]

K = TypeVar("K")
V = TypeVar("V", CalibratablePositiveFloat, CalibratableUnitInterval)


class ValidatedDict(RootModel[dict[K, V]]):
    root: dict[K, V]

    def __setitem__(self, key, value):
        field_type = self.model_fields["root"].annotation
        f_validate = get_args(get_args(field_type)[1])[-1]

        value = f_validate(value)
        self.root[key] = value

    def __getitem__(self, key):
        return self.root.get(key, None)

    def __iter__(self):
        return iter(self.root)

    def __eq__(self, other: ValidatedDict):
        return self.root.__eq__(other)

    def update(self, data: dict[K, V]):
        for key, value in data.items():
            self.__setitem__(key, value)

    def __repr__(self):
        return self.root.__repr__()
