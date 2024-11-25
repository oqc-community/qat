from __future__ import annotations

from typing import Annotated, TypeVar, get_args

import numpy as np
from pydantic import AfterValidator, RootModel

# This base file is used to implement classes/methods common to all hardware.


def validate_non_negative(v: int):
    if v < 0:
        raise ValueError(f"Given value {v} must be an int and >=0.")
    return v


NonNegativeInt = Annotated[
    int,
    AfterValidator(validate_non_negative),
]

QubitId = NonNegativeInt


def validate_calibratable_positive_float(v: CalibratablePositiveFloat):
    if not np.isnan(v) and v < 0.0:
        raise ValueError(f"Given value {v} must be >=0.")
    return v


CalibratablePositiveFloat = Annotated[
    float,
    AfterValidator(validate_calibratable_positive_float),
]


def validate_calibratable_unit_interval(v: CalibratableUnitInterval):
    if not np.isnan(v):
        if v < 0.0 or v > 1.0:
            raise ValueError("Given value {v} must be in the interval [0, 1].")
    return v


CalibratableUnitInterval = Annotated[
    float,
    AfterValidator(validate_calibratable_unit_interval),
]

K = TypeVar("K")
V = TypeVar("V", CalibratablePositiveFloat, CalibratableUnitInterval)


def get_validator_from_annotated(annotated_type):
    metadata = get_args(annotated_type)
    for item in metadata:
        if isinstance(item, AfterValidator):
            return item.func
    return None


class ValidatedSet(RootModel[set[V]]):
    root: set[V]

    def add(self, value: V):
        field_type = self.model_fields["root"].annotation
        f_validate = get_validator_from_annotated(get_args(field_type)[0])

        value = f_validate(value)
        self.root.add(value)

    def discard(self, value):
        self.root.discard(value)

    def remove(self, value):
        self.root.remove(value)

    def __iter__(self):
        return iter(self.root)

    def __eq__(self, other: ValidatedSet):
        if isinstance(other, ValidatedSet):
            return self.root.__eq__(other.root)
        elif isinstance(other, set):
            return self.root.__eq__(other)
        return False

    def __le__(self, other: ValidatedSet):
        if isinstance(other, ValidatedSet):
            return self.root.issubset(other.root)
        elif isinstance(other, set):
            return self.root.issubset(other)
        else:
            raise NotImplemented(
                f"Unsupported processing for `ValidatedSet`s {self} and {other}."
            )

    def __repr__(self):
        return self.root.__repr__()


class ValidatedDict(RootModel[dict[K, V]]):
    root: dict[K, V]

    def __setitem__(self, key, value):
        field_type = self.model_fields["root"].annotation
        f_validate = get_validator_from_annotated(get_args(field_type)[1])

        value = f_validate(value)
        self.root[key] = value

    def __getitem__(self, key):
        return self.root.get(key, None)

    def get(self, key, default=None):
        return self.root.get(key, default)

    def keys(self):
        return self.root.keys()

    def values(self):
        return self.root.values()

    def __iter__(self):
        return iter(self.root)

    def items(self):
        return self.root.items()

    def __eq__(self, other: ValidatedDict):
        if isinstance(other, ValidatedDict):
            return self.root.__eq__(other.root)
        elif isinstance(other, dict):
            return self.root.__eq__(other)
        return False

    def update(self, data: dict[K, V]):
        for key, value in data.items():
            self.__setitem__(key, value)

    def __len__(self):
        return self.root.__len__()

    def __repr__(self):
        return self.root.__repr__()
