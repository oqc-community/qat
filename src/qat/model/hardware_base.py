from __future__ import annotations

import numpy as np
from pydantic import NonNegativeInt
from pydantic_core import core_schema

# This base file is used to implement classes/methods common to all hardware.

QubitId = NonNegativeInt


class CalibratablePositiveFloat(float):
    """
    A calibratable float whose value must be >=0.
    """

    @classmethod
    def validate(cls, v):
        if not np.isnan(v) and v < 0.0:
            raise ValueError(f"Given value {v} must be >=0.")
        return cls(v)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(cls.validate)

    def __eq__(self, other: CalibratablePositiveFloat):
        if np.isnan(self) and np.isnan(other):
            return True
        else:
            return super().__eq__(other)


class CalibratableUnitInterval(CalibratablePositiveFloat):
    """
    A calibratable float whose value must lie in the interval [0, 1].
    """

    @classmethod
    def validate(cls, v):
        super().validate(v)

        if not np.isnan(v) and v > 1.0:
            raise ValueError("Given value {v} must be <= 1.")
        return cls(v)
