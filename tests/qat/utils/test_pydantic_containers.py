# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
import random

import pytest

from qat.model.device import QubitId
from qat.utils.pydantic import (
    CalibratablePositiveFloat,
    CalibratableUnitInterval,
    FrozenDict,
    FrozenSet,
    ValidatedDict,
    ValidatedSet,
)


@pytest.mark.parametrize("seed", [6, 7, 8, 9])
class TestValidatedContainer:
    def test_validated_calibratable_set(self, seed):
        s = ValidatedSet[CalibratablePositiveFloat](set())

        valid_value = random.Random(seed).uniform(0, 1e10)
        s.add(valid_value)

        invalid_value = random.Random(seed).uniform(-1e-06, -1e10)
        with pytest.raises(ValueError):
            s.add(invalid_value)

    def test_validated_qubitid_set(self, seed):
        s = ValidatedSet[QubitId]()

        valid_value = 64
        s.add(valid_value)

        invalid_value = -1  # Qubit indices are assumed to start at 0.
        with pytest.raises(ValueError):
            s.add(invalid_value)

        invalid_type = random.Random(seed).uniform(1e-03, 9e-03)
        with pytest.raises(TypeError):  # Type is float and should be int.
            s.add(invalid_type)

        with pytest.raises(TypeError):
            s.add("a")

    def test_validated_calibratable_dict(self, seed):
        d = ValidatedDict[QubitId, CalibratableUnitInterval]({})

        valid_value = random.Random(seed).uniform(0, 1)
        d[1] = valid_value
        d.update({2: valid_value})

        invalid_value = random.Random(seed).uniform(1.1, 10)
        with pytest.raises(ValueError):
            d[1] = invalid_value

        with pytest.raises(ValueError):
            d.update({2: invalid_value})

        with pytest.raises(TypeError):
            d[2] = "a"

        with pytest.raises(TypeError):
            d.update({2: "a"})


@pytest.mark.parametrize("seed", [10, 11, 12, 13])
class TestFrozenContainer:
    def test_frozen_set(self, seed):
        elements = random.Random(seed).sample(list(range(1, 100)), 32)
        s = FrozenSet[QubitId](set(elements))

        element = random.Random(seed).sample(list(range(1, 100)), 1)[0]
        with pytest.raises(AttributeError):
            s.add(element)

        with pytest.raises(AttributeError):
            s.pop()

        for element in s:
            with pytest.raises(AttributeError):
                s.discard(element)

            with pytest.raises(AttributeError):
                s.remove(element)

    def test_frozen_dict(self, seed):
        elements = {
            k: random.Random(seed).uniform(0, 100)
            for k in random.Random(seed).sample(list(range(1, 100)), 32)
        }
        d = FrozenDict[QubitId, CalibratablePositiveFloat](elements)

        element = random.Random(seed)
        with pytest.raises(TypeError):
            d[101] = element

        with pytest.raises(AttributeError):
            d.update({101: element})
