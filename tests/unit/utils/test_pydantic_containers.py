# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
import random
from copy import deepcopy

import numpy as np
import pytest
from numpydantic import NDArray, Shape
from pydantic import ValidationError

from qat.utils.pydantic import (
    CalibratablePositiveFloat,
    CalibratableUnitInterval,
    FrozenDict,
    FrozenSet,
    PydDictBase,
    PydListBase,
    PydSetBase,
    QubitId,
    ValidatedDict,
    ValidatedSet,
)


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
class TestBaseContainer:
    def test_equals_list(self, seed):
        random_floats1 = [random.Random(seed).uniform(0.01, 1) for _ in range(10)]
        random_floats2 = deepcopy(random_floats1)
        assert PydListBase(random_floats1) == PydListBase(random_floats2)
        assert PydListBase(random_floats1) == random_floats2

        random_floats2[0] = -random_floats2[0]
        assert PydListBase(random_floats1) != PydListBase(random_floats2)
        assert PydListBase(random_floats1) != random_floats2

    def test_equals_set(self, seed):
        random_floats1 = set([random.Random(seed).uniform(0.01, 1) for _ in range(10)])
        random_floats2 = deepcopy(random_floats1)
        assert PydSetBase(random_floats1) == PydSetBase(random_floats2)
        assert PydSetBase(random_floats1) == random_floats2

        removed_float = random.Random(seed).sample(list(random_floats2), 1)[0]
        random_floats2.remove(removed_float)
        assert PydSetBase(random_floats1) != PydSetBase(random_floats2)
        assert PydSetBase(random_floats1) != random_floats2

        random_floats2.add(-removed_float)
        assert PydSetBase(random_floats1) != PydSetBase(random_floats2)
        assert PydSetBase(random_floats1) != random_floats2

    def test_equals_dict(self, seed):
        random_map1 = {
            random.Random(seed).uniform(0.01, 1): random.Random(seed).uniform(0.01, 1) + 10
            for _ in range(10)
        }
        random_map2 = deepcopy(random_map1)
        assert PydDictBase(random_map1) == PydDictBase(random_map2)
        assert PydDictBase(random_map1) == random_map2

        removed_key = random.Random(seed).sample(list(random_map2), 1)[0]
        removed_value = random_map2[removed_key]
        random_map2.pop(removed_key)
        assert PydDictBase(random_map1) != PydDictBase(random_map2)
        assert PydDictBase(random_map1) != random_map2

        random_map2[removed_key] = -removed_value
        assert PydDictBase(random_map1) != PydDictBase(random_map2)
        assert PydDictBase(random_map1) != random_map2


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


@pytest.mark.parametrize("array_type", [float, int, complex])
class TestNDArray:
    a = np.random.rand(2, 3, 4)  # 3-dimensional tensor
    b = np.random.rand(2, 3, 4)  # 3-dimensional tensor

    def test_dict_multi_dimensional_array(self, array_type):
        a = self.a.astype(dtype=array_type)
        b = self.b.astype(dtype=array_type)
        NDArray_annot = NDArray[Shape["* x, * y, * z"], array_type]

        d = PydDictBase[str, NDArray_annot]({"a": a, "b": b})
        assert np.allclose(d["a"], a)
        assert np.allclose(d["b"], b)

        blob = d.model_dump()
        d_deserialised = PydDictBase[str, NDArray_annot].model_validate(blob)
        assert np.allclose(d["a"], d_deserialised["a"])
        assert np.allclose(d["b"], d_deserialised["b"])

        with pytest.raises(ValidationError):
            PydDictBase[int, NDArray_annot].model_validate(blob)

    def test_list_multi_dimensional_array(self, array_type):
        a = self.a.astype(dtype=array_type)
        b = self.b.astype(dtype=array_type)
        NDArray_annot = NDArray[Shape["* x, * y, * z"], array_type]

        l = PydListBase[NDArray_annot]([a, b])
        assert np.allclose(l[0], a)
        assert np.allclose(l[1], b)

        blob = l.model_dump()
        l_deserialised = PydListBase[NDArray_annot].model_validate(blob)
        assert np.allclose(l[0], l_deserialised[0])
        assert np.allclose(l[1], l_deserialised[1])

        with pytest.raises(ValidationError):
            PydListBase[str].model_validate(blob)
