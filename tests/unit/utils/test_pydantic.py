# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from numpydantic import NDArray, Shape
from pydantic import BaseModel, ValidationError

from qat.utils.pydantic import (
    CalibratablePositiveFloat,
    CalibratableUnitInterval,
    ComplexNDArray,
    FloatNDArray,
    FrozenDict,
    FrozenSet,
    IntNDArray,
    PydArray,
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
        s = ValidatedSet[CalibratablePositiveFloat]()

        valid_value = random.Random(seed).uniform(0, 1e10)
        s.add(valid_value)

        invalid_value = random.Random(seed).uniform(-1e-06, -1e10)
        with pytest.raises(ValueError):
            s.add(invalid_value)

        with pytest.raises(ValueError):
            ValidatedSet[CalibratablePositiveFloat]({-1.3})

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
        d = ValidatedDict[QubitId, CalibratableUnitInterval]()

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

        with pytest.raises(ValidationError):
            ValidatedDict[QubitId, CalibratableUnitInterval](
                {"a": 0.5}
            )  # Key is not a QubitId

        with pytest.raises(ValueError):
            ValidatedDict[QubitId, CalibratableUnitInterval](
                {0: invalid_value}
            )  # Value is not within [0, 1]


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

        base_list = PydListBase[NDArray_annot]([a, b])
        assert np.allclose(base_list[0], a)
        assert np.allclose(base_list[1], b)

        blob = base_list.model_dump()
        deserialised_list = PydListBase[NDArray_annot].model_validate(blob)
        assert np.allclose(base_list[0], deserialised_list[0])
        assert np.allclose(base_list[1], deserialised_list[1])

        with pytest.raises(ValidationError):
            PydListBase[str].model_validate(blob)


class TestPydArray:
    def test_empty_array(self):
        arr = PydArray(value=np.array([]))
        assert arr.value.size == 0
        assert arr.value.shape == (0,)

    def test_1d_value(self):
        a = np.random.rand(3)
        b = np.random.rand(5)

        arr = PydArray(value=a)
        assert np.allclose(arr.value, a)

        arr.value = b
        assert np.allclose(arr.value, b)

    def test_nd_value(self):
        a = np.random.rand(2, 3, 4)

        arr = PydArray(value=a)
        assert np.allclose(arr.value, a)

    def nameless_constructor(self):
        a = np.random.rand(2, 3, 4)

        arr = PydArray(a)
        assert np.allclose(arr.value, a)

    def test_equals(self):
        a = np.random.rand(4, 5, 6, 7)

        arr1 = PydArray(value=a)
        arr2 = PydArray(value=deepcopy(a))

        assert arr1 == arr2
        assert arr1 == a

        arr2 += 1e-03
        assert arr1 != arr2
        assert arr1 != a + 1e-03

    def test_empty_array_serialisation(self):
        arr = PydArray(value=np.array([]))

        blob = arr.model_dump()
        arr_deserialised = PydArray.model_validate(blob)
        assert arr_deserialised.value.size == 0
        assert arr_deserialised.value.shape == (0,)
        assert arr_deserialised == arr

    def test_serialisation(self):
        arr = PydArray(np.random.rand(2, 3, 4))

        blob = arr.model_dump()
        arr_deserialised = PydArray.model_validate(blob)
        assert np.allclose(arr_deserialised.value, arr.value)
        assert arr_deserialised == arr

    # TODO Model-level validator COMPILER-769
    @pytest.mark.skip(reason="Needs model-level validator COMPILER-769")
    @pytest.mark.parametrize("pyd_array", ["pyd_array_1.json", "pyd_array_2.json"])
    def test_serialisation_json(self, testpath, pyd_array):
        file_path = Path(testpath, "files", "payload", pyd_array)
        with open(file_path, "r") as f:
            arr = IntNDArray.model_validate_json(f.read())

        assert arr.shape == (1000,)

    def test_list_input(self):
        arr = PydArray([1.0, 2.0, 3.0])
        assert isinstance(arr.value, np.ndarray)
        assert arr.value.shape == (3,)

    @pytest.mark.parametrize(
        "arr_type,lst",
        [
            (ComplexNDArray, [1.0 + 3.2j, 2.0 + 4.2j, 3.0 + 5.2j]),
            (FloatNDArray, [1.0, 2.0, 3.0]),
            (IntNDArray, [1, 2, 3]),
        ],
    )
    def test_typed_arrays(self, arr_type, lst):
        arr = arr_type(lst)
        assert isinstance(arr.value, np.ndarray)
        assert arr.value.shape == (len(lst),)

        arr2 = arr_type(value=deepcopy(lst))
        assert arr == arr2

        arr2[0] += 1
        assert arr != arr2

    def test_wrong_type(self):
        with pytest.raises(ValidationError):
            IntNDArray(["a", "b", "c"])

        # Downconversion is fine.
        class MockContainer(BaseModel):
            a: IntNDArray

        m = MockContainer(a=[1.0, 2.0, 3.0])
        assert m.a.dtype == np.dtype(int)

        # Upconversion too!
        class MockContainer(BaseModel):
            a: ComplexNDArray

        m = MockContainer(a=[1.0, 2.0, 3.0])
        assert m.a.dtype == np.dtype(complex)


class TestPydArrayNumpyInteroperability:
    """
    Tests interoperability between PydArray and Numpy arrays. If an operator isn't tested
    or the test is skipped, it's safe to assume that support for that operator is missing.
    """

    def test_equality(self):
        """
        Equality tested according to how it's been defined, which not exactly the same
        as NumPy array equality.
        """

        arr0 = PydArray(np.zeros(5))
        arr1 = PydArray(np.ones(5))
        assert arr0 == arr0
        assert arr0 != arr1

    def test_append(self):
        arr1 = PydArray(np.random.rand(5))
        arr2 = PydArray(np.random.rand(5))
        arr = np.append(arr1, arr2)
        assert arr.shape == (10,)
        assert np.all(arr[:5] == arr1)
        assert np.all(arr[5:] == arr2)

    # TODO - `PydArray` <> `np.ndarray` interoperability COMPILER-769
    @pytest.mark.skip(reason="`PydArray` <> `np.ndarray` interoperability COMPILER-769")
    def test_all(self):
        shape = (10, 2)

        arr0 = PydArray(np.zeros(shape))
        arr1 = PydArray(np.ones(shape))
        assert np.all(arr0 == 0)
        assert np.all(arr1 == 1)
        assert np.all(arr1 + arr0 == arr1)

    # TODO - `PydArray` <> `np.ndarray` interoperability COMPILER-769
    @pytest.mark.skip(reason="`PydArray` <> `np.ndarray` interoperability COMPILER-769")
    def test_any(self):
        shape = (5, 10)

        arr0 = PydArray(np.zeros(shape))
        arr1 = PydArray(np.ones(shape))
        i = np.random.choice(shape[0])
        j = np.random.choice(shape[1])
        random_value = np.random.random_sample()
        arr0[i, j] = random_value
        arr1[i, j] = random_value

        assert np.any(arr0 == random_value)
        assert np.any(arr1 == random_value)
