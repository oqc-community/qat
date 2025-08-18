# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import numpy as np

from qat.purr.utils.serializer import json_dumps, json_loads


class TestQatJsonSerializer:
    def test_numpy_array(self):
        arr = np.random.rand(80) + 1j * np.random.rand(80)
        serialized_arr = json_dumps(arr)
        assert isinstance(serialized_arr, str)
        deserialized_arr = json_loads(serialized_arr)
        assert isinstance(deserialized_arr, np.ndarray)
        assert np.allclose(arr, deserialized_arr)
