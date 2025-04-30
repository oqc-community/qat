# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd

import pytest
from pydantic import ValidationError

from qat.model.error_mitigation import ErrorMitigation, ReadoutMitigation
from qat.utils.hardware_model import generate_random_linear


@pytest.mark.parametrize("n_qubits", [1, 2, 4, 8, 31, 64])
@pytest.mark.parametrize("m3_available", [True, False])
class TestReadoutMitigation:
    def test_constructor(self, n_qubits, m3_available):
        qubit_indices = list(range(n_qubits))
        linear = generate_random_linear(qubit_indices)
        readout_mit = ReadoutMitigation(linear=linear, m3_available=m3_available)

        assert readout_mit.linear == linear
        assert readout_mit.m3_available == m3_available

    def test_invalid_linear_keys(self, n_qubits, m3_available):
        qubit_indices_str = ["Q" + str(i) for i in range(1, n_qubits + 1)]
        linear = generate_random_linear(qubit_indices_str)

        with pytest.raises(ValidationError):
            ReadoutMitigation(linear=linear, m3_available=m3_available)

    def test_invalid_matrix_elements(self, n_qubits, m3_available):
        qubit_indices = list(range(n_qubits))
        linear = generate_random_linear(qubit_indices)

        # Matrix elements must be in [0, 1].
        linear[0][0][0] = 1.1
        linear[0][1][0] = 0.9
        with pytest.raises(ValueError):
            ReadoutMitigation(linear=linear, m3_available=m3_available)

        linear[0][0][0] = -0.1
        linear[0][1][0] = 1.1
        with pytest.raises(ValueError):
            ReadoutMitigation(linear=linear, m3_available=m3_available)

        # Sum of columns must be equal to 1.
        linear[0][0][0] = 0.5
        linear[0][1][0] = 0.6
        with pytest.raises(ValidationError):
            ReadoutMitigation(linear=linear, m3_available=m3_available)

    def test_serialisation(self, n_qubits, m3_available):
        qubit_indices = list(range(n_qubits))
        linear = generate_random_linear(qubit_indices)
        readout_mit = ReadoutMitigation(linear=linear, m3_available=m3_available)
        blob = readout_mit.model_dump()

        readout_mit_deserialised = ReadoutMitigation(**blob)
        assert readout_mit == readout_mit_deserialised

    def test_dump_load_eq(self, n_qubits, m3_available):
        qubit_indices = list(range(n_qubits))
        linear = generate_random_linear(qubit_indices)
        readout_mit = ReadoutMitigation(linear=linear, m3_available=m3_available)
        blob = readout_mit.model_dump()

        readout_mit_deserialised = ReadoutMitigation(**blob)
        blob2 = readout_mit_deserialised.model_dump()
        assert blob == blob2

        readout_mit_deserialised.m3_available = not readout_mit_deserialised.m3_available
        blob3 = readout_mit_deserialised.model_dump()
        assert blob2 != blob3


class TestErrorMitigation:
    def test_default_constructor(self):
        error_mit = ErrorMitigation()
        assert error_mit.is_enabled is False

        with pytest.raises(AttributeError):
            error_mit.is_enabled = True

    @pytest.mark.parametrize("n_qubits", [1, 2, 4, 8, 31, 64])
    @pytest.mark.parametrize("m3_available", [True, False])
    def test_error_mit_enabled(self, n_qubits, m3_available):
        qubit_indices = list(range(n_qubits))
        linear = generate_random_linear(qubit_indices)
        readout_mit = ReadoutMitigation(linear=linear, m3_available=m3_available)
        error_mit = ErrorMitigation(readout_mitigation=readout_mit)

        assert error_mit.is_enabled

    def test_default_serialisation(self):
        error_mit1 = ErrorMitigation()
        blob1 = error_mit1.model_dump()

        error_mit2 = ErrorMitigation()
        blob2 = error_mit2.model_dump()

        assert blob1 == blob2

        error_mit1_deserialised = ErrorMitigation(**blob1)
        error_mit2_deserialised = ErrorMitigation(**blob2)
        assert error_mit1_deserialised == error_mit2_deserialised
        assert error_mit1_deserialised.is_enabled is False
        assert error_mit2_deserialised.is_enabled is False

    @pytest.mark.parametrize("n_qubits", [1, 2, 4, 8, 31, 64])
    @pytest.mark.parametrize("m3_available", [True, False])
    def test_serialisation(self, n_qubits, m3_available):
        qubit_indices = list(range(n_qubits))
        linear = generate_random_linear(qubit_indices)
        readout_mit = ReadoutMitigation(linear=linear, m3_available=m3_available)
        error_mit = ErrorMitigation(readout_mitigation=readout_mit)

        blob = error_mit.model_dump()
        error_mit_deserialised = ErrorMitigation(**blob)
        assert error_mit == error_mit_deserialised
