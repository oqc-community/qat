# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from copy import deepcopy

import pytest

from qat.ir.instruction_builder import (
    QuantumInstructionBuilder as PydQuantumInstructionBuilder,
)
from qat.middleend.passes.validation import PydNoMidCircuitMeasurementValidation
from qat.utils.hardware_model import generate_hw_model


class TestNoMidCircuitMeasurementValidation:
    hw = generate_hw_model(32)

    def test_no_mid_circuit_meas_found(self):
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        for qubit in self.hw.qubits.values():
            builder.measure_single_shot_z(target=qubit)

        p = PydNoMidCircuitMeasurementValidation(self.hw)
        builder_before = deepcopy(builder)
        p.run(builder)

        assert builder_before.number_of_instructions == builder.number_of_instructions
        for instr_before, instr_after in zip(builder_before, builder):
            assert instr_before == instr_after

    def test_throw_error_mid_circuit_meas(self):
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        for qubit in self.hw.qubits.values():
            builder.measure_single_shot_z(target=qubit)

        builder.X(target=self.hw.qubit_with_index(0))

        p = PydNoMidCircuitMeasurementValidation(self.hw)
        with pytest.raises(ValueError):
            p.run(builder)
