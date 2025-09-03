# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import pytest

from qat.frontend import QatFrontend
from qat.ir.instruction_builder import QuantumInstructionBuilder
from qat.model.loaders.lucy import LucyModelLoader


class TestQatFrontend:
    invalid_src_programs = ["test.qasm", 123]

    @pytest.mark.parametrize("src", invalid_src_programs)
    def test_non_ir_builder_is_rejected(self, src):
        frontend = QatFrontend(model=None)
        assert frontend.check_and_return_source(src) is False

    @pytest.mark.parametrize("src", invalid_src_programs)
    def test_emit_for_non_ir_builder_raises(self, src):
        frontend = QatFrontend(model=None)
        with pytest.raises(ValueError, match="Source is not a valid InstructionBuilder."):
            frontend.emit(src)

    def test_ir_builder_with_different_model_is_rejected(self):
        model1 = LucyModelLoader(qubit_count=2).load()
        model2 = LucyModelLoader(qubit_count=3).load()
        builder = QuantumInstructionBuilder(hardware_model=model1)
        frontend = QatFrontend(model=model2)
        assert frontend.check_and_return_source(builder) is False

    def test_valid_ir_builder_is_accepted(self):
        model = LucyModelLoader(qubit_count=2).load()
        builder = QuantumInstructionBuilder(hardware_model=model)
        qubits = [model.qubit_with_index(i) for i in range(2)]
        builder.had(qubits[0])
        builder.cnot(qubits[0], qubits[1])
        builder.measure(qubits[0])
        builder.measure(qubits[1])
        frontend = QatFrontend(model=model)
        assert frontend.check_and_return_source(builder) is builder
