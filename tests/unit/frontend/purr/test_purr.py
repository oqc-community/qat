# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import pytest

from qat.frontend.purr.purr import PurrFrontend
from qat.ir.instruction_builder import PydQuantumInstructionBuilder
from qat.model.loaders.lucy import LucyModelLoader
from qat.model.loaders.purr import EchoModelLoader


class TestPurrFrontend:
    invalid_src_programs = [
        "test.qasm",
        PydQuantumInstructionBuilder(hardware_model=LucyModelLoader().load()),
    ]

    @pytest.mark.parametrize("src", invalid_src_programs)
    def test_non_ir_builder_is_rejected(self, src):
        frontend = PurrFrontend(model=None)
        assert frontend.check_and_return_source(src) is False

    @pytest.mark.parametrize("src", invalid_src_programs)
    def test_emit_for_non_ir_builder_raises(self, src):
        frontend = PurrFrontend(model=None)
        with pytest.raises(
            ValueError, match="Source is not a valid purr InstructionBuilder."
        ):
            frontend.emit(src)

    def test_ir_builder_with_different_model_is_rejected(self):
        model1 = EchoModelLoader(qubit_count=2).load()
        model2 = EchoModelLoader(qubit_count=3).load()
        builder = model1.create_builder()
        frontend = PurrFrontend(model=model2)
        assert frontend.check_and_return_source(builder) is False

    def test_ir_builder_with_sweeps_is_rejected(self):
        model = EchoModelLoader(qubit_count=2).load()
        builder = model.create_builder()
        builder.sweep([])
        frontend = PurrFrontend(model=model)
        assert frontend.check_and_return_source(builder) is False

    def test_ir_builder_with_device_assign_is_rejected(self):
        model = EchoModelLoader(qubit_count=2).load()
        builder = model.create_builder()
        builder.device_assign(
            next(iter(model.physical_channels.values())), "imbalance", 1.05
        )
        frontend = PurrFrontend(model=model)
        assert frontend.check_and_return_source(builder) is False

    def test_valid_ir_builder_is_accepted(self):
        model = EchoModelLoader(qubit_count=2).load()
        builder = model.create_builder()
        qubits = model.qubits[0:2]
        builder.had(qubits[0])
        builder.cnot(qubits[0], qubits[1])
        builder.measure(qubits[0])
        builder.measure(qubits[1])
        frontend = PurrFrontend(model=model)
        assert frontend.check_and_return_source(builder) is builder
