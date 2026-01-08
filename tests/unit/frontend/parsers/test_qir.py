# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import pytest

from qat.frontend.parsers.qir import QIRParser as PydQIRParser
from qat.ir.instruction_builder import QuantumInstructionBuilder
from qat.ir.measure import Acquire, MeasureBlock
from qat.ir.waveforms import Pulse
from qat.model.convert_purr import convert_purr_echo_hw_to_pydantic
from qat.model.device import Qubit
from qat.model.loaders.lucy import LucyModelLoader
from qat.purr.backends.echo import apply_setup_to_hardware
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.integrations.qir import QIRParser as LegQIRParser

from tests.unit.purr.integrations.test_qir import _get_qir_path
from tests.unit.utils.instruction import (
    count_number_of_non_sync_non_phase_reset_non_delay_non_post_processing_instructions,
)
from tests.unit.utils.qasm_qir import get_qir, qir_files, short_file_name

n_qubits = 32
linear_topology = []
for q in range(n_qubits):
    q2 = q + 1 if q != n_qubits - 1 else 0
    linear_topology.append((q, q2))

leg_hw_model = apply_setup_to_hardware(
    QuantumHardwareModel(), qubit_count=n_qubits, connectivity=linear_topology
)
pyd_hw_model = convert_purr_echo_hw_to_pydantic(leg_hw_model)


class TestQIRParser:
    @pytest.mark.parametrize(
        "qir_file",
        list(qir_files),
        ids=short_file_name,
    )
    @pytest.mark.parametrize(
        "model", [LucyModelLoader(32).load(), LucyModelLoader(32, start_index=2).load()]
    )
    def test_programs_parse(self, model, qir_file):
        """Basic smoke test to ensure QIR programs parse without error."""
        parser = PydQIRParser()
        qir_string = get_qir(qir_file)
        builder = parser.parse(QuantumInstructionBuilder(model), qir_string)
        assert isinstance(builder, QuantumInstructionBuilder)

    def test_first_logical_indices_are_used(self):
        model = LucyModelLoader(16, start_index=3).load()
        parser = PydQIRParser()
        qir_string = _get_qir_path("generator-bell.ll")
        builder = parser.parse(QuantumInstructionBuilder(model), qir_string)

        qubits = set()
        for instruction in builder.instructions:
            if isinstance(instruction, (Pulse, Acquire, MeasureBlock)):
                for target in instruction.targets:
                    device = model.device_for_pulse_channel_id(target)
                    if not isinstance(device, Qubit):
                        device = model.qubit_for_resonator(device)
                    qubit_id = model.index_of_qubit(device)
                    qubits.update({qubit_id})
        assert qubits == {3, 4}

    @pytest.mark.parametrize(
        "qir_file",
        list(qir_files),
        ids=short_file_name,
    )
    def test_qir_legacy_vs_pyd_parse(self, qir_file):
        pyd_parser = PydQIRParser()
        leg_parser = LegQIRParser(leg_hw_model)

        qir_string = get_qir(qir_file)

        pyd_builder = pyd_parser.parse(QuantumInstructionBuilder(pyd_hw_model), qir_string)
        leg_builder = leg_parser.parse(qir_string)

        assert (
            count_number_of_non_sync_non_phase_reset_non_delay_non_post_processing_instructions(
                pyd_builder._ir
            )
            == count_number_of_non_sync_non_phase_reset_non_delay_non_post_processing_instructions(
                leg_builder.instructions
            )
        )
