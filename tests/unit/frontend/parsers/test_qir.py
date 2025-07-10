# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import pytest

from qat.frontend.parsers.qir import QIRParser as PydQIRParser
from qat.ir.instruction_builder import QuantumInstructionBuilder
from qat.ir.instructions import Return
from qat.model.convert_purr import convert_purr_echo_hw_to_pydantic
from qat.purr.backends.echo import apply_setup_to_hardware
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.integrations.qir import QIRParser as LegQIRParser

from tests.unit.purr.integrations.test_qir import _get_qir_path
from tests.unit.utils.instruction import (
    count_number_of_non_sync_non_phase_reset_non_delay_instructions,
)

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
        [
            "base_profile_ops.ll",
            "basic_cudaq.ll",
            "bell_psi_minus.ll",
            "bell_psi_plus.ll",
            "bell_theta_minus.ll",
            "bell_theta_plus.ll",
            "complicated.ll",
            "generator-bell.ll",
            "out_of_order_measure.ll",
        ],
    )
    def test_qir_legacy_vs_pyd_parse(self, qir_file):
        pyd_parser = PydQIRParser(pyd_hw_model)
        leg_parser = LegQIRParser(leg_hw_model)

        qir_string = _get_qir_path(qir_file)

        pyd_builder = pyd_parser.parse(qir_string)
        leg_builder = leg_parser.parse(qir_string)

        assert count_number_of_non_sync_non_phase_reset_non_delay_instructions(
            pyd_builder._ir
        ) == count_number_of_non_sync_non_phase_reset_non_delay_instructions(
            leg_builder.instructions
        )

    def test_results_variables_are_cleaned(self):
        parser = PydQIRParser(pyd_hw_model)
        qir_string = _get_qir_path("bell_psi_plus.ll")
        builder = parser.parse(qir_string)
        assert isinstance(builder, QuantumInstructionBuilder)
        assert parser.result_variables == []
        return_insts = [inst for inst in builder.instructions if isinstance(inst, Return)]
        assert len(return_insts) > 0
        vars = []
        for inst in return_insts:
            vars.extend(inst.variables)
        assert len(vars) > 0
