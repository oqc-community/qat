# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import pytest

from qat.frontend.parsers.qir import QIRParser as PydQIRParser
from qat.model.convert_legacy import convert_legacy_echo_hw_to_pydantic
from qat.purr.backends.echo import apply_setup_to_hardware
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.integrations.qir import QIRParser as LegQIRParser

from tests.qat.test_qir import _get_qir_path
from tests.qat.utils.instruction import count_number_of_non_sync_instructions

n_qubits = 32
linear_topology = []
for q in range(n_qubits):
    q2 = q + 1 if q != n_qubits - 1 else 0
    linear_topology.append((q, q2))

leg_hw_model = apply_setup_to_hardware(
    QuantumHardwareModel(), qubit_count=n_qubits, connectivity=linear_topology
)
pyd_hw_model = convert_legacy_echo_hw_to_pydantic(leg_hw_model)


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

        assert count_number_of_non_sync_instructions(
            pyd_builder._ir
        ) == count_number_of_non_sync_instructions(leg_builder.instructions)
