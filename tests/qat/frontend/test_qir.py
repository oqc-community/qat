from itertools import zip_longest

import pytest

from qat.frontend.qir_parser import QIRParser as PydQIRParser
from qat.model.convert_legacy import convert_legacy_echo_hw_to_pydantic
from qat.purr.backends.echo import apply_setup_to_hardware
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.integrations.qir import QIRParser as LegQIRParser

from tests.qat.test_qir import _get_qir_path

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

    # @pytest.mark.parametrize(("qir_file", "n_instr"), [("generator-bell.ll", 96)])
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

        # Since we allow a choice between syncs between all pulse channels
        # within a qubit and a syncs between all pulse channels of all qubits
        # simultaneously, there is a discrepancy in the number of syncs between
        # the experimental and legacy code. However, the number of non-sync
        # instructions should still be equal.
        n_pyd_instr_no_sync = 0
        n_leg_instr_no_sync = 0
        for pydinstr, leginstr in zip_longest(pyd_builder._ir, leg_builder.instructions):
            if pydinstr and pydinstr.__class__.__name__ != "Synchronize":
                n_pyd_instr_no_sync += 1
            if leginstr and leginstr.__class__.__name__ != "Synchronize":
                n_leg_instr_no_sync += 1

        assert n_pyd_instr_no_sync == n_leg_instr_no_sync
