# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from qat.experimental.system_data.canonical.schema import OperationData
from qat.experimental.system_data.materialisers.operations.operation_builder import (
    AbstractOperationBuilder,
)


def _op(op_id: str, interface: str = "public") -> OperationData:
    return OperationData(id=op_id, kind="gate", interface=interface)


class _BuilderWithoutPrivateOps(AbstractOperationBuilder):
    def make_z_operation(self) -> OperationData:
        return _op("Z")

    def make_x_gate(self) -> OperationData:
        return _op("X")

    def make_y_gate(self) -> OperationData:
        return _op("Y")

    def make_u_gate(self) -> OperationData:
        return _op("U")

    def make_had_operation(self) -> OperationData:
        return _op("H")

    def make_sx_operation(self) -> OperationData:
        return _op("SX")

    def make_sxdg_operation(self) -> OperationData:
        return _op("SXdg")

    def make_s_operation(self) -> OperationData:
        return _op("S")

    def make_sdg_operation(self) -> OperationData:
        return _op("Sdg")

    def make_t_operation(self) -> OperationData:
        return _op("T")

    def make_tdg_operation(self) -> OperationData:
        return _op("Tdg")

    def make_rx_gate(self) -> OperationData:
        return _op("rx")

    def make_ry_gate(self) -> OperationData:
        return _op("ry")

    def make_rz_gate(self) -> OperationData:
        return _op("rz")

    def make_u1_gate(self) -> OperationData:
        return _op("u1")

    def make_u2_gate(self) -> OperationData:
        return _op("u2")

    def make_id_gate(self) -> OperationData:
        return _op("id")

    def make_delay_operation(self) -> OperationData:
        return _op("delay")

    def make_measure_operation(self) -> OperationData:
        return _op("measure")

    def make_initiate_operation(self) -> OperationData:
        return _op("initiate")

    def make_reset_operation(self) -> OperationData:
        return _op("reset")

    def make_two_qubit_operations(self) -> tuple[OperationData, ...]:
        return (_op("twoq"),)


class _BuilderWithPrivateOps(_BuilderWithoutPrivateOps):
    def make_private_single_qubit_operations(self) -> tuple[OperationData, ...]:
        return (_op("priv0", interface="private"), _op("priv1", interface="private"))


class TestAbstractOperationBuilderContract:
    def test_private_single_qubit_hook_defaults_to_empty(self):
        builder = _BuilderWithoutPrivateOps(qubit_id="q0")
        ids = {op.id for op in builder.build_single_qubit_operations()}
        assert "priv0" not in ids
        assert "priv1" not in ids

    def test_private_single_qubit_hook_is_included_in_build_single_qubit(self):
        builder = _BuilderWithPrivateOps(qubit_id="q0")
        ids = [op.id for op in builder.build_single_qubit_operations()]
        assert ids[:2] == ["priv0", "priv1"]

    def test_build_includes_two_qubit_hook_operations(self):
        builder = _BuilderWithoutPrivateOps(qubit_id="q0")
        ids = {op.id for op in builder.build()}
        assert "twoq" in ids

    def test_build_last_wins_merge_still_applies(self):
        builder = _BuilderWithoutPrivateOps(qubit_id="q0")
        replacement_rx = OperationData(id="rx", kind="gate", interface="public")
        ops = builder.build(extra_operations=(replacement_rx,))
        rx = [op for op in ops if op.id == "rx"]
        assert len(rx) == 1
        assert rx[0] is replacement_rx
