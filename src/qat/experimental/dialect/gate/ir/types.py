# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from xdsl.dialects.builtin import IntegerAttr, i64
from xdsl.ir import ParametrizedAttribute, TypeAttribute
from xdsl.irdl import irdl_attr_definition, param_def
from xdsl.utils.exceptions import VerifyException


class QubitType(ParametrizedAttribute, TypeAttribute):
    """Abstract base for qubit type.

    Concrete subtypes distinguish algorithmic qubits (pre-placement, directly from the
    source program) from physical qubits (post-placement), allowing placement to be
    expressed as a type-level rewrite without any structural change to the circuit ops.
    """

    name = "gate.qubit"


@irdl_attr_definition
class AlgorithmicQubitType(QubitType):
    """An algorithmic qubit type, assigned at circuit construction time.

    Represents a qubit as it appears in an algorithm, such as a qubit in a source QASM2
    program, with no hardware assignment. Placement maps algorithmic qubits to physical
    qubits.
    """

    name = "gate.qubit.algorithmic"


@irdl_attr_definition
class PhysicalQubitType(QubitType):
    """A physical qubit type, assigned after placement.

    :ivar index: The index of the qubit on the device. Must not be negative.
    """

    name = "gate.qubit.physical"
    index: IntegerAttr = param_def()

    def __init__(self, index: IntegerAttr | int):
        if isinstance(index, int):
            index = IntegerAttr(index, i64)
        super().__init__(index)

    def verify(self) -> None:
        super().verify()
        if self.index.value.data < 0:
            raise VerifyException(
                f"a physical qubit index cannot be negative, got {self.index.value.data}"
            )
