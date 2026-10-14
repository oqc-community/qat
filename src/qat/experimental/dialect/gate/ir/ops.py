# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Operations of the gate dialect.

Each gate is its own op, such as ``HOp`` or ``CZOp``. Base classes hold the shape of each
family of gates: the qubits a gate acts on, the angles it takes, and the rule that each
result keeps the qubit type of its operand. Angles are ``f64`` operands, so a gate can take
a run-time parameter as well as a constant.

Every op that takes a qubit consumes it, so a qubit value should be used once. Op
verifiers check this only within one op: a two-qubit gate or a barrier rejects the same
qubit twice. Checking it across a whole program is left to a verification pass.

Gate ops are not ``Pure``. They change the state of a qubit, and dead-code elimination would
otherwise delete a gate whose result is unused.
"""

from collections.abc import Sequence

from xdsl.dialects.builtin import IntegerAttr, f64, i1
from xdsl.ir import Operation, SSAValue
from xdsl.irdl import (
    BaseAttr,
    IRDLOperation,
    VarConstraint,
    irdl_op_definition,
    operand_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.utils.exceptions import VerifyException

from qat.experimental.dialect.gate.ir.traits import CliffordTrait
from qat.experimental.dialect.gate.ir.types import (
    AlgorithmicQubitType,
    PhysicalQubitType,
    QubitType,
)

# A gate must return each qubit with the type it consumed, and for a physical qubit, the
# same index. Binding a qubit operand and its result to one variable enforces this.
_QUBIT_TYPE_BIND = VarConstraint("Q", BaseAttr(QubitType))
_MULTI_0_QUBIT_TYPE_BIND = VarConstraint("Q0", BaseAttr(QubitType))
_MULTI_1_QUBIT_TYPE_BIND = VarConstraint("Q1", BaseAttr(QubitType))


def _verify_distinct_qubits(first: SSAValue, second: SSAValue) -> None:
    """Rejects a two-qubit gate whose operands are not two qubits of one kind.

    The same SSA value twice is the same qubit. Both qubits must be algorithmic, or both
    physical. Two physical qubits with the same index are the same qubit. Two algorithmic
    qubits cannot be compared further before placement.
    """
    if first is second:
        raise VerifyException("a two-qubit gate needs two qubits, got the same value twice")
    if type(first.type) is not type(second.type):
        raise VerifyException(
            "a two-qubit gate needs both qubits algorithmic or both physical, got "
            f"{first.type} and {second.type}"
        )
    if (
        isinstance(first.type, PhysicalQubitType)
        and isinstance(second.type, PhysicalQubitType)
        and first.type.index.value.data == second.type.index.value.data
    ):
        raise VerifyException(
            "a two-qubit gate needs two qubits, got physical qubit "
            f"{first.type.index.value.data} twice"
        )


@irdl_op_definition
class AllocateQubitOp(IRDLOperation):
    """Allocates a new qubit of any qubit type, initialised to the |0⟩ state.

    The result type says whether the qubit is algorithmic or physical.
    ``AllocateAlgorithmicQubitOp`` and ``AllocatePhysicalQubitOp`` each allow only one.

    :ivar result: The new qubit.
    """

    name = "gate.op.allocate_qubit"
    result = result_def(QubitType)

    def __init__(self, qubit_type: QubitType):
        """:param qubit_type: The type of the new qubit."""
        super().__init__(result_types=[qubit_type])


@irdl_op_definition
class AllocateAlgorithmicQubitOp(AllocateQubitOp):
    """Allocates a new algorithmic qubit, initialised to the |0⟩ state.

    An algorithmic qubit is a qubit as written in the source program, before placement.
    Compiling a program starts by allocating its qubits with this op.
    """

    name = "gate.op.allocate_qubit.algorithmic"
    result = result_def(AlgorithmicQubitType)

    def __init__(self):
        """Allocates an algorithmic qubit."""
        super().__init__(AlgorithmicQubitType())


@irdl_op_definition
class AllocatePhysicalQubitOp(AllocateQubitOp):
    """Allocates a new physical qubit, initialised to the |0⟩ state."""

    name = "gate.op.allocate_qubit.physical"
    result = result_def(PhysicalQubitType)

    def __init__(self, index: IntegerAttr | int):
        """:param index: The index of the qubit on the device."""
        super().__init__(PhysicalQubitType(index))


@irdl_op_definition
class DeallocateQubitOp(IRDLOperation):
    """Ends the lifetime of a qubit.

    Consumes the qubit and returns nothing. Use it for a qubit that is not measured, so its
    chain of values has an explicit end. The verifier does not require one.

    :ivar qubit: The qubit to deallocate.
    """

    name = "gate.op.deallocate_qubit"
    qubit = operand_def(QubitType)

    def __init__(self, qubit: SSAValue | Operation):
        """:param qubit: The qubit to deallocate."""
        super().__init__(operands=[qubit])


@irdl_op_definition
class MeasureOp(IRDLOperation):
    """Measures a single qubit.

    Consumes the qubit and returns the outcome as an ``i1``.

    :ivar qubit: The qubit to measure.
    :ivar result: The measurement outcome.
    """

    name = "gate.op.measure"
    qubit = operand_def(QubitType)
    result = result_def(i1)

    def __init__(self, qubit: SSAValue | Operation):
        """:param qubit: The qubit to measure."""
        super().__init__(operands=[qubit], result_types=[i1])


class OneQubitGate(IRDLOperation):
    """Base class for a gate on one qubit, with no angle.

    :ivar qubit: Qubit operand to which the gate is applied.
    :ivar result: The qubit after the gate.
    """

    qubit = operand_def(_QUBIT_TYPE_BIND)
    result = result_def(_QUBIT_TYPE_BIND)

    def __init__(self, qubit: SSAValue | Operation):
        """:param qubit: The qubit to act on. The result has the same type."""
        qubit = SSAValue.get(qubit)
        super().__init__(operands=[qubit], result_types=[qubit.type])


class CliffordOneQubitGate(OneQubitGate):
    """Base class for a Clifford gate on one qubit."""

    traits = traits_def(CliffordTrait())


class TwoQubitGate(IRDLOperation):
    """Base class for a gate on two qubits, with no angle.

    The two qubits must be distinct, and both algorithmic or both physical.

    :ivar qubit_0: First qubit operand to which the gate is applied.
    :ivar qubit_1: Second qubit operand to which the gate is applied.
    :ivar result_qubit_0: The first qubit after the gate.
    :ivar result_qubit_1: The second qubit after the gate.
    """

    qubit_0 = operand_def(_MULTI_0_QUBIT_TYPE_BIND)
    qubit_1 = operand_def(_MULTI_1_QUBIT_TYPE_BIND)
    result_qubit_0 = result_def(_MULTI_0_QUBIT_TYPE_BIND)
    result_qubit_1 = result_def(_MULTI_1_QUBIT_TYPE_BIND)

    def __init__(self, qubit_0: SSAValue | Operation, qubit_1: SSAValue | Operation):
        """Builds the gate. Each result has the type of its qubit.

        :param qubit_0: The first qubit.
        :param qubit_1: The second qubit.
        """
        qubit_0, qubit_1 = SSAValue.get(qubit_0), SSAValue.get(qubit_1)
        super().__init__(
            operands=[qubit_0, qubit_1], result_types=[qubit_0.type, qubit_1.type]
        )

    def verify_(self) -> None:
        _verify_distinct_qubits(self.qubit_0, self.qubit_1)


class CliffordTwoQubitGate(TwoQubitGate):
    """Base class for a Clifford gate on two qubits."""

    traits = traits_def(CliffordTrait())


class DirectionalTwoQubitGate(TwoQubitGate):
    """Base class for a gate on two qubits with control and target roles, with no angle.

    The control is the first qubit and the target the second. The properties below name them
    by role.
    """

    def __init__(self, control: SSAValue | Operation, target: SSAValue | Operation):
        """Builds the gate. Each result has the type of its qubit.

        :param control: The control qubit.
        :param target: The target qubit.
        """
        super().__init__(control, target)

    @property
    def qubit_control(self) -> SSAValue:
        """The control qubit operand."""
        return self.qubit_0

    @property
    def qubit_target(self) -> SSAValue:
        """The target qubit operand."""
        return self.qubit_1

    @property
    def result_qubit_control(self) -> SSAValue:
        """The control qubit after the gate."""
        return self.result_qubit_0

    @property
    def result_qubit_target(self) -> SSAValue:
        """The target qubit after the gate."""
        return self.result_qubit_1


class CliffordDirectionalTwoQubitGate(DirectionalTwoQubitGate):
    """Base class for a Clifford gate on two qubits with control and target roles."""

    traits = traits_def(CliffordTrait())


class OneAngleGate(IRDLOperation):
    """Base class for a gate on one qubit with one angle.

    :ivar qubit: Qubit operand to which the gate is applied.
    :ivar theta: Angle in radians. A constant, or a run-time parameter.
    :ivar result: The qubit after the gate.
    """

    qubit = operand_def(_QUBIT_TYPE_BIND)
    theta = operand_def(f64)
    result = result_def(_QUBIT_TYPE_BIND)

    def __init__(self, qubit: SSAValue | Operation, theta: SSAValue | Operation):
        """Builds the gate. The result has the type of the qubit.

        :param qubit: The qubit to act on.
        :param theta: The angle, an ``f64`` value.
        """
        qubit = SSAValue.get(qubit)
        super().__init__(operands=[qubit, theta], result_types=[qubit.type])


class ThreeAngleGate(IRDLOperation):
    """Base class for a gate on one qubit with three angles.

    :ivar qubit: Qubit operand to which the gate is applied.
    :ivar theta: First angle in radians.
    :ivar phi: Second angle in radians.
    :ivar lam: Third angle in radians (lambda).
    :ivar result: The qubit after the gate.
    """

    qubit = operand_def(_QUBIT_TYPE_BIND)
    theta = operand_def(f64)
    phi = operand_def(f64)
    lam = operand_def(f64)
    result = result_def(_QUBIT_TYPE_BIND)

    def __init__(
        self,
        qubit: SSAValue | Operation,
        theta: SSAValue | Operation,
        phi: SSAValue | Operation,
        lam: SSAValue | Operation,
    ):
        """Builds the gate. The result has the type of the qubit.

        :param qubit: The qubit to act on.
        :param theta: The first angle, an ``f64`` value.
        :param phi: The second angle, an ``f64`` value.
        :param lam: The third angle, an ``f64`` value.
        """
        qubit = SSAValue.get(qubit)
        super().__init__(operands=[qubit, theta, phi, lam], result_types=[qubit.type])


@irdl_op_definition
class HOp(CliffordOneQubitGate):
    """The Hadamard gate."""

    name = "gate.op.h"


@irdl_op_definition
class XOp(CliffordOneQubitGate):
    """The Pauli-X gate."""

    name = "gate.op.x"


@irdl_op_definition
class CZOp(CliffordTwoQubitGate):
    """The controlled-Z gate."""

    name = "gate.op.cz"


@irdl_op_definition
class CXOp(CliffordDirectionalTwoQubitGate):
    """The controlled-X (CNOT) gate."""

    name = "gate.op.cx"


@irdl_op_definition
class RZOp(OneAngleGate):
    """A rotation about the Z axis by ``theta``."""

    name = "gate.op.rz"


@irdl_op_definition
class U3Op(ThreeAngleGate):
    """The general single-qubit gate U3(theta, phi, lambda)."""

    name = "gate.op.u3"


@irdl_op_definition
class ResetOp(IRDLOperation):
    """Resets a qubit to the |0⟩ state.

    ``MeasureOp`` consumes its qubit, so a measured qubit cannot be reset.

    :ivar qubit: Qubit operand to which the reset is applied.
    :ivar result: The qubit after the reset.
    """

    name = "gate.op.reset"
    qubit = operand_def(_QUBIT_TYPE_BIND)
    result = result_def(_QUBIT_TYPE_BIND)

    def __init__(self, qubit: SSAValue | Operation):
        """:param qubit: The qubit to act on. The result has the same type."""
        qubit = SSAValue.get(qubit)
        super().__init__(operands=[qubit], result_types=[qubit.type])


@irdl_op_definition
class IdentityOp(IRDLOperation):
    """Identity gate — passes the qubit state through unchanged.

    Used as a canonical no-op placeholder, e.g. after gate cancellation leaves a qubit with
    an empty operation sequence before dead-qubit elimination runs.

    :ivar qubit: Qubit operand to which the gate is applied.
    :ivar result: The qubit after the gate.
    """

    name = "gate.op.identity"
    traits = traits_def(CliffordTrait())

    qubit = operand_def(_QUBIT_TYPE_BIND)
    result = result_def(_QUBIT_TYPE_BIND)

    def __init__(self, qubit: SSAValue | Operation):
        """:param qubit: The qubit to act on. The result has the same type."""
        qubit = SSAValue.get(qubit)
        super().__init__(operands=[qubit], result_types=[qubit.type])


@irdl_op_definition
class BarrierOp(IRDLOperation):
    """Optimisation barrier across a set of qubits.

    Stops optimisations moving operations on these qubits across the barrier. It carries no
    timing: whether it also lines up the qubits' timelines is decided when it is lowered.
    Takes a variadic set of distinct qubits and returns each one unchanged.

    :ivar qubits: Qubits operand to which the barrier is applied.
    :ivar result_qubits: The qubits after the barrier.
    """

    name = "gate.op.barrier"
    qubits = var_operand_def(QubitType)
    result_qubits = var_result_def(QubitType)

    def __init__(self, qubits: Sequence[SSAValue | Operation]):
        """:param qubits: The qubits to fence. Each result has the type of its qubit."""
        qubits = [SSAValue.get(qubit) for qubit in qubits]
        super().__init__(operands=[qubits], result_types=[[q.type for q in qubits]])

    def verify_(self) -> None:
        # A variadic group cannot share one VarConstraint (that would force every qubit to
        # have the same type), so check each operand/result pair explicitly.
        if len(self.qubits) != len(self.result_qubits):
            raise VerifyException(
                f"expected as many results as operands, got {len(self.qubits)} operands "
                f"and {len(self.result_qubits)} results"
            )
        for i, (qubit, result) in enumerate(
            zip(self.qubits, self.result_qubits, strict=False)
        ):
            if qubit.type != result.type:
                raise VerifyException(
                    f"result {i} type {result.type} must match operand {i} type "
                    f"{qubit.type}"
                )
        if len(set(self.qubits)) != len(self.qubits):
            raise VerifyException(
                "a barrier needs distinct qubits, got the same value twice"
            )
        indices = [
            qubit.type.index.value.data
            for qubit in self.qubits
            if isinstance(qubit.type, PhysicalQubitType)
        ]
        repeated = sorted({index for index in indices if indices.count(index) > 1})
        if repeated:
            raise VerifyException(
                f"a barrier needs distinct qubits, got physical qubit {repeated[0]} twice"
            )
