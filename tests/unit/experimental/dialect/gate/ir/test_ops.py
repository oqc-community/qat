# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Unit tests for the gate dialect operations and dialect registration."""

from io import StringIO

import pytest
from xdsl.context import Context
from xdsl.dialects.arith import Arith, ConstantOp
from xdsl.dialects.builtin import Builtin, FloatAttr, IntegerAttr, ModuleOp, f64, i1, i32
from xdsl.ir import Block, OpResult
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import Commutative, NoMemoryEffect, Pure
from xdsl.transforms.dead_code_elimination import DeadCodeElimination
from xdsl.utils.exceptions import ParseError, VerifyException
from xdsl.utils.test_value import create_ssa_value

from qat.experimental.dialect.gate.ir import (
    AlgorithmicQubitType,
    AllocateAlgorithmicQubitOp,
    AllocatePhysicalQubitOp,
    AllocateQubitOp,
    BarrierOp,
    CliffordTrait,
    CXOp,
    CZOp,
    DeallocateQubitOp,
    Gate,
    HOp,
    IdentityOp,
    MeasureOp,
    PhysicalQubitType,
    ResetOp,
    RZOp,
    U3Op,
    XOp,
)
from qat.experimental.dialect.gate.ir.ops import TwoQubitGate

ALG = AlgorithmicQubitType()
PHYS = PhysicalQubitType(0)
PHYS_1 = PhysicalQubitType(1)


def _qubit(type_=ALG):
    """An SSA qubit value defined by an allocation op (so modules stay well-formed)."""
    return AllocateQubitOp(type_).result


def _angle(value: float = 0.5):
    """An SSA ``f64`` angle defined by a constant op."""
    return ConstantOp(FloatAttr(value, f64)).result


# Each gate op, with the number of qubits and angles it takes.
GATES = [
    (HOp, 1, 0),
    (XOp, 1, 0),
    (CZOp, 2, 0),
    (CXOp, 2, 0),
    (RZOp, 1, 1),
    (U3Op, 1, 3),
]
GATE_OPS = [op_type for op_type, _, _ in GATES]
CLIFFORD_OPS = [HOp, XOp, CZOp, CXOp, IdentityOp]
ALLOCATE_OPS = [AllocateQubitOp, AllocateAlgorithmicQubitOp, AllocatePhysicalQubitOp]


def _gate(op_type, n_qubits, n_angles, physical=False):
    """A gate op on fresh qubits, with constant angles.

    Physical qubits get distinct indices, as a two-qubit gate needs.
    """
    types = [PhysicalQubitType(i) if physical else ALG for i in range(n_qubits)]
    angles = [_angle(0.1 * (i + 1)) for i in range(n_angles)]
    return op_type(*map(_qubit, types), *angles)


def _ctx() -> Context:
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Arith)
    ctx.load_dialect(Gate)
    return ctx


def _print(op) -> str:
    stream = StringIO()
    Printer(stream=stream).print_op(op)
    return stream.getvalue()


def _round_trip(op) -> str:
    """Print a module containing ``op``, parse it back, verify and re-print."""
    inputs = [v.owner for v in op.operands if isinstance(v, OpResult)]
    first = _print(ModuleOp([*inputs, op]))
    parsed = Parser(_ctx(), first).parse_module()
    parsed.verify()
    assert _print(parsed) == first
    return first


ALL_OPS = [
    *ALLOCATE_OPS,
    DeallocateQubitOp,
    MeasureOp,
    *GATE_OPS,
    ResetOp,
    IdentityOp,
    BarrierOp,
]


class TestDialect:
    def test_name(self):
        assert Gate.name == "gate"

    def test_registers_every_op(self):
        assert set(Gate.operations) == set(ALL_OPS)

    def test_registers_only_qubit_types(self):
        """Gates are ops, so the dialect defines no gate attributes."""
        assert set(Gate.attributes) == {AlgorithmicQubitType, PhysicalQubitType}

    def test_abstract_qubit_type_does_not_parse(self):
        """``QubitType`` is abstract, so ``!gate.qubit`` is not a type in the IR."""
        with pytest.raises(ParseError):
            Parser(_ctx(), "!gate.qubit").parse_type()

    @pytest.mark.parametrize("op_type", ALL_OPS)
    def test_op_names_are_in_gate_namespace(self, op_type):
        assert op_type.name.startswith("gate.op.")

    def test_op_names_are_unique(self):
        names = [op_type.name for op_type in ALL_OPS]
        assert len(names) == len(set(names))


class TestAllocation:
    def test_allocate_algorithmic(self):
        op = AllocateAlgorithmicQubitOp()
        op.verify()
        assert op.result.type == ALG

    @pytest.mark.parametrize("index", [0, IntegerAttr(0, i32)])
    def test_allocate_physical(self, index):
        op = AllocatePhysicalQubitOp(index)
        op.verify()
        assert op.result.type == PhysicalQubitType(index)

    @pytest.mark.parametrize("qubit_type", [ALG, PHYS])
    def test_generic_allocate_accepts_any_qubit_type(self, qubit_type):
        op = AllocateQubitOp(qubit_type)
        op.verify()
        assert op.result.type == qubit_type

    def test_generic_allocate_rejects_non_qubit_type(self):
        with pytest.raises(VerifyException):
            AllocateQubitOp.create(result_types=[i32]).verify()

    def test_algorithmic_allocate_rejects_physical_type(self):
        with pytest.raises(VerifyException):
            AllocateAlgorithmicQubitOp.create(result_types=[PHYS]).verify()

    def test_physical_allocate_rejects_algorithmic_type(self):
        with pytest.raises(VerifyException):
            AllocatePhysicalQubitOp.create(result_types=[ALG]).verify()


class TestDeallocate:
    @pytest.mark.parametrize("qubit_type", [ALG, PHYS])
    def test_consumes_a_qubit_and_returns_nothing(self, qubit_type):
        op = DeallocateQubitOp(_qubit(qubit_type))
        op.verify()
        assert len(op.results) == 0

    def test_operand_must_be_qubit(self):
        with pytest.raises(VerifyException):
            DeallocateQubitOp(create_ssa_value(i32)).verify()


class TestMeasure:
    @pytest.mark.parametrize("qubit_type", [ALG, PHYS])
    def test_valid(self, qubit_type):
        op = MeasureOp(_qubit(qubit_type))
        op.verify()
        assert op.result.type == i1

    def test_result_must_be_i1(self):
        with pytest.raises(VerifyException):
            MeasureOp.create(operands=[_qubit()], result_types=[ALG]).verify()

    def test_operand_must_be_qubit(self):
        with pytest.raises(VerifyException):
            MeasureOp(create_ssa_value(i32)).verify()


class TestGates:
    @pytest.mark.parametrize("op_type, n_qubits, n_angles", GATES)
    @pytest.mark.parametrize("physical", [False, True], ids=["algorithmic", "physical"])
    def test_valid(self, op_type, n_qubits, n_angles, physical):
        _gate(op_type, n_qubits, n_angles, physical).verify()

    @pytest.mark.parametrize("op_type, n_qubits, n_angles", GATES)
    def test_result_types_follow_the_qubits(self, op_type, n_qubits, n_angles):
        op = _gate(op_type, n_qubits, n_angles, physical=True)
        qubit_types = [operand.type for operand in op.operands[:n_qubits]]
        assert [result.type for result in op.results] == qubit_types

    @pytest.mark.parametrize("op_type, n_qubits, n_angles", GATES)
    def test_rejects_wrong_number_of_qubits(self, op_type, n_qubits, n_angles):
        wrong = 2 if n_qubits == 1 else 1
        angles = [_angle() for _ in range(n_angles)]
        op = op_type.create(
            operands=[*(_qubit() for _ in range(wrong)), *angles],
            result_types=[ALG] * wrong,
        )
        with pytest.raises(VerifyException):
            op.verify()

    @pytest.mark.parametrize("op_type", [RZOp, U3Op])
    def test_rejects_missing_angle(self, op_type):
        op = op_type.create(operands=[_qubit()], result_types=[ALG])
        with pytest.raises(VerifyException):
            op.verify()

    @pytest.mark.parametrize("op_type, n_angles", [(RZOp, 1), (U3Op, 3)])
    def test_rejects_non_f64_angle(self, op_type, n_angles):
        angles = [create_ssa_value(i32) for _ in range(n_angles)]
        op = op_type(_qubit(), *angles)
        with pytest.raises(VerifyException):
            op.verify()

    @pytest.mark.parametrize("op_type, n_qubits, n_angles", GATES)
    def test_constructor_accepts_ops(self, op_type, n_qubits, n_angles):
        qubits = [AllocatePhysicalQubitOp(i) for i in range(n_qubits)]
        angles = [ConstantOp(FloatAttr(0.5, f64)) for _ in range(n_angles)]
        op = op_type(*qubits, *angles)
        op.verify()
        assert list(op.operands) == [
            *(q.result for q in qubits),
            *(a.result for a in angles),
        ]

    def test_directional_gate_is_a_two_qubit_gate(self):
        assert issubclass(CXOp, TwoQubitGate)

    def test_directional_gate_names_its_qubits_by_role(self):
        control, target = _qubit(PHYS), _qubit(PHYS_1)
        op = CXOp(control, target)
        assert op.qubit_control is control is op.qubit_0
        assert op.qubit_target is target is op.qubit_1
        assert op.result_qubit_control is op.result_qubit_0
        assert op.result_qubit_target is op.result_qubit_1


class TestAngleOperands:
    """Angles are SSA values, so a circuit can take them as run-time parameters."""

    @pytest.mark.parametrize("op_type, n_angles", [(RZOp, 1), (U3Op, 3)])
    def test_block_argument_angles_verify(self, op_type, n_angles):
        block = Block(arg_types=[f64] * n_angles)
        op = op_type(_qubit(), *block.args)
        block.add_op(op)
        op.verify()

    def test_binding_an_angle_to_a_constant_keeps_the_op(self):
        block = Block(arg_types=[f64])
        op = RZOp(_qubit(), block.args[0])
        block.add_op(op)
        constant = ConstantOp(FloatAttr(0.25, f64))
        block.insert_op_before(constant, op)

        block.args[0].replace_all_uses_with(constant.result)

        op.verify()
        assert list(block.ops) == [constant, op]
        assert op.theta is constant.result


class TestResetAndIdentity:
    @pytest.mark.parametrize("op_type", [ResetOp, IdentityOp])
    @pytest.mark.parametrize("qubit_type", [ALG, PHYS])
    def test_valid(self, op_type, qubit_type):
        op = op_type(AllocateQubitOp(qubit_type))
        op.verify()
        assert op.result.type == qubit_type

    @pytest.mark.parametrize("op_type", [ResetOp, IdentityOp])
    def test_requires_qubit_operand(self, op_type):
        op = op_type.create(operands=[create_ssa_value(i32)], result_types=[i32])
        with pytest.raises(VerifyException):
            op.verify()


class TestBarrier:
    @pytest.mark.parametrize("n_qubits", [0, 1, 3])
    def test_variadic(self, n_qubits):
        op = BarrierOp([_qubit() for _ in range(n_qubits)])
        op.verify()
        assert len(op.result_qubits) == n_qubits

    def test_result_types_follow_the_qubits(self):
        op = BarrierOp([_qubit(ALG), _qubit(PHYS)])
        assert [result.type for result in op.result_qubits] == [ALG, PHYS]

    def test_rejects_non_qubit_operand(self):
        with pytest.raises(VerifyException):
            BarrierOp([create_ssa_value(i32)]).verify()

    def test_accepts_ops(self):
        qubits = [AllocatePhysicalQubitOp(0), AllocatePhysicalQubitOp(1)]
        BarrierOp(qubits).verify()

    def test_rejects_the_same_value_twice(self):
        qubit = _qubit()
        with pytest.raises(VerifyException, match="same value twice"):
            BarrierOp([qubit, _qubit(), qubit]).verify()

    def test_rejects_the_same_physical_qubit_twice(self):
        op = BarrierOp([_qubit(PHYS_1), _qubit(ALG), _qubit(PHYS_1)])
        with pytest.raises(VerifyException, match="physical qubit 1 twice"):
            op.verify()


ONE_QUBIT_GATES = [(op_type, n) for op_type, q, n in GATES if q == 1]
TWO_QUBIT_GATES = [op_type for op_type, q, _ in GATES if q == 2]


class TestQubitPreservation:
    """Gates must return each qubit with the same type and identity they consumed."""

    CHANGED = [(ALG, PHYS), (PHYS, ALG), (PHYS, PHYS_1)]

    @pytest.mark.parametrize("in_type, out_type", CHANGED)
    @pytest.mark.parametrize(
        "op_type, n_angles", [*ONE_QUBIT_GATES, (ResetOp, 0), (IdentityOp, 0)]
    )
    def test_one_qubit_ops_reject_changed_qubit(self, op_type, n_angles, in_type, out_type):
        angles = [_angle() for _ in range(n_angles)]
        op = op_type.create(operands=[_qubit(in_type), *angles], result_types=[out_type])
        with pytest.raises(VerifyException):
            op.verify()

    @pytest.mark.parametrize("op_type", TWO_QUBIT_GATES)
    @pytest.mark.parametrize(
        "in_types, out_types",
        [
            ((ALG, ALG), (PHYS, ALG)),
            ((ALG, ALG), (ALG, PHYS)),
            ((PHYS, PHYS_1), (PHYS_1, PHYS)),  # swapped identities
            ((PHYS, PHYS_1), (PHYS, PHYS)),
        ],
    )
    def test_two_qubit_gates_reject_changed_qubits(self, op_type, in_types, out_types):
        op = op_type.create(
            operands=[_qubit(t) for t in in_types], result_types=list(out_types)
        )
        with pytest.raises(VerifyException):
            op.verify()

    @pytest.mark.parametrize("op_type", TWO_QUBIT_GATES)
    def test_two_qubit_gates_accept_distinct_preserved_qubits(self, op_type):
        op_type(_qubit(PHYS), _qubit(PHYS_1)).verify()

    @pytest.mark.parametrize("op_type", TWO_QUBIT_GATES)
    @pytest.mark.parametrize("types", [(ALG, PHYS), (PHYS, ALG)])
    def test_two_qubit_gates_reject_mixed_qubit_kinds(self, op_type, types):
        op = op_type(*map(_qubit, types))
        with pytest.raises(VerifyException, match="qubits algorithmic or both physical"):
            op.verify()

    @pytest.mark.parametrize("op_type", TWO_QUBIT_GATES)
    def test_two_qubit_gates_reject_the_same_value_twice(self, op_type):
        qubit = _qubit()
        op = op_type(qubit, qubit)
        with pytest.raises(VerifyException, match="same value twice"):
            op.verify()

    @pytest.mark.parametrize("op_type", TWO_QUBIT_GATES)
    @pytest.mark.parametrize(
        "second", [PHYS_1, PhysicalQubitType(IntegerAttr(1, i32))], ids=["i64", "i32"]
    )
    def test_two_qubit_gates_reject_the_same_physical_qubit_twice(self, op_type, second):
        op = op_type(_qubit(PHYS_1), _qubit(second))
        with pytest.raises(VerifyException, match="physical qubit 1 twice"):
            op.verify()

    @pytest.mark.parametrize(
        "in_types, out_types",
        [
            ([ALG], [PHYS]),
            ([PHYS], [PHYS_1]),
            ([ALG, PHYS], [PHYS, ALG]),
            ([ALG, ALG], [ALG]),
        ],
    )
    def test_barrier_rejects_changed_qubits(self, in_types, out_types):
        op = BarrierOp.create(
            operands=[_qubit(t) for t in in_types], result_types=out_types
        )
        with pytest.raises(VerifyException):
            op.verify()

    def test_barrier_accepts_mixed_preserved_qubits(self):
        BarrierOp([_qubit(ALG), _qubit(PHYS)]).verify()


class TestTraits:
    @pytest.mark.parametrize("op_type", CLIFFORD_OPS)
    def test_clifford_ops_have_clifford_trait(self, op_type):
        assert op_type.has_trait(CliffordTrait)

    @pytest.mark.parametrize("op_type", [op for op in ALL_OPS if op not in CLIFFORD_OPS])
    def test_other_ops_do_not_have_clifford_trait(self, op_type):
        assert not op_type.has_trait(CliffordTrait)

    @pytest.mark.parametrize("trait", [Pure, NoMemoryEffect, Commutative])
    @pytest.mark.parametrize("op_type", ALL_OPS)
    def test_no_op_is_pure_or_commutative(self, op_type, trait):
        assert not op_type.has_trait(trait)


class TestDeadCodeElimination:
    """Gates change qubit state, so dead-code elimination must not remove them."""

    def test_keeps_a_gate_whose_result_is_unused(self):
        q = AllocateAlgorithmicQubitOp()
        x = XOp(q.result)
        module = ModuleOp([q, x])

        DeadCodeElimination().apply(_ctx(), module)

        assert x in list(module.walk())

    def test_keeps_a_deallocation(self):
        q = AllocateAlgorithmicQubitOp()
        free = DeallocateQubitOp(q.result)
        module = ModuleOp([q, free])

        DeadCodeElimination().apply(_ctx(), module)

        assert free in list(module.walk())


class TestRoundTrip:
    @pytest.mark.parametrize(
        "make_op",
        [
            lambda: AllocateAlgorithmicQubitOp(),
            lambda: AllocatePhysicalQubitOp(0),
            lambda: DeallocateQubitOp(_qubit()),
            lambda: MeasureOp(_qubit()),
            *((lambda gate=gate: _gate(*gate)) for gate in GATES),
            lambda: ResetOp(_qubit()),
            lambda: IdentityOp(_qubit()),
            lambda: BarrierOp([_qubit(), _qubit()]),
        ],
    )
    def test_op(self, make_op):
        op = make_op()
        assert op.name in _round_trip(op)

    def test_circuit(self):
        """A Bell-pair circuit: H, CX, measure one qubit and deallocate the other."""
        q0 = AllocateAlgorithmicQubitOp()
        q1 = AllocateAlgorithmicQubitOp()
        h = HOp(q0.result)
        cx = CXOp(h.result, q1.result)
        m0 = MeasureOp(cx.result_qubit_control)
        free = DeallocateQubitOp(cx.result_qubit_target)

        module = ModuleOp([q0, q1, h, cx, m0, free])
        module.verify()
        text = _print(module)
        parsed = Parser(_ctx(), text).parse_module()
        parsed.verify()
        assert _print(parsed) == text
