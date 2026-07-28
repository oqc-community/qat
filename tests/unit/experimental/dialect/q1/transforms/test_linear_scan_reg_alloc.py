# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
import pytest
from xdsl.context import Context
from xdsl.dialects.builtin import IntAttr, ModuleOp, NoneAttr
from xdsl.ir import Block, Operation, Region, SSAValue
from xdsl.irdl import irdl_op_definition, operand_def, result_def
from xdsl.utils.exceptions import DiagnosticException

from qat.experimental.dialect.q1 import IntRegisterType, Q1RegisterType
from qat.experimental.dialect.q1.ir.abstract_ops import Q1RegAllocOperation
from qat.experimental.dialect.q1.ir.reg_desc import Registers
from qat.experimental.dialect.q1.transforms.reg_alloc import (
    LinearScanRegisterAllocationPass,
    Q1LinearScanAllocator,
    Q1RegisterStack,
)
from qat.experimental.dialect.q1_cf import BinaryPredicate, UnaryPredicate
from qat.experimental.dialect.q1_scf import ConditionOp, ForOp, IfOp, WhileOp, YieldOp
from qat.experimental.dialect.q1_sequence import SequenceOp

_NUM_REGISTERS = len(Registers.GPR)


def _assert_q1_registers_allocated_in_value(value: SSAValue) -> None:
    if isinstance(value.type, Q1RegisterType):
        assert value.type.index != NoneAttr()
        assert 0 <= value.type.index.data < _NUM_REGISTERS


def _assert_q1_registers_unallocated(op: Operation) -> None:
    for region in op.regions:
        for block in region.blocks:
            for block_arg in block.args:
                if isinstance(block_arg.type, Q1RegisterType):
                    assert block_arg.type.index == NoneAttr()

    for nested_op in op.walk():
        for operand in nested_op.operands:
            if isinstance(operand.type, Q1RegisterType):
                assert operand.type.index == NoneAttr()
        for result in nested_op.results:
            if isinstance(result.type, Q1RegisterType):
                assert result.type.index == NoneAttr()


def _assert_q1_registers_allocated(op: Operation) -> None:
    for region in op.regions:
        for block in region.blocks:
            for block_arg in block.args:
                _assert_q1_registers_allocated_in_value(block_arg)

    for nested_op in op.walk():
        for operand in nested_op.operands:
            _assert_q1_registers_allocated_in_value(operand)
        for result in nested_op.results:
            _assert_q1_registers_allocated_in_value(result)


def _reserved_registers(count: int) -> list[IntRegisterType]:
    return [IntRegisterType.from_index(i) for i in range(count)]


def _allocator(reserved_count: int = 0) -> Q1LinearScanAllocator:
    # TODO: add robustness with COMPILER-1239 and reserved registers
    reserved = _reserved_registers(reserved_count) if reserved_count else None
    stack = Q1RegisterStack.from_reserved_registers(reserved)
    return Q1LinearScanAllocator(stack)


def _linear_producer_consumer_program(num_producers: int) -> list[Operation]:
    producers = [_MockProducerOp() for _ in range(num_producers)]
    consumers = [_MockConsumerOp(producer.result) for producer in producers]
    return [*producers, *consumers]


@irdl_op_definition
class _MockProducerOp(Q1RegAllocOperation):
    """A mock operation that produces a single SSA value of type Q1RegisterType."""

    name = "mock.producer"
    result = result_def(IntRegisterType)

    def __init__(self):
        return super().__init__(result_types=[IntRegisterType.unallocated()])


@irdl_op_definition
class _MockConsumerOp(Q1RegAllocOperation):
    """A mock operation that consumes a single SSA value of type Q1RegisterType."""

    name = "mock.consumer"
    operand = operand_def(IntRegisterType)

    def __init__(self, operand):
        return super().__init__(operands=[operand])


@irdl_op_definition
class _MockProducerConsumerOp(Q1RegAllocOperation):
    """A mock operation that produces and consumes a single SSA value of type
    Q1RegisterType."""

    name = "mock.producer_consumer"
    operand = operand_def(IntRegisterType)
    result = result_def(IntRegisterType)

    def __init__(self, operand):
        return super().__init__(
            operands=[operand], result_types=[IntRegisterType.unallocated()]
        )


@irdl_op_definition
class _MockMultiResultOp(Q1RegAllocOperation):
    """A mock operation that produces two SSA values of type Q1RegisterType."""

    name = "mock.multi_result"
    result_1 = result_def(IntRegisterType)
    result_2 = result_def(IntRegisterType)

    def __init__(self):
        return super().__init__(
            result_types=[IntRegisterType.unallocated(), IntRegisterType.unallocated()]
        )


class TestLinearAllocatorOnSequence:
    """Tests the linear allocator running on a sequence op."""

    def test_sequence_with_no_block_runs(self):
        """Tests that a sequence with no block runs without error."""

        sequence_op = SequenceOp(program=Region(), channel_id="test")
        allocator = _allocator()
        allocator.allocate_sequence(sequence_op)

    def test_sequence_with_exactly_one_block_runs(self):
        """Tests that a sequence with exactly one block runs without error."""

        sequence_op = SequenceOp(program=[], channel_id="test")
        allocator = _allocator()
        allocator.allocate_sequence(sequence_op)

    def test_sequence_with_more_than_one_block_raises_diagnostic_error(self):
        """Tests that a sequence with more than one block raises a diagnostic error."""

        sequence_op = SequenceOp(program=[], channel_id="test")
        sequence_op.body.add_block(Block())
        allocator = _allocator()
        with pytest.raises(
            DiagnosticException,
            match="Q1LinearScanAllocator does not support SequenceOps with more than one block.",
        ):
            allocator.allocate_sequence(sequence_op)


class TestLinearOpList:
    """Tests register allocation with a simple linear list of operations."""

    def test_producer_and_consumer_get_allocated(self):
        """Tests that a producer and consumer operation get allocated a register."""
        allocator = _allocator()
        producer = _MockProducerOp()
        consumer = _MockConsumerOp(producer.result)
        op_list = [producer, consumer]
        sequence_op = SequenceOp(program=op_list, channel_id="test")

        _assert_q1_registers_unallocated(sequence_op)
        allocator.allocate_sequence(sequence_op)
        _assert_q1_registers_allocated(sequence_op)
        assert isinstance(producer.result.type.index, IntAttr)

    @pytest.mark.parametrize("reserved_registers", [0, 5, _NUM_REGISTERS])
    def test_allocator_fails_when_requested_more_registers_than_available(
        self, reserved_registers
    ):
        ops = []
        for _ in range(_NUM_REGISTERS + 1 - reserved_registers):
            ops.append(_MockProducerOp())
        for i in range(_NUM_REGISTERS + 1 - reserved_registers):
            ops.append(_MockConsumerOp(ops[i].result))
        sequence_op = SequenceOp(program=ops, channel_id="test")

        allocator = _allocator(reserved_registers)
        with pytest.raises(DiagnosticException, match="Out of registers."):
            allocator.allocate_sequence(sequence_op)

    def test_reg_chain_preserves_register_allocation(self):
        """Creates a producer and a consumer, followed by another producer, and then a
        consumer on both.

        This tests that the register allocation is not freed until the last consumer is
        reached. We reserve all but two of the registers.
        """

        ops = []
        producer_1 = _MockProducerOp()
        consumer_1 = _MockConsumerOp(producer_1.result)
        producer_2 = _MockProducerOp()
        consumer_2 = _MockConsumerOp(producer_2.result)
        consumer_3 = _MockConsumerOp(producer_1.result)
        ops.extend([producer_1, consumer_1, producer_2, consumer_2, consumer_3])
        sequence_op = SequenceOp(program=ops, channel_id="test")

        allocator = _allocator(_NUM_REGISTERS - 2)
        allocator.allocate_sequence(sequence_op)

        _assert_q1_registers_allocated(sequence_op)

        reg_1 = producer_1.result.type
        reg_2 = producer_2.result.type
        assert reg_1 != reg_2
        assert reg_1.index.data != reg_2.index.data

    def test_register_is_freed(self):
        """Tests that the register is freed after the last consumer is reached, allowing it
        to be reused.

        Reserves all but one register to force that register to be reused.
        """

        ops = []
        producer_1 = _MockProducerOp()
        consumer_1 = _MockConsumerOp(producer_1.result)
        producer_2 = _MockProducerOp()
        consumer_2 = _MockConsumerOp(producer_2.result)
        ops.extend([producer_1, consumer_1, producer_2, consumer_2])
        sequence_op = SequenceOp(program=ops, channel_id="test")

        allocator = _allocator(_NUM_REGISTERS - 1)
        allocator.allocate_sequence(sequence_op)

        _assert_q1_registers_allocated(sequence_op)

        reg_1 = producer_1.result.type
        reg_2 = producer_2.result.type
        assert reg_1 == reg_2

    def test_consumer_producer_reallocates(self):
        """Tests that a consumer followed by a producer reallocates the register."""

        ops = []
        producer = _MockProducerOp()
        prod_consumer = _MockProducerConsumerOp(producer.result)
        consumer = _MockConsumerOp(prod_consumer.result)
        ops.extend([producer, prod_consumer, consumer])
        sequence_op = SequenceOp(program=ops, channel_id="test")

        allocator = _allocator(_NUM_REGISTERS - 1)
        allocator.allocate_sequence(sequence_op)

        _assert_q1_registers_allocated(sequence_op)

        assert producer.result.type == prod_consumer.operand.type
        assert prod_consumer.result.type == consumer.operand.type
        assert consumer.operand.type == producer.result.type

    def test_consumer_producer_reallocates_when_original_producer_has_later_consumer(self):
        """Tests that a consumer followed by a producer reallocates the register, even when
        the original producer has a later consumer.

        This tests that the allocator does not free registers too early.
        """

        producer = _MockProducerOp()
        prod_consumer = _MockProducerConsumerOp(producer.result)
        consumer_1 = _MockConsumerOp(producer.result)
        consumer_2 = _MockConsumerOp(prod_consumer.result)
        sequence_op = SequenceOp(
            program=[producer, prod_consumer, consumer_1, consumer_2], channel_id="test"
        )

        allocator = _allocator(_NUM_REGISTERS - 2)
        allocator.allocate_sequence(sequence_op)

        _assert_q1_registers_allocated(sequence_op)

        assert producer.result.type == prod_consumer.operand.type
        assert prod_consumer.result.type == consumer_2.operand.type
        assert consumer_1.operand.type != prod_consumer.result.type


class TestRegAllocOnForOp:
    """Tests register allocation with a ForOp."""

    def check_boundaries_have_same_allocation(self, for_op: ForOp):
        """Checks that the boundary values of a ForOp have the same register allocation."""

        block = for_op.body.block
        yield_op = block.last_op
        assert isinstance(yield_op, YieldOp)

        assert isinstance(for_op.iter_count.type.index, IntAttr)
        assert for_op.iter_count.type == block.args[0].type

        for iter_arg, block_arg, yield_arg, result in zip(
            for_op.iter_args,
            block.args[1:],
            yield_op.arguments,
            for_op.results,
            strict=True,
        ):
            assert isinstance(result.type.index, IntAttr)
            assert iter_arg.type == block_arg.type
            assert iter_arg.type == yield_arg.type
            assert iter_arg.type == result.type

    def test_allocation_matches_inductor_with_yield_and_block_arg_and_operand(self):
        """Tests that a ForOp with a yield and result has the same register allocation as an
        equivalent inductor with a yield and result."""

        block = Block(arg_types=(IntRegisterType.unallocated(),))
        yield_op = YieldOp()
        block.add_op(yield_op)

        producer = _MockProducerOp()
        for_op = ForOp(iter_count=producer.result, iter_args=[], body=block)
        sequence_op = SequenceOp(program=[producer, for_op], channel_id="test")

        _assert_q1_registers_unallocated(sequence_op)
        allocator = _allocator()
        allocator.allocate_sequence(sequence_op)

        _assert_q1_registers_allocated(sequence_op)
        self.check_boundaries_have_same_allocation(for_op)

    def test_allocation_within_block_uses_unallocated_registers(self):
        """Tests that a ForOp with a yield and result has the same register allocation as an
        equivalent inductor with a yield and result."""

        block = Block(
            arg_types=(IntRegisterType.unallocated(), IntRegisterType.unallocated())
        )
        yield_op = YieldOp(*block.args[1:])
        body_producer1 = _MockProducerOp()
        body_producer2 = _MockProducerOp()
        body_consumer1 = _MockConsumerOp(body_producer1.result)
        body_consumer2 = _MockConsumerOp(body_producer2.result)
        block.add_ops(
            [body_producer1, body_producer2, body_consumer1, body_consumer2, yield_op]
        )

        producer1 = _MockProducerOp()
        producer2 = _MockProducerOp()
        for_op = ForOp(
            iter_count=producer1.result, iter_args=[producer2.result], body=block
        )
        sequence_op = SequenceOp(program=[producer1, producer2, for_op], channel_id="test")

        _assert_q1_registers_unallocated(sequence_op)
        allocator = _allocator()
        allocator.allocate_sequence(sequence_op)

        _assert_q1_registers_allocated(sequence_op)

        # 2 loop-carried registers + 2 body registers = 4 distinct registers
        register_indices = {
            producer1.result.type.index.data,
            producer2.result.type.index.data,
            body_producer1.result.type.index.data,
            body_producer2.result.type.index.data,
        }
        assert len(register_indices) == 4
        self.check_boundaries_have_same_allocation(for_op)

    def test_allocator_fails_with_too_many_producers_in_block(self):
        """Tests that a ForOp with a yield and result has the same register allocation as an
        equivalent inductor with a yield and result."""

        block = Block(arg_types=(IntRegisterType.unallocated(),))
        yield_op = YieldOp()
        body_producer1 = _MockProducerOp()
        body_producer2 = _MockProducerOp()
        body_consumer1 = _MockConsumerOp(body_producer1.result)
        body_consumer2 = _MockConsumerOp(body_producer2.result)
        block.add_ops(
            [body_producer1, body_producer2, body_consumer1, body_consumer2, yield_op]
        )

        producer = _MockProducerOp()
        for_op = ForOp(iter_count=producer.result, iter_args=[], body=block)
        sequence_op = SequenceOp(program=[producer, for_op], channel_id="test")

        allocator = _allocator(_NUM_REGISTERS - 2)

        with pytest.raises(DiagnosticException, match="Out of registers."):
            allocator.allocate_sequence(sequence_op)

    def test_values_defined_outside_of_loop_are_allocated(self):
        """Tests that values defined outside of a ForOp are allocated registers."""

        induction_producer = _MockProducerOp()
        outside_producer_1 = _MockProducerOp()
        outside_producer_2 = _MockProducerOp()
        inside_consumer_1 = _MockConsumerOp(outside_producer_1.result)
        inside_consumer_2 = _MockConsumerOp(outside_producer_2.result)

        block = Block(arg_types=(IntRegisterType.unallocated(),))
        yield_op = YieldOp()
        block.add_ops([inside_consumer_1, inside_consumer_2, yield_op])

        for_op = ForOp(
            iter_count=induction_producer.result,
            iter_args=[],
            body=block,
        )
        sequence_op = SequenceOp(
            program=[
                induction_producer,
                outside_producer_1,
                outside_producer_2,
                for_op,
            ],
            channel_id="test",
        )

        _assert_q1_registers_unallocated(sequence_op)
        allocator = _allocator()
        allocator.allocate_sequence(sequence_op)

        _assert_q1_registers_allocated(sequence_op)

        indices = {
            induction_producer.result.type.index.data,
            outside_producer_1.result.type.index.data,
            outside_producer_2.result.type.index.data,
        }
        assert len(indices) == 3
        self.check_boundaries_have_same_allocation(for_op)

    def test_nested_loop_allocates_different_registers(self):
        """Tests that nested loops allocate different registers."""

        inner_block = Block(arg_types=(IntRegisterType.unallocated(),))
        inner_yield_op = YieldOp()
        inner_block.add_op(inner_yield_op)

        inner_producer = _MockProducerOp()
        inner_for_op = ForOp(
            iter_count=inner_producer.result,
            iter_args=[],
            body=inner_block,
        )

        outer_block = Block(arg_types=(IntRegisterType.unallocated(),))
        outer_yield_op = YieldOp()
        outer_block.add_ops([inner_producer, inner_for_op, outer_yield_op])

        outer_producer = _MockProducerOp()
        outer_for_op = ForOp(
            iter_count=outer_producer.result,
            iter_args=[],
            body=outer_block,
        )
        sequence_op = SequenceOp(
            program=[outer_producer, outer_for_op],
            channel_id="test",
        )

        _assert_q1_registers_unallocated(sequence_op)
        allocator = _allocator()
        allocator.allocate_sequence(sequence_op)

        _assert_q1_registers_allocated(sequence_op)
        self.check_boundaries_have_same_allocation(inner_for_op)
        self.check_boundaries_have_same_allocation(outer_for_op)

        # Check the inner and outer induction variables are allocated to different registers
        assert inner_producer.result.type != outer_producer.result.type

    def test_consumer_and_producer_in_loop_body_allocates_successfully(
        self,
    ):
        """Tests that a ForOp body with consumer-producer ops and multiple carried values
        allocates successfully."""

        block = Block(
            arg_types=(
                IntRegisterType.unallocated(),
                IntRegisterType.unallocated(),
                IntRegisterType.unallocated(),
            )
        )
        consumer_producer_1 = _MockProducerConsumerOp(block.args[1])
        consumer_producer_2 = _MockProducerConsumerOp(block.args[2])
        consumer = _MockConsumerOp(consumer_producer_1.result)
        yield_op = YieldOp(consumer_producer_1.result, consumer_producer_2.result)
        block.add_ops([consumer_producer_1, consumer_producer_2, consumer, yield_op])

        producer_1 = _MockProducerOp()
        producer_2 = _MockProducerOp()
        producer_3 = _MockProducerOp()
        for_op = ForOp(
            iter_count=producer_1.result,
            iter_args=[producer_2.result, producer_3.result],
            body=block,
        )
        sequence_op = SequenceOp(
            program=[producer_1, producer_2, producer_3, for_op],
            channel_id="test",
        )

        allocator = _allocator()
        allocator.allocate_sequence(sequence_op)

        _assert_q1_registers_allocated(sequence_op)
        self.check_boundaries_have_same_allocation(for_op)

    def test_tangle_block_args_with_consumer_producer_chain_allocates_successfully(self):
        """Tests when block args and yield operands are tangled through consumer-producer
        chains, the allocation passes.

        %block_arg_1 --> consumer_producer_1 ----> consumer_producer_2 ----> yield_arg_2
        %block_arg_2 ------> consumer_producer_3 ----> consumer_producer_4 ----> yield_arg_1

        The staggering indicates the order they happen in the block.

        This works because additional registers can be allocated between the consumer
        producers, meaning the block arguments aren't overwritten until they're needed.
        """

        block = Block(
            arg_types=(
                IntRegisterType.unallocated(),
                IntRegisterType.unallocated(),
                IntRegisterType.unallocated(),
            )
        )
        consumer_producer_1 = _MockProducerConsumerOp(block.args[1])
        consumer_producer_2 = _MockProducerConsumerOp(consumer_producer_1.result)
        consumer_producer_3 = _MockProducerConsumerOp(block.args[2])
        consumer_producer_4 = _MockProducerConsumerOp(consumer_producer_3.result)
        yield_op = YieldOp(consumer_producer_4.result, consumer_producer_2.result)
        block.add_ops(
            [
                consumer_producer_1,
                consumer_producer_3,
                consumer_producer_2,
                consumer_producer_4,
                yield_op,
            ]
        )

        producer_1 = _MockProducerOp()
        producer_2 = _MockProducerOp()
        producer_3 = _MockProducerOp()
        for_op = ForOp(
            iter_count=producer_1.result,
            iter_args=[producer_2.result, producer_3.result],
            body=block,
        )
        sequence_op = SequenceOp(
            program=[producer_1, producer_2, producer_3, for_op],
            channel_id="test",
        )

        allocator = _allocator()
        _assert_q1_registers_unallocated(sequence_op)
        allocator.allocate_sequence(sequence_op)

    def test_multi_result_producer_checks_all_yield_lanes(self):
        """Tests that a producer with multiple results validates every yielded lane.

        If one carried lane is used after the producer and another is not, the validation
        must still catch the bad lane instead of overwriting its index.
        """

        block = Block(
            arg_types=(
                IntRegisterType.unallocated(),
                IntRegisterType.unallocated(),
                IntRegisterType.unallocated(),
            )
        )
        multi_result_op = _MockMultiResultOp()
        consumer = _MockConsumerOp(block.args[1])
        yield_op = YieldOp(multi_result_op.result_1, multi_result_op.result_2)
        block.add_ops([multi_result_op, consumer, yield_op])

        producer_1 = _MockProducerOp()
        producer_2 = _MockProducerOp()
        producer_3 = _MockProducerOp()
        for_op = ForOp(
            iter_count=producer_1.result,
            iter_args=[producer_2.result, producer_3.result],
            body=block,
        )
        sequence_op = SequenceOp(
            program=[producer_1, producer_2, producer_3, for_op],
            channel_id="test",
        )

        allocator = _allocator()
        with pytest.raises(
            DiagnosticException,
            match=(
                "ForOp carried block argument is used after the operation that "
                "defines the corresponding yield operand. Register allocation is "
                "not possible without relocation, which is expected to happen upstream."
            ),
        ):
            allocator.allocate_sequence(sequence_op)


class TestForOpUnsupportedAllocations:
    """Tests the different scenarios that are unsupported with lowering for ops."""

    def test_for_op_with_iter_args_that_have_multiple_uses_raises_diagnostic_error(self):
        """Tests that a ForOp with iter args that have uses after the ForOp raises a
        diagnostic error."""

        block = Block(
            arg_types=(
                IntRegisterType.unallocated(),
                IntRegisterType.unallocated(),
            )
        )
        yield_op = YieldOp(*block.args[1:])
        block.add_op(yield_op)

        producer_1 = _MockProducerOp()
        producer_2 = _MockProducerOp()
        for_op = ForOp(
            iter_count=producer_1.result,
            iter_args=[producer_2.result],
            body=block,
        )
        consumer_1 = _MockConsumerOp(producer_2.result)
        sequence_op = SequenceOp(
            program=[producer_1, producer_2, for_op, consumer_1],
            channel_id="test",
        )

        allocator = _allocator()
        with pytest.raises(
            DiagnosticException,
            match="implies that the iter_arg is used within the body of the loop or ",
        ):
            allocator.allocate_sequence(sequence_op)

    def test_iter_arg_used_directly_in_loop_raises(self):
        """Tests that a ForOp with an iter arg that is used directly in the loop raises a
        diagnostic error."""

        block = Block(
            arg_types=(
                IntRegisterType.unallocated(),
                IntRegisterType.unallocated(),
            )
        )

        producer_1 = _MockProducerOp()
        producer_2 = _MockProducerOp()
        consumer = _MockConsumerOp(producer_2.result)
        yield_op = YieldOp(*block.args[1:])
        block.add_ops([consumer, yield_op])

        for_op = ForOp(
            iter_count=producer_1.result,
            iter_args=[producer_2.result],
            body=block,
        )
        sequence_op = SequenceOp(
            program=[producer_1, producer_2, for_op],
            channel_id="test",
        )

        allocator = _allocator()
        with pytest.raises(
            DiagnosticException,
            match="implies that the iter_arg is used within the body of the loop or ",
        ):
            allocator.allocate_sequence(sequence_op)

    def test_yielding_live_in_value_allocates_successfully(self):
        """Tests that yielding a value defined outside the loop body allocates.

        The body still uses the carried block argument, which is the path that used to
        trigger a KeyError during validation.
        """

        external_producer = _MockProducerOp()

        block = Block(
            arg_types=(
                IntRegisterType.unallocated(),
                IntRegisterType.unallocated(),
            )
        )
        _induction, carried = block.args
        consumer = _MockConsumerOp(carried)
        block.add_ops([consumer, YieldOp(external_producer.result)])

        init_producer = _MockProducerOp()
        iter_producer = _MockProducerOp()
        for_op = ForOp(
            iter_count=init_producer.result,
            iter_args=[iter_producer.result],
            body=block,
        )
        sequence_op = SequenceOp(
            program=[external_producer, init_producer, iter_producer, for_op],
            channel_id="test",
        )

        _assert_q1_registers_unallocated(sequence_op)
        allocator = _allocator()
        allocator.allocate_sequence(sequence_op)
        _assert_q1_registers_allocated(sequence_op)

    def test_induction_variable_with_direct_use_within_loop_raises_diagnostic_error(self):
        """Tests that a ForOp with an induction variable that has a use within the loop
        raises a diagnostic error."""

        producer = _MockProducerOp()
        block = Block(arg_types=(IntRegisterType.unallocated(),))
        consumer = _MockConsumerOp(producer.result)
        yield_op = YieldOp()
        block.add_ops([consumer, yield_op])

        for_op = ForOp(
            iter_count=producer.result,
            iter_args=[],
            body=block,
        )
        sequence_op = SequenceOp(
            program=[producer, for_op],
            channel_id="test",
        )

        allocator = _allocator()
        with pytest.raises(
            DiagnosticException,
            match="implies that the iter_count is used within the body of the loop or",
        ):
            allocator.allocate_sequence(sequence_op)

    def test_for_op_with_induction_variable_with_use_after_loop_raises_diagnostic_error(
        self,
    ):
        """Tests that a ForOp with an induction variable that has multiple uses raises a
        diagnostic error."""

        block = Block(arg_types=(IntRegisterType.unallocated(),))
        yield_op = YieldOp()
        block.add_op(yield_op)

        producer = _MockProducerOp()
        for_op = ForOp(
            iter_count=producer.result,
            iter_args=[],
            body=block,
        )
        consumer_1 = _MockConsumerOp(producer.result)
        sequence_op = SequenceOp(
            program=[producer, for_op, consumer_1],
            channel_id="test",
        )

        allocator = _allocator()
        with pytest.raises(
            DiagnosticException,
            match="implies that the iter_count is used within the body of the loop or",
        ):
            allocator.allocate_sequence(sequence_op)

    def test_for_op_with_block_args_that_have_multiple_uses_raises_diagnostic_error(self):
        """Tests that a ForOp with block args that have multiple uses raises a diagnostic
        error.

        One particularly telling example is the following situation:

        %intermediate_arg = mock.consumer_producer %block_arg_1;
        mock.consumer %block_arg_1;
        q1_scf.yield %intermediate_arg;

        Or diagrammatically:

        block_arg_1 ---------> consumer_producer -----------> yield
                        |
                        -------------------------> consumer

        Allocating the block arg and the yield to the same register constraints the
        consumer_producer to use that register for both its operand and result, reallocating
        that register. That means that value is no longer available for the consumer.
        """

        block = Block(
            arg_types=(
                IntRegisterType.unallocated(),
                IntRegisterType.unallocated(),
            )
        )
        consumer_producer = _MockProducerConsumerOp(block.args[1])
        consumer = _MockConsumerOp(block.args[1])
        yield_op = YieldOp(consumer_producer.result)
        block.add_ops([consumer_producer, consumer, yield_op])

        producer_1 = _MockProducerOp()
        producer_2 = _MockProducerOp()
        for_op = ForOp(
            iter_count=producer_1.result,
            iter_args=[producer_2.result],
            body=block,
        )
        sequence_op = SequenceOp(
            program=[producer_1, producer_2, for_op],
            channel_id="test",
        )
        allocator = _allocator()
        with pytest.raises(
            DiagnosticException,
            match=(
                "ForOp carried block argument is used after the operation that "
                "defines the corresponding yield operand. Register allocation is "
                "not possible without relocation, which is expected to happen upstream."
            ),
        ):
            allocator.allocate_sequence(sequence_op)

    def test_for_op_with_block_args_that_have_multiple_uses_with_nested_loop(self):
        """Same as the previous test, but the consumer is within a nested loop as a live
        in."""

        outer_block = Block(
            arg_types=(
                IntRegisterType.unallocated(),
                IntRegisterType.unallocated(),
            )
        )
        outer_inductor = _MockProducerOp()
        outer_producer = _MockProducerOp()
        consumer_producer = _MockProducerConsumerOp(outer_block.args[1])

        inner_inductor = _MockProducerOp()
        inner_block = Block(arg_types=(IntRegisterType.unallocated(),))
        consumer = _MockConsumerOp(outer_block.args[1])
        inner_yield_op = YieldOp()
        inner_block.add_ops([consumer, inner_yield_op])
        inner_for_op = ForOp(
            iter_count=inner_inductor.result,
            iter_args=[],
            body=inner_block,
        )

        outer_yield_op = YieldOp(consumer_producer.result)
        outer_block.add_ops(
            [inner_inductor, consumer_producer, inner_for_op, outer_yield_op]
        )

        outer_for_op = ForOp(
            iter_count=outer_inductor.result,
            iter_args=[outer_producer.result],
            body=outer_block,
        )
        sequence_op = SequenceOp(
            program=[outer_inductor, outer_producer, outer_for_op],
            channel_id="test",
        )
        allocator = _allocator()
        with pytest.raises(
            DiagnosticException,
            match=(
                "ForOp carried block argument is used after the operation that "
                "defines the corresponding yield operand. Register allocation is "
                "not possible without relocation, which is expected to happen upstream."
            ),
        ):
            allocator.allocate_sequence(sequence_op)

    def test_for_op_with_consumer_producer_chain_and_later_block_arg_use_raises_diagnostic_error(
        self,
    ):
        """Tests that a ForOp with a consumer-producer chain and a later block arg use
        raises a diagnostic error.

        %block_arg_1 ---------> consumer_producer --------> consumer_producer -----> yield
                        |
                        ----------------------------------------------------> consumer

        The coalescing of the block arg and the yield to the same register constrains the
        yield operand and the block arg to be allocated to the same register. The second
        consumer and producer will reallocate that register, meaning that the value is no
        longer available for the consumer.
        """

        block = Block(
            arg_types=(
                IntRegisterType.unallocated(),
                IntRegisterType.unallocated(),
            )
        )
        consumer_producer_1 = _MockProducerConsumerOp(block.args[1])
        consumer_producer_2 = _MockProducerConsumerOp(consumer_producer_1.result)
        consumer = _MockConsumerOp(block.args[1])
        yield_op = YieldOp(consumer_producer_2.result)
        block.add_ops([consumer_producer_1, consumer_producer_2, consumer, yield_op])

        producer_1 = _MockProducerOp()
        producer_2 = _MockProducerOp()
        for_op = ForOp(
            iter_count=producer_1.result,
            iter_args=[producer_2.result],
            body=block,
        )
        sequence_op = SequenceOp(
            program=[producer_1, producer_2, for_op],
            channel_id="test",
        )
        allocator = _allocator()
        with pytest.raises(
            DiagnosticException,
            match=(
                "ForOp carried block argument is used after the operation that "
                "defines the corresponding yield operand. Register allocation is "
                "not possible without relocation, which is expected to happen upstream."
            ),
        ):
            allocator.allocate_sequence(sequence_op)

    def test_tangled_block_args_and_yield_args_raises_diagnostic_error(self):
        """Tests that a ForOp with block args and yield args that are tangled raises a
        diagnostic error.

        One particularly telling example is the following situation:

        %block_arg_1 ---------> yield_arg_2
        %block_arg_2 ---------> yield_arg_1

        The flow of the program demands that allocation, but the allocator simultaneously
        wants to allocate each block arg to the same register as its corresponding yield
        arg. This causes a contradiction, and the allocator will raise a diagnostic error.
        """

        block = Block(
            arg_types=(
                IntRegisterType.unallocated(),
                IntRegisterType.unallocated(),
                IntRegisterType.unallocated(),
            )
        )
        yield_op = YieldOp(block.args[2], block.args[1])
        block.add_op(yield_op)

        producer_1 = _MockProducerOp()
        producer_2 = _MockProducerOp()
        producer_3 = _MockProducerOp()
        for_op = ForOp(
            iter_count=producer_1.result,
            iter_args=[producer_2.result, producer_3.result],
            body=block,
        )
        sequence_op = SequenceOp(
            program=[producer_1, producer_2, producer_3, for_op],
            channel_id="test",
        )

        allocator = _allocator()
        with pytest.raises(
            DiagnosticException,
            match=(
                "A yield operand found with a block argument at an index that does not "
                "match the index of the block argument. Register allocation is not "
                "possible without relocation, which is expected to happen upstream."
            ),
        ):
            allocator.allocate_sequence(sequence_op)

    def test_tangle_block_args_with_consumer_producer_chain_raises_diagnostic_error(self):
        """Tests that a ForOp with block args and yield args that are tangled through
        consumer producers with a later consumer raises a diagnostic error.

        %block_arg_1 ---------> consumer_producer_1 --------> yield_arg_2
                        |
                        -------------------------> consumer_1

        %block_arg_2 ---------> consumer_producer_2 --------> yield_arg_1
        """

        block = Block(
            arg_types=(
                IntRegisterType.unallocated(),
                IntRegisterType.unallocated(),
                IntRegisterType.unallocated(),
            )
        )
        consumer_producer_1 = _MockProducerConsumerOp(block.args[1])
        consumer_producer_2 = _MockProducerConsumerOp(block.args[2])
        consumer = _MockConsumerOp(block.args[1])
        yield_op = YieldOp(consumer_producer_2.result, consumer_producer_1.result)
        block.add_ops([consumer_producer_1, consumer_producer_2, consumer, yield_op])

        producer_1 = _MockProducerOp()
        producer_2 = _MockProducerOp()
        producer_3 = _MockProducerOp()
        for_op = ForOp(
            iter_count=producer_1.result,
            iter_args=[producer_2.result, producer_3.result],
            body=block,
        )
        sequence_op = SequenceOp(
            program=[producer_1, producer_2, producer_3, for_op],
            channel_id="test",
        )

        allocator = _allocator()
        with pytest.raises(
            DiagnosticException,
            match=(
                "ForOp carried block argument is used after the operation that "
                "defines the corresponding yield operand. Register allocation is "
                "not possible without relocation, which is expected to happen upstream."
            ),
        ):
            allocator.allocate_sequence(sequence_op)

    def test_tangle_block_args_with_parallel_consumer_producer_chain_raises_diagnostic_error(
        self,
    ):
        """Tests when block args and yield operands are tangled through consumer-producer
        chains, the allocation passes.

        %block_arg_1 ---------> consumer_producer_1 --------> yield_arg_2

        %block_arg_2 ---------> consumer_producer_2 --------> yield_arg_1
        """

        block = Block(
            arg_types=(
                IntRegisterType.unallocated(),
                IntRegisterType.unallocated(),
                IntRegisterType.unallocated(),
            )
        )
        consumer_producer_1 = _MockProducerConsumerOp(block.args[1])
        consumer_producer_2 = _MockProducerConsumerOp(block.args[2])
        yield_op = YieldOp(consumer_producer_2.result, consumer_producer_1.result)
        block.add_ops([consumer_producer_1, consumer_producer_2, yield_op])

        producer_1 = _MockProducerOp()
        producer_2 = _MockProducerOp()
        producer_3 = _MockProducerOp()
        for_op = ForOp(
            iter_count=producer_1.result,
            iter_args=[producer_2.result, producer_3.result],
            body=block,
        )
        sequence_op = SequenceOp(
            program=[producer_1, producer_2, producer_3, for_op],
            channel_id="test",
        )

        allocator = _allocator()
        with pytest.raises(
            DiagnosticException,
            match=(
                "ForOp carried block argument is used after the operation that "
                "defines the corresponding yield operand. Register allocation is "
                "not possible without relocation, which is expected to happen upstream."
            ),
        ):
            allocator.allocate_sequence(sequence_op)


class TestRegAllocOnIfOp:
    """Tests register allocation with an IfOp."""

    def check_boundaries_have_same_allocation(self, if_op: IfOp):
        """Checks that the boundary values of an IfOp have the same register allocation.

        Verifies that each yield operand in every region that has a yield terminator is
        coalesced with the corresponding IfOp result.
        """

        for region in (if_op.then_region, if_op.else_region):
            if not region.blocks:
                continue
            yield_op = region.blocks[0].last_op
            assert isinstance(yield_op, YieldOp)
            for yield_arg, result in zip(yield_op.arguments, if_op.output, strict=True):
                assert isinstance(result.type.index, IntAttr)
                assert yield_arg.type == result.type

    def test_if_op_with_no_else_region_and_single_operand_allocates_without_error(self):
        """Tests that an IfOp with no else region allocates without error, making sure that
        the predicates are allocated."""

        producer_1 = _MockProducerOp()
        producer_2 = _MockProducerOp()
        producer_3 = _MockProducerOp()
        consumer_1 = _MockConsumerOp(producer_2.result)
        consumer_2 = _MockConsumerOp(producer_3.result)
        yield_op = YieldOp()
        then_block = Block([producer_2, producer_3, consumer_1, consumer_2, yield_op])
        if_op = IfOp(
            predicate=UnaryPredicate.eqz,
            predicate_args=[producer_1.result],
            then_region=[then_block],
            result_types=[],
        )

        sequence_op = SequenceOp(
            program=[producer_1, if_op],
            channel_id="test",
        )

        _assert_q1_registers_unallocated(sequence_op)
        allocator = _allocator(_NUM_REGISTERS - 2)
        allocator.allocate_sequence(sequence_op)

        # check register allocations; should only use two registers
        _assert_q1_registers_allocated(sequence_op)
        inner_block_registers = {
            producer_2.result.type.index.data,
            producer_3.result.type.index.data,
        }
        assert len(inner_block_registers) == 2
        assert producer_1.result.type.index.data in inner_block_registers
        self.check_boundaries_have_same_allocation(if_op)

    def test_if_op_with_no_else_region_and_two_operands_allocates_without_error(self):
        """Tests that an IfOp with no else region allocates without error, making sure that
        the predicates are allocated."""

        producer_1 = _MockProducerOp()
        producer_2 = _MockProducerOp()
        producer_3 = _MockProducerOp()
        producer_4 = _MockProducerOp()
        consumer_1 = _MockConsumerOp(producer_3.result)
        consumer_2 = _MockConsumerOp(producer_4.result)
        yield_op = YieldOp()
        then_block = Block([producer_3, producer_4, consumer_1, consumer_2, yield_op])
        if_op = IfOp(
            predicate=BinaryPredicate.eq,
            predicate_args=[producer_1.result, producer_2.result],
            then_region=[then_block],
            result_types=[],
        )

        sequence_op = SequenceOp(
            program=[producer_1, producer_2, if_op],
            channel_id="test",
        )

        _assert_q1_registers_unallocated(sequence_op)
        allocator = _allocator(_NUM_REGISTERS - 2)
        allocator.allocate_sequence(sequence_op)

        # check register allocations; should only use two registers
        _assert_q1_registers_allocated(sequence_op)

        inner_block_registers = {
            producer_1.result.type.index.data,
            producer_2.result.type.index.data,
        }
        assert len(inner_block_registers) == 2
        outer_block_registers = {
            producer_3.result.type.index.data,
            producer_4.result.type.index.data,
        }
        assert len(outer_block_registers) == 2
        assert inner_block_registers == outer_block_registers
        self.check_boundaries_have_same_allocation(if_op)

    def test_if_op_with_no_else_region_and_overallocation_raises_diagnostic_error(self):
        """Tests that an IfOp with no else region raises a diagnostic error when the
        allocation exceeds the number of available registers."""

        producer_1 = _MockProducerOp()
        producer_2 = _MockProducerOp()
        producer_3 = _MockProducerOp()
        consumer_1 = _MockConsumerOp(producer_2.result)
        consumer_2 = _MockConsumerOp(producer_3.result)
        yield_op = YieldOp()
        then_block = Block([producer_2, producer_3, consumer_1, consumer_2, yield_op])
        if_op = IfOp(
            predicate=UnaryPredicate.eqz,
            predicate_args=[producer_1.result],
            then_region=[then_block],
            result_types=[],
        )

        sequence_op = SequenceOp(
            program=[producer_1, if_op],
            channel_id="test",
        )

        allocator = _allocator(_NUM_REGISTERS - 1)
        with pytest.raises(DiagnosticException, match="Out of registers."):
            allocator.allocate_sequence(sequence_op)

    def test_if_op_with_else_region_allocates_without_error(self):
        """Tests that an IfOp with an else region allocates without error, making sure that
        the predicates are allocated and the yield operand registers match the outputs."""

        producer_1 = _MockProducerOp()
        producer_2 = _MockProducerOp()
        producer_3 = _MockProducerOp()
        producer_4 = _MockProducerOp()
        producer_5 = _MockProducerOp()

        yield_1 = YieldOp(producer_2.result, producer_3.result)
        yield_2 = YieldOp(producer_4.result, producer_5.result)

        then_block = Block([producer_2, producer_3, yield_1])
        else_block = Block([producer_4, producer_5, yield_2])

        if_op = IfOp(
            predicate=UnaryPredicate.eqz,
            predicate_args=[producer_1.result],
            then_region=[then_block],
            else_region=[else_block],
            result_types=[IntRegisterType.unallocated(), IntRegisterType.unallocated()],
        )

        consumer_1 = _MockConsumerOp(if_op.output[0])
        consumer_2 = _MockConsumerOp(if_op.output[1])

        sequence_op = SequenceOp(
            program=[producer_1, if_op, consumer_1, consumer_2],
            channel_id="test",
        )

        _assert_q1_registers_unallocated(sequence_op)
        allocator = _allocator(_NUM_REGISTERS - 2)
        allocator.allocate_sequence(sequence_op)

        # check the allocations afterwards
        _assert_q1_registers_allocated(sequence_op)
        self.check_boundaries_have_same_allocation(if_op)

        # check the producers match the yields
        assert producer_2.result.type == yield_1.arguments[0].type
        assert producer_3.result.type == yield_1.arguments[1].type
        assert producer_4.result.type == yield_2.arguments[0].type
        assert producer_5.result.type == yield_2.arguments[1].type

        # check the consumers match the if_op outputs
        assert consumer_1.operand.type == if_op.output[0].type
        assert consumer_2.operand.type == if_op.output[1].type

    def test_if_op_with_else_region_and_overallocation_raises_diagnostic_error(self):
        """Tests that an IfOp with an else region raises a diagnostic error when the
        allocation exceeds the number of available registers."""

        producer_1 = _MockProducerOp()
        producer_2 = _MockProducerOp()
        producer_3 = _MockProducerOp()
        producer_4 = _MockProducerOp()
        consumer_1 = _MockConsumerOp(producer_3.result)

        yield_1 = YieldOp(producer_2.result)
        yield_2 = YieldOp(producer_4.result)

        then_block = Block([producer_2, producer_3, consumer_1, yield_1])
        else_block = Block([producer_4, yield_2])

        if_op = IfOp(
            predicate=UnaryPredicate.eqz,
            predicate_args=[producer_1.result],
            then_region=[then_block],
            else_region=[else_block],
            result_types=[IntRegisterType.unallocated()],
        )

        consumer_2 = _MockConsumerOp(if_op.output[0])

        sequence_op = SequenceOp(
            program=[producer_1, if_op, consumer_2],
            channel_id="test",
        )

        allocator = _allocator(_NUM_REGISTERS - 1)
        with pytest.raises(DiagnosticException, match="Out of registers."):
            allocator.allocate_sequence(sequence_op)

    def test_if_op_with_live_ins(self):
        """Tests that an IfOp with live ins allocates without error, making sure that the
        predicates are allocated and the yield operand registers match the outputs."""

        producer_1 = _MockProducerOp()
        producer_2 = _MockProducerOp()
        producer_3 = _MockProducerOp()

        consumer_1 = _MockConsumerOp(producer_1.result)
        consumer_2 = _MockConsumerOp(producer_2.result)
        consumer_producer = _MockProducerConsumerOp(producer_2.result)

        yield_1 = YieldOp(consumer_producer.result)
        yield_2 = YieldOp(producer_3.result)

        block_1 = Block([consumer_1, consumer_producer, yield_1])
        block_2 = Block([consumer_2, yield_2])

        if_op = IfOp(
            predicate=UnaryPredicate.eqz,
            predicate_args=[producer_1.result],
            then_region=[block_1],
            else_region=[block_2],
            result_types=[IntRegisterType.unallocated()],
        )
        consumer_3 = _MockConsumerOp(if_op.output[0])

        sequence_op = SequenceOp(
            program=[producer_1, producer_2, producer_3, if_op, consumer_3],
            channel_id="test",
        )

        _assert_q1_registers_unallocated(sequence_op)
        allocator = _allocator(_NUM_REGISTERS - 3)
        allocator.allocate_sequence(sequence_op)

        # Check register allocation after allocation
        _assert_q1_registers_allocated(sequence_op)
        self.check_boundaries_have_same_allocation(if_op)

        # Check the former producers have the correct register allocation
        assert producer_1.result.type == if_op.predicate_args[0].type
        assert producer_2.result.type == consumer_producer.operand.type
        assert producer_2.result.type == consumer_2.operand.type

        # Check the consumer has the correct register allocation
        assert consumer_3.operand.type == if_op.output[0].type

    def test_if_op_with_live_ins_and_overallocation_raises_diagnostic_error(self):
        """Tests that an IfOp with live ins raises a diagnostic error when the allocation
        exceeds the number of available registers."""

        producer_1 = _MockProducerOp()
        producer_2 = _MockProducerOp()
        producer_3 = _MockProducerOp()

        consumer_1 = _MockConsumerOp(producer_1.result)
        consumer_2 = _MockConsumerOp(producer_2.result)
        consumer_producer = _MockProducerConsumerOp(producer_2.result)

        yield_1 = YieldOp(consumer_producer.result)
        yield_2 = YieldOp(producer_3.result)

        block_1 = Block([consumer_1, consumer_producer, yield_1])
        block_2 = Block([consumer_2, yield_2])

        if_op = IfOp(
            predicate=UnaryPredicate.eqz,
            predicate_args=[producer_1.result],
            then_region=[block_1],
            else_region=[block_2],
            result_types=[IntRegisterType.unallocated()],
        )
        consumer_3 = _MockConsumerOp(if_op.output[0])

        sequence_op = SequenceOp(
            program=[producer_1, producer_2, producer_3, if_op, consumer_3],
            channel_id="test",
        )

        allocator = _allocator(_NUM_REGISTERS - 2)
        with pytest.raises(DiagnosticException, match="Out of registers."):
            allocator.allocate_sequence(sequence_op)


class TestRegAllocOnWhileOp:
    """Tests register allocation with a WhileOp."""

    def check_boundaries_have_same_allocation(self, while_op: WhileOp):
        """Checks that the boundary values of a WhileOp have the same register
        allocation."""

        before = while_op.before_region.block
        after = while_op.after_region.block
        condition = before.last_op
        assert isinstance(condition, ConditionOp)
        yield_op = after.last_op
        assert isinstance(yield_op, YieldOp)

        for forward_arg, after_arg, res in zip(
            condition.forward_args, after.args, while_op.res, strict=True
        ):
            assert isinstance(res.type.index, IntAttr)
            assert res.type == forward_arg.type
            assert res.type == after_arg.type

        for backward_arg, before_arg in zip(yield_op.arguments, before.args, strict=True):
            assert isinstance(backward_arg.type.index, IntAttr)
            assert backward_arg.type == before_arg.type

    def test_basic_while_with_single_carried_value_allocates(self):
        """Tests that a basic WhileOp with a single carried value allocates all
        registers."""

        init_producer = _MockProducerOp()

        before = Block(arg_types=[IntRegisterType.unallocated()])
        (acc,) = before.args
        before.add_op(ConditionOp(UnaryPredicate.nez, [acc], [acc]))

        after = Block(arg_types=[IntRegisterType.unallocated()])
        (val,) = after.args
        after.add_op(YieldOp(val))

        while_op = WhileOp(
            [init_producer.result],
            [IntRegisterType.unallocated()],
            Region([before]),
            Region([after]),
        )
        sequence_op = SequenceOp(program=[init_producer, while_op], channel_id="test")

        _assert_q1_registers_unallocated(sequence_op)
        allocator = _allocator()
        allocator.allocate_sequence(sequence_op)
        _assert_q1_registers_allocated(sequence_op)
        self.check_boundaries_have_same_allocation(while_op)

    def test_init_arg_allocates_with_loop_result(self):
        """Tests that the loop init argument and the WhileOp result share a register."""

        init_producer = _MockProducerOp()

        before = Block(arg_types=[IntRegisterType.unallocated()])
        (acc,) = before.args
        before.add_op(ConditionOp(UnaryPredicate.nez, [acc], [acc]))

        after = Block(arg_types=[IntRegisterType.unallocated()])
        (val,) = after.args
        after.add_op(YieldOp(val))

        while_op = WhileOp(
            [init_producer.result],
            [IntRegisterType.unallocated()],
            Region([before]),
            Region([after]),
        )
        sequence_op = SequenceOp(program=[init_producer, while_op], channel_id="test")

        allocator = _allocator()
        allocator.allocate_sequence(sequence_op)

        self.check_boundaries_have_same_allocation(while_op)
        assert isinstance(init_producer.result.type.index, IntAttr)
        assert init_producer.result.type == while_op.res[0].type

    def test_predicate_live_in_allocates_without_error(self):
        """Tests that a value used by the before-region predicate is allocated."""

        pred_producer = _MockProducerOp()
        init_producer = _MockProducerOp()

        before = Block(arg_types=[IntRegisterType.unallocated()])
        (acc,) = before.args
        before.add_op(ConditionOp(UnaryPredicate.nez, [pred_producer.result], [acc]))

        after = Block(arg_types=[IntRegisterType.unallocated()])
        (val,) = after.args
        after.add_op(YieldOp(val))

        while_op = WhileOp(
            [init_producer.result],
            [IntRegisterType.unallocated()],
            Region([before]),
            Region([after]),
        )
        sequence_op = SequenceOp(
            program=[pred_producer, init_producer, while_op], channel_id="test"
        )

        allocator = _allocator()
        allocator.allocate_sequence(sequence_op)

        self.check_boundaries_have_same_allocation(while_op)
        assert isinstance(pred_producer.result.type.index, IntAttr)
        assert init_producer.result.type == while_op.res[0].type

    def test_multiple_carried_values_get_distinct_registers(self):
        """Tests that multiple carried values each get their own distinct register."""

        init_producer_1 = _MockProducerOp()
        init_producer_2 = _MockProducerOp()

        before = Block(
            arg_types=[IntRegisterType.unallocated(), IntRegisterType.unallocated()]
        )
        acc_a, acc_b = before.args
        before.add_op(ConditionOp(UnaryPredicate.nez, [acc_a], [acc_a, acc_b]))

        after = Block(
            arg_types=[IntRegisterType.unallocated(), IntRegisterType.unallocated()]
        )
        val_a, val_b = after.args
        after.add_op(YieldOp(val_a, val_b))

        while_op = WhileOp(
            [init_producer_1.result, init_producer_2.result],
            [IntRegisterType.unallocated(), IntRegisterType.unallocated()],
            Region([before]),
            Region([after]),
        )
        sequence_op = SequenceOp(
            program=[init_producer_1, init_producer_2, while_op], channel_id="test"
        )

        allocator = _allocator()
        allocator.allocate_sequence(sequence_op)

        _assert_q1_registers_allocated(sequence_op)
        self.check_boundaries_have_same_allocation(while_op)
        assert init_producer_1.result.type != init_producer_2.result.type

    def test_body_ops_in_before_region_are_allocated(self):
        """Tests that body ops in the before region get registers allocated."""

        init_producer = _MockProducerOp()

        before = Block(arg_types=[IntRegisterType.unallocated()])
        (acc,) = before.args
        body_producer = _MockProducerOp()
        body_consumer = _MockConsumerOp(body_producer.result)
        before.add_ops(
            [body_producer, body_consumer, ConditionOp(UnaryPredicate.nez, [acc], [acc])]
        )

        after = Block(arg_types=[IntRegisterType.unallocated()])
        (val,) = after.args
        after.add_op(YieldOp(val))

        while_op = WhileOp(
            [init_producer.result],
            [IntRegisterType.unallocated()],
            Region([before]),
            Region([after]),
        )
        sequence_op = SequenceOp(program=[init_producer, while_op], channel_id="test")

        allocator = _allocator()
        allocator.allocate_sequence(sequence_op)
        _assert_q1_registers_allocated(sequence_op)

        self.check_boundaries_have_same_allocation(while_op)

    def test_body_ops_in_after_region_are_allocated(self):
        """Tests that body ops in the after region get registers allocated."""

        init_producer = _MockProducerOp()

        before = Block(arg_types=[IntRegisterType.unallocated()])
        (acc,) = before.args
        before.add_op(ConditionOp(UnaryPredicate.nez, [acc], [acc]))

        after = Block(arg_types=[IntRegisterType.unallocated()])
        (val,) = after.args
        body_producer = _MockProducerOp()
        body_consumer = _MockConsumerOp(body_producer.result)
        after.add_ops([body_producer, body_consumer, YieldOp(val)])

        while_op = WhileOp(
            [init_producer.result],
            [IntRegisterType.unallocated()],
            Region([before]),
            Region([after]),
        )
        sequence_op = SequenceOp(program=[init_producer, while_op], channel_id="test")

        allocator = _allocator()
        allocator.allocate_sequence(sequence_op)
        _assert_q1_registers_allocated(sequence_op)
        self.check_boundaries_have_same_allocation(while_op)

    def test_live_ins_used_in_before_region_are_allocated(self):
        """Tests that values defined outside the loop and used in the before region are
        allocated registers."""

        external_producer = _MockProducerOp()
        init_producer = _MockProducerOp()

        before = Block(arg_types=[IntRegisterType.unallocated()])
        (acc,) = before.args
        external_consumer = _MockConsumerOp(external_producer.result)
        before.add_ops([external_consumer, ConditionOp(UnaryPredicate.nez, [acc], [acc])])

        after = Block(arg_types=[IntRegisterType.unallocated()])
        (val,) = after.args
        after.add_op(YieldOp(val))

        while_op = WhileOp(
            [init_producer.result],
            [IntRegisterType.unallocated()],
            Region([before]),
            Region([after]),
        )
        sequence_op = SequenceOp(
            program=[external_producer, init_producer, while_op], channel_id="test"
        )

        _assert_q1_registers_unallocated(sequence_op)
        allocator = _allocator()
        allocator.allocate_sequence(sequence_op)

        _assert_q1_registers_allocated(sequence_op)
        self.check_boundaries_have_same_allocation(while_op)

    def test_live_ins_used_in_after_region_are_allocated(self):
        """Tests that values defined outside the loop and used in the after region are
        allocated registers."""

        external_producer = _MockProducerOp()
        init_producer = _MockProducerOp()

        before = Block(arg_types=[IntRegisterType.unallocated()])
        (acc,) = before.args
        before.add_op(ConditionOp(UnaryPredicate.nez, [acc], [acc]))

        after = Block(arg_types=[IntRegisterType.unallocated()])
        (val,) = after.args
        external_consumer = _MockConsumerOp(external_producer.result)
        after.add_ops([external_consumer, YieldOp(val)])

        while_op = WhileOp(
            [init_producer.result],
            [IntRegisterType.unallocated()],
            Region([before]),
            Region([after]),
        )
        sequence_op = SequenceOp(
            program=[external_producer, init_producer, while_op], channel_id="test"
        )

        _assert_q1_registers_unallocated(sequence_op)
        allocator = _allocator()
        allocator.allocate_sequence(sequence_op)

        _assert_q1_registers_allocated(sequence_op)
        self.check_boundaries_have_same_allocation(while_op)

    def test_two_sequential_while_ops_reuse_registers(self):
        """Tests that two sequential WhileOps reuse registers after the first loop frees
        them."""

        init_producer_1 = _MockProducerOp()
        before_1 = Block(arg_types=[IntRegisterType.unallocated()])
        (acc_1,) = before_1.args
        before_1.add_op(ConditionOp(UnaryPredicate.nez, [acc_1], [acc_1]))
        after_1 = Block(arg_types=[IntRegisterType.unallocated()])
        (val_1,) = after_1.args
        after_1.add_op(YieldOp(val_1))
        while_op_1 = WhileOp(
            [init_producer_1.result],
            [IntRegisterType.unallocated()],
            Region([before_1]),
            Region([after_1]),
        )

        init_producer_2 = _MockProducerOp()
        before_2 = Block(arg_types=[IntRegisterType.unallocated()])
        (acc_2,) = before_2.args
        before_2.add_op(ConditionOp(UnaryPredicate.nez, [acc_2], [acc_2]))
        after_2 = Block(arg_types=[IntRegisterType.unallocated()])
        (val_2,) = after_2.args
        after_2.add_op(YieldOp(val_2))
        while_op_2 = WhileOp(
            [init_producer_2.result],
            [IntRegisterType.unallocated()],
            Region([before_2]),
            Region([after_2]),
        )

        sequence_op = SequenceOp(
            program=[init_producer_1, while_op_1, init_producer_2, while_op_2],
            channel_id="test",
        )

        # Only one register available; both loops must share it sequentially.
        allocator = _allocator(_NUM_REGISTERS - 1)
        allocator.allocate_sequence(sequence_op)

        _assert_q1_registers_allocated(sequence_op)
        assert acc_1.type == acc_2.type
        self.check_boundaries_have_same_allocation(while_op_1)
        self.check_boundaries_have_same_allocation(while_op_2)


class TestRegAllocOnWhileOpDiagnosticErrors:
    """Tests that diagnostic errors are raised for WhileOps when register allocation is
    impossible."""

    def test_overallocation_in_before_region_raises_diagnostic_error(self):
        """Tests that exceeding the register budget inside the before region raises a
        diagnostic error."""

        init_producer = _MockProducerOp()

        before = Block(arg_types=[IntRegisterType.unallocated()])
        (acc,) = before.args
        body_producer_1 = _MockProducerOp()
        body_producer_2 = _MockProducerOp()
        body_consumer_1 = _MockConsumerOp(body_producer_1.result)
        body_consumer_2 = _MockConsumerOp(body_producer_2.result)
        before.add_ops(
            [
                body_producer_1,
                body_producer_2,
                body_consumer_1,
                body_consumer_2,
                ConditionOp(UnaryPredicate.nez, [acc], [acc]),
            ]
        )

        after = Block(arg_types=[IntRegisterType.unallocated()])
        (val,) = after.args
        after.add_op(YieldOp(val))

        while_op = WhileOp(
            [init_producer.result],
            [IntRegisterType.unallocated()],
            Region([before]),
            Region([after]),
        )
        sequence_op = SequenceOp(program=[init_producer, while_op], channel_id="test")

        allocator = _allocator(_NUM_REGISTERS - 2)
        with pytest.raises(DiagnosticException, match="Out of registers."):
            allocator.allocate_sequence(sequence_op)

    def test_overallocation_in_after_region_raises_diagnostic_error(self):
        """Tests that exceeding the register budget inside the after region raises a
        diagnostic error."""

        init_producer = _MockProducerOp()

        before = Block(arg_types=[IntRegisterType.unallocated()])
        (acc,) = before.args
        before.add_op(ConditionOp(UnaryPredicate.nez, [acc], [acc]))

        after = Block(arg_types=[IntRegisterType.unallocated()])
        (val,) = after.args
        body_producer_1 = _MockProducerOp()
        body_producer_2 = _MockProducerOp()
        body_consumer_1 = _MockConsumerOp(body_producer_1.result)
        body_consumer_2 = _MockConsumerOp(body_producer_2.result)
        after.add_ops(
            [
                body_producer_1,
                body_producer_2,
                body_consumer_1,
                body_consumer_2,
                YieldOp(val),
            ]
        )

        while_op = WhileOp(
            [init_producer.result],
            [IntRegisterType.unallocated()],
            Region([before]),
            Region([after]),
        )
        sequence_op = SequenceOp(program=[init_producer, while_op], channel_id="test")

        allocator = _allocator(_NUM_REGISTERS - 2)
        with pytest.raises(DiagnosticException, match="Out of registers."):
            allocator.allocate_sequence(sequence_op)

    def test_init_arg_used_after_loop_raises_diagnostic_error(self):
        """Tests that an init_arg consumed after the WhileOp raises a diagnostic error
        because it will have been allocated already when the allocator tries to coalesce it
        with the before block arg."""

        init_producer = _MockProducerOp()

        before = Block(arg_types=[IntRegisterType.unallocated()])
        (acc,) = before.args
        before.add_op(ConditionOp(UnaryPredicate.nez, [acc], [acc]))

        after = Block(arg_types=[IntRegisterType.unallocated()])
        (val,) = after.args
        after.add_op(YieldOp(val))

        while_op = WhileOp(
            [init_producer.result],
            [IntRegisterType.unallocated()],
            Region([before]),
            Region([after]),
        )
        post_consumer = _MockConsumerOp(init_producer.result)
        sequence_op = SequenceOp(
            program=[init_producer, while_op, post_consumer], channel_id="test"
        )

        allocator = _allocator()
        with pytest.raises(
            DiagnosticException,
            match="implies that the init_arg is used within the body of the loop or",
        ):
            allocator.allocate_sequence(sequence_op)

    def test_init_arg_used_in_before_region_as_live_in_raises_diagnostic_error(self):
        """Tests that an init_arg used as a live-in directly inside the before region raises
        a diagnostic error."""

        init_producer = _MockProducerOp()

        before = Block(arg_types=[IntRegisterType.unallocated()])
        (acc,) = before.args
        live_in_consumer = _MockConsumerOp(init_producer.result)
        before.add_ops([live_in_consumer, ConditionOp(UnaryPredicate.nez, [acc], [acc])])

        after = Block(arg_types=[IntRegisterType.unallocated()])
        (val,) = after.args
        after.add_op(YieldOp(val))

        while_op = WhileOp(
            [init_producer.result],
            [IntRegisterType.unallocated()],
            Region([before]),
            Region([after]),
        )
        sequence_op = SequenceOp(program=[init_producer, while_op], channel_id="test")

        allocator = _allocator()
        with pytest.raises(
            DiagnosticException,
            match="implies that the init_arg is used within the body of the loop or",
        ):
            allocator.allocate_sequence(sequence_op)

    def test_init_arg_used_in_after_region_as_live_in_raises_diagnostic_error(self):
        """Tests that an init_arg used as a live-in directly inside the after region raises
        a diagnostic error."""

        init_producer = _MockProducerOp()

        before = Block(arg_types=[IntRegisterType.unallocated()])
        (acc,) = before.args
        before.add_op(ConditionOp(UnaryPredicate.nez, [acc], [acc]))

        after = Block(arg_types=[IntRegisterType.unallocated()])
        (val,) = after.args
        live_in_consumer = _MockConsumerOp(init_producer.result)
        after.add_ops([live_in_consumer, YieldOp(val)])

        while_op = WhileOp(
            [init_producer.result],
            [IntRegisterType.unallocated()],
            Region([before]),
            Region([after]),
        )
        sequence_op = SequenceOp(program=[init_producer, while_op], channel_id="test")

        allocator = _allocator()
        with pytest.raises(
            DiagnosticException,
            match="implies that the init_arg is used within the body of the loop or",
        ):
            allocator.allocate_sequence(sequence_op)


class TestLinearScanRegisterAllocatorPass:
    """Tests the LinearScanRegisterAllocationPass running on a module with many sequence
    ops."""

    def test_with_sequence_that_uses_all_registers_allocates(self):
        """Tests that a sequence that uses all registers allocates without error."""

        sequence_ops = []
        for i in range(6):
            ops = _linear_producer_consumer_program(_NUM_REGISTERS - 1)

            sequence_op = SequenceOp(program=ops, channel_id=f"test_{i}")
            sequence_ops.append(sequence_op)

        module = ModuleOp(sequence_ops)

        LinearScanRegisterAllocationPass().apply(Context(), module)

        # Check that all registers are allocated
        _assert_q1_registers_allocated(module)

    def test_with_sequence_that_cannot_allocate_due_to_register_pressure(self):
        """Tests that a sequence that cannot allocate due to register pressure raises a
        diagnostic error.

        This does this for a single sequence op, and lets the others have few enough to not
        be a problem.
        """

        sequence_ops = []
        for i in range(6):
            if i == 5:
                ops = _linear_producer_consumer_program(_NUM_REGISTERS)
            else:
                ops = _linear_producer_consumer_program(_NUM_REGISTERS - 1)

            sequence_op = SequenceOp(program=ops, channel_id=f"test_{i}")
            sequence_ops.append(sequence_op)

        module = ModuleOp(sequence_ops)

        with pytest.raises(DiagnosticException, match="Out of registers."):
            LinearScanRegisterAllocationPass().apply(Context(), module)
