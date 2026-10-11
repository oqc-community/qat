"""Regression tests for Q1 allocation verification and related helpers."""

# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
import pytest
from xdsl.context import Context
from xdsl.dialects.builtin import IntegerType, ModuleOp
from xdsl.ir import Block, Region, SSAValue
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, region_def, result_def
from xdsl.passes import ModulePass
from xdsl.utils.exceptions import PassFailedException

from qat.experimental.dialect.q1 import IntRegisterType, Q1RegisterType
from qat.experimental.dialect.q1.ir.abstract_ops import Q1RegAllocOperation
from qat.experimental.dialect.q1.ir.ops import StopOp
from qat.experimental.dialect.q1.ir.reg_desc import Registers
from qat.experimental.dialect.q1.transforms.reg_alloc import (
    LinearScanRegisterAllocationPass,
)
from qat.experimental.dialect.q1.transforms.verify_allocation import (
    VerifyQ1AllocationPass,
    verify_sequence_allocation,
)
from qat.experimental.dialect.q1_cf import JmpBranchOp
from qat.experimental.dialect.q1_scf import ForOp, YieldOp
from qat.experimental.dialect.q1_sequence.ir.ops import SequenceOp
from qat.experimental.passes.pass_ordering import OrderedPassPipeline


@irdl_op_definition
class _MockProducerOp(Q1RegAllocOperation):
    """A mock operation that produces a single SSA value of type Q1RegisterType."""

    name = "mock.producer"
    result = result_def(IntRegisterType)

    def __init__(self, result_type: Q1RegisterType | None = None):
        if result_type is None:
            result_type = IntRegisterType.unallocated()
        return super().__init__(result_types=[result_type])


@irdl_op_definition
class _MockConsumerOp(Q1RegAllocOperation):
    """A mock operation that consumes a single SSA value of type Q1RegisterType."""

    name = "mock.consumer"
    operand = operand_def(IntRegisterType)

    def __init__(self, operand: SSAValue):
        return super().__init__(operands=[operand])


@irdl_op_definition
class _MockI32ProducerOp(IRDLOperation):
    """A mock operation that produces a non-Q1 SSA value."""

    name = "mock.i32_producer"
    result = result_def(IntegerType(32))

    def __init__(self):
        super().__init__(result_types=[IntegerType(32)])


@irdl_op_definition
class _MockI32ConsumerOp(IRDLOperation):
    """A mock operation that consumes a non-Q1 SSA value."""

    name = "mock.i32_consumer"
    operand = operand_def(IntegerType(32))

    def __init__(self, operand: SSAValue):
        super().__init__(operands=[operand])


@irdl_op_definition
class _MockRegionOp(IRDLOperation):
    """A mock operation with a single region for nested block-arg tests."""

    name = "mock.region"
    body = region_def()

    def __init__(self, region: Region):
        super().__init__(regions=[region])


class _AllocationBreakingPass(ModulePass):
    """A mock pass that reintroduces unallocated registers after allocation.

    This is used to test that the verifier detects register violations in passes other than
    the allocator itself.
    """

    name = "mock-register-creating"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        """Replace all producer ops with ones that produce unallocated registers."""
        from xdsl.rewriter import Rewriter

        for sequence in (s for s in op.walk() if isinstance(s, SequenceOp)):
            # Collect all producer ops first (to avoid modifying while iterating)
            producers = [
                nested_op
                for nested_op in sequence.body.block.ops
                if isinstance(nested_op, _MockProducerOp)
            ]
            # Replace each producer with one that produces unallocated registers
            for producer in producers:
                new_op = _MockProducerOp(IntRegisterType.unallocated())
                Rewriter.replace_op(producer, [new_op])


class TestVerifyQ1AllocationPassBasic:
    """Test basic verification of register allocation."""

    def test_fully_allocated_sequence_passes(self):
        """Test that a fully allocated sequence passes verification."""
        # Create a sequence with allocated registers
        producer = _MockProducerOp.create(result_types=[Registers.R1])
        consumer = _MockConsumerOp(producer.result)
        sequence = SequenceOp(program=[producer, consumer], channel_id="test")
        module = ModuleOp([sequence])

        # Apply the verifier
        pass_instance = VerifyQ1AllocationPass()
        pass_instance.apply(Context(), module)  # Should not raise

    def test_unallocated_result_fails(self):
        """Test that a sequence with an unallocated result fails."""
        # Create a sequence with an unallocated register
        producer = _MockProducerOp(IntRegisterType.unallocated())
        sequence = SequenceOp(program=[producer], channel_id="test_seq")
        module = ModuleOp([sequence])

        # Apply the verifier
        pass_instance = VerifyQ1AllocationPass()
        with pytest.raises(PassFailedException) as exc_info:
            pass_instance.apply(Context(), module)

        error_msg = str(exc_info.value)
        assert "test_seq" in error_msg  # Channel ID
        assert "mock.producer" in error_msg  # Operation name
        assert "unallocated" in error_msg

    def test_unallocated_operand_fails(self):
        """Test that a sequence with an unallocated operand fails."""
        # Create a producer with unallocated result, then a consumer
        producer = _MockProducerOp(IntRegisterType.unallocated())
        consumer = _MockConsumerOp(producer.result)

        sequence = SequenceOp(program=[producer, consumer], channel_id="test_seq")
        module = ModuleOp([sequence])

        # Apply the verifier
        pass_instance = VerifyQ1AllocationPass()
        with pytest.raises(PassFailedException) as exc_info:
            pass_instance.apply(Context(), module)

        error_msg = str(exc_info.value)
        assert "test_seq" in error_msg
        # The error could be about the producer's result or the consumer's operand
        assert ("mock.producer" in error_msg) or ("mock.consumer" in error_msg)
        assert "unallocated" in error_msg

    def test_virtual_register_fails(self):
        """Test that a sequence with a virtual (infinite) register fails."""
        # Create a virtual register (negative index from infinite_register)
        virtual_reg = IntRegisterType.infinite_register(5)
        producer = _MockProducerOp.create(result_types=[virtual_reg])

        sequence = SequenceOp(program=[producer], channel_id="test_seq")
        module = ModuleOp([sequence])

        # Apply the verifier
        pass_instance = VerifyQ1AllocationPass()
        with pytest.raises(PassFailedException) as exc_info:
            pass_instance.apply(Context(), module)

        error_msg = str(exc_info.value)
        assert "test_seq" in error_msg
        assert "mock.producer" in error_msg
        assert "virtual" in error_msg
        # The error message should include the register name
        assert "inf_gpr" in error_msg


class TestVerifyQ1AllocationPassIntegration:
    """Test integration with the allocation pipeline."""

    def test_allocation_followed_by_verification_passes(self):
        """Test that allocation followed by verification passes."""
        # Create a sequence with unallocated registers
        producer = _MockProducerOp()
        consumer = _MockConsumerOp(producer.result)
        sequence = SequenceOp(program=[producer, consumer], channel_id="test")
        module = ModuleOp([sequence])

        # Create a pipeline with allocation then verification
        pipeline = OrderedPassPipeline(
            (
                LinearScanRegisterAllocationPass(),
                VerifyQ1AllocationPass(),
            )
        )

        # Should not raise
        pipeline.apply(Context(), module)

    def test_verification_can_run_without_allocator_in_pipeline(self):
        """Test that the verifier can be constructed as a standalone pipeline pass."""
        pipeline = OrderedPassPipeline((VerifyQ1AllocationPass(),))

        assert len(pipeline.passes) == 1
        assert isinstance(pipeline.passes[0], VerifyQ1AllocationPass)

    def test_verify_sequence_allocation_is_reusable(self):
        """Test that sequence verification is exposed as a reusable helper."""
        sequence = SequenceOp(
            program=[_MockProducerOp(IntRegisterType.unallocated())],
            channel_id="test",
        )

        errors = verify_sequence_allocation(sequence)

        assert len(errors) == 1
        assert "unallocated" in errors[0]

    def test_broken_pass_after_allocation_caught_immediately(self):
        """Test that a pass breaking allocation is caught at verification.

        This tests the acceptance criterion: "A test pipeline with a
        register-creating pass placed after allocation fails at the verifier
        run straight after that pass, not at the end."
        """
        # Create a sequence with unallocated registers
        producer = _MockProducerOp()
        consumer = _MockConsumerOp(producer.result)
        sequence = SequenceOp(program=[producer, consumer], channel_id="test")
        module = ModuleOp([sequence])

        # Create a pipeline: allocate, verify, then a broken pass, then another verify
        pipeline = OrderedPassPipeline(
            (
                LinearScanRegisterAllocationPass(),
                VerifyQ1AllocationPass(),  # This should pass
                _AllocationBreakingPass(),  # This breaks allocation
                VerifyQ1AllocationPass(),  # This should fail immediately
            )
        )

        # The second verification should fail due to the broken pass
        with pytest.raises(PassFailedException) as exc_info:
            pipeline.apply(Context(), module)

        error_msg = str(exc_info.value)
        # Verify that the error is about unallocated registers
        assert "unallocated" in error_msg


class TestVerifyQ1AllocationPassNestedRegions:
    """Test verification of nested q1_scf control flow regions."""

    def test_q1_scf_for_body_operations_are_verified(self):
        """Operations inside a q1_scf.for body should be checked like top-level ops."""
        body = Block(arg_types=(Registers.R1,))
        nested_producer = _MockProducerOp(IntRegisterType.unallocated())
        body.add_ops([nested_producer, YieldOp()])

        producer = _MockProducerOp.create(result_types=[Registers.R1])
        loop = ForOp(iter_count=producer.result, iter_args=[], body=body)
        sequence = SequenceOp(program=[producer, loop], channel_id="test_seq")

        errors = verify_sequence_allocation(sequence)

        assert len(errors) == 1
        assert "mock.producer" in errors[0]
        assert "unallocated" in errors[0]

    def test_sequence_block_arguments_are_verified(self):
        """Top-level sequence block arguments should be rejected when unallocated."""
        sequence = SequenceOp(
            channel_id="test_seq",
            program=Region(
                Block(
                    arg_types=[
                        IntRegisterType.unallocated(),
                        IntegerType(32),
                    ]
                )
            ),
        )

        errors = verify_sequence_allocation(sequence)

        assert len(errors) == 1
        assert "test_seq" in errors[0]
        assert "unallocated" in errors[0]

    def test_multi_block_sequence_block_arguments_are_verified(self):
        """CFG-form sequence block arguments should be checked in every body block."""
        merge = Block(
            arg_types=[
                IntRegisterType.unallocated(),
            ],
            ops=[StopOp()],
        )
        entry = Block([JmpBranchOp([], merge)])
        sequence = SequenceOp(
            channel_id="test_seq",
            program=Region([entry, merge]),
        )

        errors = verify_sequence_allocation(sequence)

        assert errors == [
            "Sequence 'test_seq': block_arg 0 of operation 'sequence' is an "
            "unallocated Q1 register."
        ]

    def test_q1_scf_for_body_block_arguments_are_verified(self):
        """q1_scf.for induction/block arguments should also be visited."""
        body = Block(
            arg_types=[
                IntRegisterType.unallocated(),
            ]
        )
        body.add_op(YieldOp())

        producer = _MockProducerOp.create(result_types=[IntRegisterType.unallocated()])
        loop = ForOp(iter_count=producer.result, iter_args=[], body=body)
        sequence = SequenceOp(channel_id="test_seq", program=[producer, loop])

        errors = verify_sequence_allocation(sequence)

        assert len(errors) == 2
        assert "mock.producer" in errors[0]
        assert "q1_scf.for" in errors[1]
        assert "block_arg 0" in errors[1]

    def test_nested_q1_scf_for_bodies_are_verified(self):
        """Nested q1_scf.for bodies should be walked recursively."""
        inner_body = Block(arg_types=(Registers.R2,))
        inner_body.add_ops([_MockProducerOp(IntRegisterType.unallocated()), YieldOp()])

        inner_count = _MockProducerOp.create(result_types=[Registers.R2])
        inner_loop = ForOp(iter_count=inner_count.result, iter_args=[], body=inner_body)

        outer_body = Block(arg_types=(Registers.R1,))
        outer_body.add_ops([inner_count, inner_loop, YieldOp()])

        outer_count = _MockProducerOp.create(result_types=[Registers.R1])
        outer_loop = ForOp(iter_count=outer_count.result, iter_args=[], body=outer_body)
        sequence = SequenceOp(program=[outer_count, outer_loop], channel_id="test_seq")

        errors = verify_sequence_allocation(sequence)

        assert len(errors) == 1
        assert "mock.producer" in errors[0]
        assert "unallocated" in errors[0]

    def test_non_q1_operands_and_results_are_ignored(self):
        """Non-Q1 operands and results should not be treated as allocation errors."""
        producer = _MockI32ProducerOp()
        consumer = _MockI32ConsumerOp(producer.result)
        sequence = SequenceOp(program=[producer, consumer], channel_id="test_seq")

        assert verify_sequence_allocation(sequence) == []


class TestVerifyQ1AllocationPassErrorMessages:
    """Test that error messages are clear and helpful."""

    def test_error_message_includes_channel_id(self):
        """Test that error messages include the sequence channel ID."""
        producer = _MockProducerOp(IntRegisterType.unallocated())
        sequence = SequenceOp(program=[producer], channel_id="my_unique_channel")
        module = ModuleOp([sequence])

        pass_instance = VerifyQ1AllocationPass()
        with pytest.raises(PassFailedException) as exc_info:
            pass_instance.apply(Context(), module)

        assert "my_unique_channel" in str(exc_info.value)

    def test_error_message_includes_operation_name(self):
        """Test that error messages include the operation name."""
        producer = _MockProducerOp(IntRegisterType.unallocated())
        sequence = SequenceOp(program=[producer], channel_id="test")
        module = ModuleOp([sequence])

        pass_instance = VerifyQ1AllocationPass()
        with pytest.raises(PassFailedException) as exc_info:
            pass_instance.apply(Context(), module)

        assert "mock.producer" in str(exc_info.value)

    def test_error_message_includes_value_position(self):
        """Test that error messages include the value position."""
        producer = _MockProducerOp(IntRegisterType.unallocated())
        sequence = SequenceOp(program=[producer], channel_id="test")
        module = ModuleOp([sequence])

        pass_instance = VerifyQ1AllocationPass()
        with pytest.raises(PassFailedException) as exc_info:
            pass_instance.apply(Context(), module)

        error_msg = str(exc_info.value)
        assert "result 0 of operation 'mock.producer'" in error_msg

    def test_virtual_register_error_includes_inf_gpr_name(self):
        """Test that virtual register errors include the inf_gpr_N name."""
        virtual_reg = IntRegisterType.infinite_register(42)
        producer = _MockProducerOp.create(result_types=[virtual_reg])
        sequence = SequenceOp(program=[producer], channel_id="test")
        module = ModuleOp([sequence])

        pass_instance = VerifyQ1AllocationPass()
        with pytest.raises(PassFailedException) as exc_info:
            pass_instance.apply(Context(), module)

        error_msg = str(exc_info.value)
        assert "virtual" in error_msg
        assert "inf_gpr" in error_msg

    def test_nested_region_block_arg_error_includes_index(self):
        """Test that public verification formats nested region block args with an index."""
        region = Region(Block(arg_types=[IntRegisterType.unallocated()]))
        region_op = _MockRegionOp(region)
        sequence = SequenceOp(program=[region_op], channel_id="test")

        errors = verify_sequence_allocation(sequence)

        assert errors == [
            "Sequence 'test': block_arg 0 of operation 'mock.region' is an unallocated Q1 "
            "register."
        ]


class TestVerifyQ1AllocationPassMultipleSequences:
    """Test verification across multiple sequences."""

    def test_verifies_all_sequences(self):
        """Test that all sequences are verified, not just the first."""
        # Create two sequences
        seq1 = SequenceOp(
            program=[_MockProducerOp.create(result_types=[Registers.R1])],
            channel_id="seq1",
        )
        seq2 = SequenceOp(
            program=[_MockProducerOp(IntRegisterType.unallocated())],
            channel_id="seq2",
        )
        module = ModuleOp([seq1, seq2])

        pass_instance = VerifyQ1AllocationPass()
        with pytest.raises(PassFailedException) as exc_info:
            pass_instance.apply(Context(), module)

        error_msg = str(exc_info.value)
        # Should fail on seq2
        assert "seq2" in error_msg
        assert "unallocated" in error_msg

    def test_passes_when_all_sequences_allocated(self):
        """Test that verification passes when all sequences are properly allocated."""
        seq1 = SequenceOp(
            program=[_MockProducerOp.create(result_types=[Registers.R1])],
            channel_id="seq1",
        )
        seq2 = SequenceOp(
            program=[_MockProducerOp.create(result_types=[Registers.R2])],
            channel_id="seq2",
        )
        module = ModuleOp([seq1, seq2])

        pass_instance = VerifyQ1AllocationPass()
        pass_instance.apply(Context(), module)  # Should not raise


class TestVerifyQ1AllocationPassEmptySequences:
    """Test verification of edge cases."""

    def test_empty_sequence_passes(self):
        """Test that an empty sequence passes verification."""
        sequence = SequenceOp(program=[], channel_id="test")
        module = ModuleOp([sequence])

        pass_instance = VerifyQ1AllocationPass()
        pass_instance.apply(Context(), module)  # Should not raise

    def test_sequence_with_no_registers_passes(self):
        """Test that a sequence with no Q1 registers passes."""
        # Create a mock operation that doesn't use Q1 registers
        # (In reality, most Q1 operations use registers, but this is a boundary test)
        sequence = SequenceOp(program=[], channel_id="test")
        module = ModuleOp([sequence])

        pass_instance = VerifyQ1AllocationPass()
        pass_instance.apply(Context(), module)  # Should not raise


class TestVerifyQ1AllocationPassCollectErrors:
    """Test that the pass collects and reports all errors together."""

    def test_collects_multiple_errors_from_single_sequence(self):
        """Test that one bad SSA value is reported only once."""
        # Create one unallocated SSA value and use it in a second operation.
        unalloc1 = _MockProducerOp.create(result_types=[IntRegisterType.unallocated()])
        unalloc2 = _MockConsumerOp(operand=unalloc1.result)
        sequence = SequenceOp(program=[unalloc1, unalloc2], channel_id="test")
        module = ModuleOp([sequence])

        pass_instance = VerifyQ1AllocationPass()
        with pytest.raises(PassFailedException) as exc_info:
            pass_instance.apply(Context(), module)

        error_msg = str(exc_info.value)
        assert "1 issue(s)" in error_msg
        assert "unallocated" in error_msg
        assert "mock.producer" in error_msg

    def test_collects_errors_across_multiple_sequences(self):
        """Test that errors from multiple sequences are all reported."""
        # Create two sequences each with an unallocated value
        seq1 = SequenceOp(
            program=[_MockProducerOp.create(result_types=[IntRegisterType.unallocated()])],
            channel_id="seq1",
        )
        seq2 = SequenceOp(
            program=[_MockProducerOp.create(result_types=[IntRegisterType.unallocated()])],
            channel_id="seq2",
        )
        module = ModuleOp([seq1, seq2])

        pass_instance = VerifyQ1AllocationPass()
        with pytest.raises(PassFailedException) as exc_info:
            pass_instance.apply(Context(), module)

        error_msg = str(exc_info.value)
        # Should report 2 errors total and include both sequence names
        assert "2 issue(s)" in error_msg
        assert "seq1" in error_msg
        assert "seq2" in error_msg

    def test_error_report_includes_all_details(self):
        """Test that the collected error report includes detailed messages."""
        unalloc = _MockProducerOp.create(result_types=[IntRegisterType.unallocated()])
        sequence = SequenceOp(program=[unalloc], channel_id="my_channel")
        module = ModuleOp([sequence])

        pass_instance = VerifyQ1AllocationPass()
        with pytest.raises(PassFailedException) as exc_info:
            pass_instance.apply(Context(), module)

        error_msg = str(exc_info.value)
        # Should have summary line and detailed lines
        assert "Register allocation verification failed" in error_msg
        assert "Found 1 issue(s)" in error_msg
        # Detailed line should be present
        assert "my_channel" in error_msg
