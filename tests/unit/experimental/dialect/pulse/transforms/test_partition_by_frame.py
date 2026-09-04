# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest
from xdsl.context import Context
from xdsl.dialects import func
from xdsl.dialects.arith import ConstantOp as ArithConstantOp
from xdsl.dialects.builtin import IndexType, ModuleOp, StringAttr
from xdsl.interpreters.scf import scf
from xdsl.ir import Block, Region
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, region_def, result_def
from xdsl.utils.exceptions import PassFailedException

from qat.experimental.dialect.pulse.ir import (
    AcquireOp,
    AmplitudeAttr,
    ConstantOp,
    CreateFrameOp,
    FrameType,
    FrequencyAttr,
    PulseOp,
    SquareWaveformOp,
    SynchronizeOp,
    TimeAttr,
    WaitOp,
)
from qat.experimental.dialect.pulse.transforms.partition_by_frame import (
    FrameLineage,
    FrameLineagePass,
    FrameNode,
    build_frame_lineage_analysis,
)


def _module_with_main(ops):
    return ModuleOp([func.FuncOp("main", ((), ()), Region(Block(ops)))])


@irdl_op_definition
class _FrameSideEffectOp(IRDLOperation):
    name = "test.frame_side_effect"
    frame = operand_def(FrameType)

    def __init__(self, frame):
        super().__init__(operands=[frame])


@irdl_op_definition
class _AmbiguousMergeOp(IRDLOperation):
    name = "test.ambiguous_merge"
    lhs = operand_def(FrameType)
    rhs = operand_def(FrameType)
    result = result_def(FrameType)

    def __init__(self, lhs, rhs):
        lhs_type = lhs.result.type if isinstance(lhs, CreateFrameOp) else lhs.type
        super().__init__(operands=[lhs, rhs], result_types=[lhs_type])


@irdl_op_definition
class _RegionBearingOp(IRDLOperation):
    name = "test.region_bearing"
    body = region_def()

    def __init__(self, body: Region):
        super().__init__(regions=[body])


class TestFrameNode:
    def test_single_node_chain_contains_only_root(self):
        """Verify that a root node's chain yields exactly itself."""
        op = ConstantOp(FrequencyAttr(4.8e9))
        node = FrameNode(op=op, parent=None)
        assert list(node.chain()) == [node]

    def test_chain_walks_to_root_in_order(self):
        """Verify that chain() yields parent nodes in leaf-to-root order."""
        op0 = ConstantOp(FrequencyAttr(4.8e9))
        op1 = ConstantOp(FrequencyAttr(5.0e9))
        op2 = ConstantOp(FrequencyAttr(5.2e9))
        root = FrameNode(op=op0, parent=None)
        mid = FrameNode(op=op1, parent=root)
        leaf = FrameNode(op=op2, parent=mid)
        assert list(leaf.chain()) == [leaf, mid, root]

    def test_root_property_returns_ancestor_without_parent(self):
        """Verify that root property returns the top of the chain."""
        op0 = ConstantOp(FrequencyAttr(4.8e9))
        op1 = ConstantOp(FrequencyAttr(5.0e9))
        root = FrameNode(op=op0, parent=None)
        child = FrameNode(op=op1, parent=root)
        assert child.root is root
        assert root.root is root


class TestFrameLineage:
    def test_root_node_is_none_when_related_ops_is_empty(self):
        """Verify that root_node returns None rather than raising when related_ops has not
        been seeded yet."""
        freq = ConstantOp(FrequencyAttr(4.8e9))
        frame = CreateFrameOp(freq, StringAttr("q0/drive"))
        lineage = FrameLineage(create_frame=frame, port="q0/drive")

        assert lineage.related_ops == []
        assert lineage.root_node is None

    def test_add_node_on_empty_related_ops_seeds_a_root(self):
        """Verify that add_node creates a parentless root node when related_ops starts
        empty, rather than indexing into an empty list."""
        freq = ConstantOp(FrequencyAttr(4.8e9))
        frame = CreateFrameOp(freq, StringAttr("q0/drive"))
        lineage = FrameLineage(create_frame=frame, port="q0/drive")

        lineage.add_node(frame)

        assert [n.op for n in lineage.related_ops] == [frame]
        assert lineage.related_ops[0].parent is None
        assert lineage.root_node is lineage.related_ops[0]


class TestFrameLineageAnalysis:
    def test_partitions_by_frame_identity_not_port(self):
        """Verify that logical frame identity, not port identity, defines partitions."""
        freq_0 = ConstantOp(FrequencyAttr(4.8e9))
        freq_1 = ConstantOp(FrequencyAttr(5.2e9))
        frame_0 = CreateFrameOp(freq_0, StringAttr("q0/drive"))
        frame_1 = CreateFrameOp(freq_1, StringAttr("q0/drive"))

        time = ConstantOp(TimeAttr(16e-9))
        wait_0 = WaitOp(frame_0, time)
        wait_1 = WaitOp(frame_1, time)
        stop = func.ReturnOp()
        analysis = build_frame_lineage_analysis(
            _module_with_main(
                [freq_0, freq_1, frame_0, frame_1, time, wait_0, wait_1, stop]
            )
        )

        assert [lin.frame for lin in analysis.lineages] == [frame_0.result, frame_1.result]
        assert analysis.lineage_for_frame(frame_0.result).port == "q0/drive"
        assert analysis.lineage_for_frame(frame_1.result).port == "q0/drive"
        assert analysis.port_counts == {"q0/drive": 2}
        assert [n.op for n in analysis.lineage_for_frame(frame_0.result).related_ops] == [
            frame_0,
            wait_0,
        ]
        assert [n.op for n in analysis.lineage_for_frame(frame_1.result).related_ops] == [
            frame_1,
            wait_1,
        ]

    def test_sync_is_attached_to_each_participating_frame(self):
        """Verify that a synchronize operation is recorded in each participating lineage."""
        freq_0 = ConstantOp(FrequencyAttr(4.8e9))
        freq_1 = ConstantOp(FrequencyAttr(6.0e9))
        frame_0 = CreateFrameOp(freq_0, StringAttr("q0/drive"))
        frame_1 = CreateFrameOp(freq_1, StringAttr("q1/drive"))
        sync = SynchronizeOp(frame_0, frame_1)
        duration = ConstantOp(TimeAttr(24e-9))
        wait_0 = WaitOp(sync.results[0], duration)
        wait_1 = WaitOp(sync.results[1], duration)
        analysis = build_frame_lineage_analysis(
            _module_with_main(
                [
                    freq_0,
                    freq_1,
                    frame_0,
                    frame_1,
                    sync,
                    duration,
                    wait_0,
                    wait_1,
                    func.ReturnOp(),
                ]
            )
        )

        assert [n.op for n in analysis.lineage_for_frame(frame_0.result).related_ops] == [
            frame_0,
            sync,
            wait_0,
        ]
        assert [n.op for n in analysis.lineage_for_frame(frame_1.result).related_ops] == [
            frame_1,
            sync,
            wait_1,
        ]

    def test_acquire_remains_in_same_partition(self):
        """Verify that acquisition and subsequent pulse use remain in one lineage."""
        freq = ConstantOp(FrequencyAttr(5.1e9))
        frame = CreateFrameOp(freq, StringAttr("q0/measure"))
        duration = ConstantOp(TimeAttr(1e-6))
        acquire = AcquireOp(frame, duration)
        amp = ConstantOp(AmplitudeAttr(0.3))
        width = ConstantOp(TimeAttr(40e-9))
        waveform = SquareWaveformOp(width, amp)
        pulse = PulseOp(acquire.frame_result, waveform)
        analysis = build_frame_lineage_analysis(
            _module_with_main(
                [
                    freq,
                    frame,
                    duration,
                    acquire,
                    amp,
                    width,
                    waveform,
                    pulse,
                    func.ReturnOp(),
                ]
            )
        )

        [lin] = analysis.lineages
        assert [n.op for n in lin.related_ops] == [frame, acquire, pulse]
        assert lin.port == "q0/measure"

    def test_raises_on_unbound_frame_operand(self):
        """Verify that analysis fails when a frame operand precedes its defining frame."""
        freq = ConstantOp(FrequencyAttr(5.1e9))
        frame = CreateFrameOp(freq, StringAttr("q0/drive"))
        duration = ConstantOp(TimeAttr(16e-9))
        wait = WaitOp(frame, duration)
        module = _module_with_main([duration, wait, freq, frame, func.ReturnOp()])

        with pytest.raises(PassFailedException, match="Unbound frame operand"):
            build_frame_lineage_analysis(module)

    def test_nested_unbound_frame_operand_is_silently_skipped(self):
        """Verify that an unresolvable frame operand inside a nested region is silently
        ignored, unlike the entry-level case which raises."""
        block = Block(arg_types=[FrameType("q0/drive")])
        block.add_op(_FrameSideEffectOp(block.args[0]))
        container = _RegionBearingOp(Region(block))
        module = _module_with_main([container, func.ReturnOp()])

        analysis = build_frame_lineage_analysis(module)
        assert analysis.lineages == []

    def test_nested_create_frame_op_is_attributed_to_enclosing_container(self):
        """Verify that a CreateFrameOp created inside a nested region starts a lineage whose
        first related op is the enclosing container, not the CreateFrameOp itself."""
        freq = ConstantOp(FrequencyAttr(4.8e9))
        frame_op = CreateFrameOp(freq, StringAttr("q0/drive"))
        container = _RegionBearingOp(Region(Block([freq, frame_op])))
        module = _module_with_main([container, func.ReturnOp()])

        analysis = build_frame_lineage_analysis(module)

        [lin] = analysis.lineages
        assert lin.port == "q0/drive"
        assert lin.frame is frame_op.result
        assert [n.op for n in lin.related_ops] == [container]
        assert lin.root_node is not None
        assert lin.root_node.op is frame_op

    def test_lineage_for_result_resolves_any_owned_value(self):
        """Verify that lineage_for_result resolves any value claimed by a lineage, not just
        the root frame value."""
        freq = ConstantOp(FrequencyAttr(4.8e9))
        frame = CreateFrameOp(freq, StringAttr("q0/drive"))
        duration = ConstantOp(TimeAttr(16e-9))
        wait = WaitOp(frame, duration)
        analysis = build_frame_lineage_analysis(
            _module_with_main([freq, frame, duration, wait, func.ReturnOp()])
        )

        lin = analysis.lineage_for_frame(frame.result)
        assert analysis.lineage_for_result(frame.result) is lin
        assert analysis.lineage_for_result(wait.result) is lin
        assert analysis.lineage_for_result(duration.result) is None

    def test_lineage_for_frame_returns_none_for_non_root_owned_value(self):
        """Verify that lineage_for_frame only matches the root create_frame value, even
        though a derived value is owned by the same lineage."""
        freq = ConstantOp(FrequencyAttr(4.8e9))
        frame = CreateFrameOp(freq, StringAttr("q0/drive"))
        duration = ConstantOp(TimeAttr(16e-9))
        wait = WaitOp(frame, duration)
        analysis = build_frame_lineage_analysis(
            _module_with_main([freq, frame, duration, wait, func.ReturnOp()])
        )

        assert analysis.lineage_for_frame(wait.result) is None

    def test_region_bearing_op_without_frame_ref_is_excluded(self):
        """Verify that a region-bearing entry op with no frame references is ignored."""
        zero = ArithConstantOp.from_int_and_width(0, IndexType())
        one = ArithConstantOp.from_int_and_width(1, IndexType())
        ten = ArithConstantOp.from_int_and_width(10, IndexType())
        loop = scf.ForOp(zero, ten, one, [], Block(arg_types=[IndexType()]))
        module = _module_with_main([zero, ten, one, loop, func.ReturnOp()])

        analysis = build_frame_lineage_analysis(module)
        assert analysis.lineages == []

    def test_region_bearing_op_outside_lineage_is_not_attributed(self):
        """Verify that a region-bearing op is only attributed when the frame is referenced
        inside its own body.

        A region-bearing op sitting alongside an unrelated frame lineage, with no reference
        to that frame anywhere in its body, must not be added to that lineage's related ops.
        """
        freq = ConstantOp(FrequencyAttr(4.8e9))
        frame_op = CreateFrameOp(freq, StringAttr("q0/drive"))
        zero = ArithConstantOp.from_int_and_width(0, IndexType())
        one = ArithConstantOp.from_int_and_width(1, IndexType())
        ten = ArithConstantOp.from_int_and_width(10, IndexType())
        loop = scf.ForOp(zero, ten, one, [], Block(arg_types=[IndexType()]))
        module = _module_with_main([freq, frame_op, zero, ten, one, loop, func.ReturnOp()])

        analysis = build_frame_lineage_analysis(module)

        lin = analysis.lineage_for_frame(frame_op.result)
        assert lin is not None
        assert [n.op for n in lin.related_ops] == [frame_op]

    def test_region_bearing_op_with_frame_ref_is_attributed_to_lineage(self):
        """Verify that a region-bearing entry op referencing a frame is added to its
        lineage."""
        freq = ConstantOp(FrequencyAttr(4.8e9))
        frame_op = CreateFrameOp(freq, StringAttr("q0/drive"))
        side_effect = _FrameSideEffectOp(frame_op)
        container = _RegionBearingOp(Region(Block([side_effect])))
        analysis = build_frame_lineage_analysis(
            _module_with_main([freq, frame_op, container, func.ReturnOp()])
        )

        lin = analysis.lineage_for_frame(frame_op.result)
        assert lin is not None
        assert [n.op for n in lin.related_ops] == [frame_op, container]

    def test_region_bearing_op_shared_by_two_frames_attributed_to_both(self):
        """Verify that a region-bearing op referencing multiple frames appears in each
        lineage."""
        freq_0 = ConstantOp(FrequencyAttr(4.8e9))
        freq_1 = ConstantOp(FrequencyAttr(5.2e9))
        frame_0 = CreateFrameOp(freq_0, StringAttr("q0/drive"))
        frame_1 = CreateFrameOp(freq_1, StringAttr("q1/drive"))
        body = Region(Block([_FrameSideEffectOp(frame_0), _FrameSideEffectOp(frame_1)]))
        container = _RegionBearingOp(body)
        analysis = build_frame_lineage_analysis(
            _module_with_main(
                [freq_0, freq_1, frame_0, frame_1, container, func.ReturnOp()]
            )
        )

        lin_0 = analysis.lineage_for_frame(frame_0.result)
        lin_1 = analysis.lineage_for_frame(frame_1.result)
        assert lin_0 is not None and [n.op for n in lin_0.related_ops] == [
            frame_0,
            container,
        ]
        assert lin_1 is not None and [n.op for n in lin_1.related_ops] == [
            frame_1,
            container,
        ]

    def test_frame_operand_without_frame_result_keeps_partition(self):
        """Verify that frame-consuming side effects remain attached to their lineage."""
        freq = ConstantOp(FrequencyAttr(5.1e9))
        frame = CreateFrameOp(freq, StringAttr("q0/drive"))
        side_effect = _FrameSideEffectOp(frame)
        analysis = build_frame_lineage_analysis(
            _module_with_main([freq, frame, side_effect, func.ReturnOp()])
        )
        [lin] = analysis.lineages
        assert [n.op for n in lin.related_ops] == [frame, side_effect]

    def test_raises_when_multiple_entry_functions_exist(self):
        """Verify that analysis requires at most one entry function."""
        main_a = func.FuncOp("main_a", ((), ()), Region(Block([func.ReturnOp()])))
        main_b = func.FuncOp("main_b", ((), ()), Region(Block([func.ReturnOp()])))
        module = ModuleOp([main_a, main_b])
        with pytest.raises(PassFailedException, match="single entry function"):
            build_frame_lineage_analysis(module)

    def test_raises_when_module_mixes_function_and_top_level_ops(self):
        """Verify that analysis rejects modules mixing entry shapes."""
        freq = ConstantOp(FrequencyAttr(4.8e9))
        frame = CreateFrameOp(freq, StringAttr("q0/drive"))
        main = func.FuncOp("main", ((), ()), Region(Block([func.ReturnOp()])))
        module = ModuleOp([main, freq, frame])
        with pytest.raises(PassFailedException, match="either a flat module"):
            build_frame_lineage_analysis(module)

    def test_raises_for_ambiguous_multi_frame_result_mapping(self):
        """Verify that ambiguous multi-frame result ownership is rejected."""
        freq_0 = ConstantOp(FrequencyAttr(4.8e9))
        freq_1 = ConstantOp(FrequencyAttr(5.2e9))
        frame_0 = CreateFrameOp(freq_0, StringAttr("q0/drive"))
        frame_1 = CreateFrameOp(freq_1, StringAttr("q1/drive"))
        ambiguous = _AmbiguousMergeOp(frame_0, frame_1)
        module = _module_with_main(
            [freq_0, freq_1, frame_0, frame_1, ambiguous, func.ReturnOp()]
        )
        with pytest.raises(PassFailedException, match="Cannot map frame results"):
            build_frame_lineage_analysis(module)

    def test_sync_with_duplicate_frame_operand_is_recorded_once(self):
        """Verify that duplicate operands in one sync do not duplicate lineage entries."""
        freq = ConstantOp(FrequencyAttr(4.8e9))
        frame = CreateFrameOp(freq, StringAttr("q0/drive"))
        sync = SynchronizeOp(frame, frame)
        analysis = build_frame_lineage_analysis(
            _module_with_main([freq, frame, sync, func.ReturnOp()])
        )
        [lin] = analysis.lineages
        assert [n.op for n in lin.related_ops] == [frame, sync]

    def test_compute_pass_apply_sets_analysis(self):
        """Verify that the compute pass stores the analysis it produces."""
        freq = ConstantOp(FrequencyAttr(4.8e9))
        frame = CreateFrameOp(freq, StringAttr("q0/drive"))
        module = _module_with_main([freq, frame, func.ReturnOp()])
        pass_instance = FrameLineagePass()
        pass_instance.apply(Context(), module)
        assert pass_instance.analysis is not None
