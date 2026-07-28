# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest
from xdsl.context import Context
from xdsl.dialects import func
from xdsl.dialects.arith import ConstantOp as ArithConstantOp
from xdsl.dialects.builtin import IndexType, ModuleOp, StringAttr
from xdsl.interpreters.scf import scf
from xdsl.ir import Block, Region
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, result_def
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
    FrameLineagePass,
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

        assert list(analysis.frame_to_operations) == [frame_0.result, frame_1.result]
        assert analysis.frame_to_port == {
            frame_0.result: "q0/drive",
            frame_1.result: "q0/drive",
        }
        assert analysis.port_to_frames == {"q0/drive": (frame_0.result, frame_1.result)}
        assert analysis.frame_to_operations[frame_0.result] == (frame_0, wait_0)
        assert analysis.frame_to_operations[frame_1.result] == (frame_1, wait_1)

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

        assert analysis.frame_to_operations[frame_0.result] == (frame_0, sync, wait_0)
        assert analysis.frame_to_operations[frame_1.result] == (frame_1, sync, wait_1)

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

        assert analysis.frame_to_operations[frame.result] == (frame, acquire, pulse)
        assert analysis.frame_to_port[frame.result] == "q0/measure"

    def test_raises_on_unbound_frame_operand(self):
        """Verify that analysis fails when a frame operand precedes its defining frame."""
        freq = ConstantOp(FrequencyAttr(5.1e9))
        frame = CreateFrameOp(freq, StringAttr("q0/drive"))
        duration = ConstantOp(TimeAttr(16e-9))
        wait = WaitOp(frame, duration)
        module = _module_with_main([duration, wait, freq, frame, func.ReturnOp()])

        with pytest.raises(PassFailedException, match="Unbound frame operand"):
            build_frame_lineage_analysis(module)

    def test_raises_when_control_regions_are_present(self):
        """Verify that region-bearing entry operations are rejected by the analysis."""
        zero = ArithConstantOp.from_int_and_width(0, IndexType())
        one = ArithConstantOp.from_int_and_width(1, IndexType())
        ten = ArithConstantOp.from_int_and_width(10, IndexType())
        loop = scf.ForOp(zero, ten, one, [], Block(arg_types=[IndexType()]))
        module = _module_with_main([zero, ten, one, loop, func.ReturnOp()])

        with pytest.raises(PassFailedException, match="region-free entry blocks"):
            build_frame_lineage_analysis(module)

    def test_frame_operand_without_frame_result_keeps_partition(self):
        """Verify that frame-consuming side effects remain attached to their lineage."""
        freq = ConstantOp(FrequencyAttr(5.1e9))
        frame = CreateFrameOp(freq, StringAttr("q0/drive"))
        side_effect = _FrameSideEffectOp(frame)
        analysis = build_frame_lineage_analysis(
            _module_with_main([freq, frame, side_effect, func.ReturnOp()])
        )
        assert analysis.frame_to_operations[frame.result] == (frame, side_effect)

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
        assert analysis.frame_to_operations[frame.result] == (frame, sync)

    def test_compute_pass_apply_sets_analysis(self):
        """Verify that the compute pass stores the analysis it produces."""
        freq = ConstantOp(FrequencyAttr(4.8e9))
        frame = CreateFrameOp(freq, StringAttr("q0/drive"))
        module = _module_with_main([freq, frame, func.ReturnOp()])
        pass_instance = FrameLineagePass()
        pass_instance.apply(Context(), module)
        assert pass_instance.analysis is not None
