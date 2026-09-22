# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest
from xdsl.context import Context
from xdsl.dialects import func
from xdsl.dialects.arith import ConstantOp as ArithConstantOp, IndexCastOp
from xdsl.dialects.builtin import IndexType, ModuleOp, StringAttr, i32
from xdsl.dialects.scf import ForOp, YieldOp
from xdsl.ir import Block, Region
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, region_def, result_def
from xdsl.utils.exceptions import PassFailedException

from qat.experimental.conversion.pulse_to_q1.sequence_outlining import (
    Q1OutliningPass,
    _normalize_sequence_symbol,
    _SymbolAllocator,
)
from qat.experimental.dialect.pulse.ir import (
    AcquireOp,
    AcquisitionType,
    AmplitudeAttr,
    ConstantOp,
    CreateFrameOp,
    FrequencyAttr,
    MaxTimeOp,
    PulseOp,
    SquareWaveformOp,
    SynchronizeOp,
    TimeAttr,
    TimeType,
    WaitOp,
)
from qat.experimental.dialect.pulse.transforms.partition_by_frame import (
    FrameLineage,
    FrameLineageAnalysis,
    FrameNode,
)
from qat.experimental.dialect.q1 import SetMrkImmOp, StopOp
from qat.experimental.dialect.q1_sequence import SequenceOp
from qat.experimental.dialect.results.ir import CreateOp, StoreOp

_SHOTS = 1000


def _module_with_main(ops) -> ModuleOp:
    return ModuleOp([func.FuncOp("main", ((), ()), Region(Block(ops)))])


@irdl_op_definition
class _ContainerOp(IRDLOperation):
    """Region-bearing op used in tests to simulate SCF structures."""

    name = "test.container"
    body = region_def()

    def __init__(self, body: Region):
        super().__init__(regions=[body])


@irdl_op_definition
class _ResultContainerOp(IRDLOperation):
    """Region-bearing op with a scalar result used to test dependency ownership."""

    name = "test.result_container"
    body = region_def()
    result = result_def(TimeType)

    def __init__(self, body: Region):
        super().__init__(regions=[body], result_types=[TimeType()])


@irdl_op_definition
class _AcquisitionConsumerOp(IRDLOperation):
    """Consumes an acquisition result; stands in for a consumer fission would drop."""

    name = "test.acquisition_consumer"
    acquisition = operand_def(AcquisitionType)

    def __init__(self, acquisition):
        super().__init__(operands=[acquisition])


def _frame(frequency: float, channel_id: str) -> tuple[ConstantOp, CreateFrameOp]:
    freq = ConstantOp(FrequencyAttr(frequency))
    return freq, CreateFrameOp(freq, StringAttr(channel_id))


def _square_pulse(frame: CreateFrameOp) -> list:
    """A square pulse played on `frame`, with the constants it consumes."""
    width = ConstantOp(TimeAttr(64e-9))
    amplitude = ConstantOp(AmplitudeAttr(1.0))
    waveform = SquareWaveformOp(width, amplitude)
    return [width, amplitude, waveform, PulseOp(frame, waveform)]


def _shot_loop_module(
    measure_channels: tuple[str, ...] = ("q0/measure",), *, shared_array: bool = False
) -> ModuleOp:
    """A shot loop driving one frame and acquiring on each of `measure_channels`.

    Every acquisition is recorded into a loop-carried results array. By default each
    channel is given its own array, matching what results-collection lowering produces.
    With `shared_array` the channels are instead chained through a single array.
    """
    array_count = 1 if shared_array else len(measure_channels)
    arrays = [CreateOp.for_array(AcquisitionType(), _SHOTS) for _ in range(array_count)]
    bounds = [
        ArithConstantOp.from_int_and_width(value, IndexType()) for value in (0, _SHOTS, 1)
    ]
    body = Block(arg_types=(IndexType(), *(array.result.type for array in arrays)))

    drive_freq, drive = _frame(4.8e9, "q0/drive")
    index = IndexCastOp(body.args[0], i32)
    ops = [drive_freq, drive, *_square_pulse(drive), index]

    carried = list(body.args[1:])
    for position, channel in enumerate(measure_channels):
        measure_freq, measure = _frame(7.1e9 + position * 1e8, channel)
        duration = ConstantOp(TimeAttr(1e-6))
        acquire = AcquireOp(measure, duration)
        slot = 0 if shared_array else position
        store = StoreOp.value_in_array(carried[slot], index, acquire.acquisition_result)
        carried[slot] = store.result
        ops += [measure_freq, measure, duration, acquire, store]

    body.add_ops([*ops, YieldOp(*carried)])
    loop = ForOp(*bounds, [array.result for array in arrays], body)
    return _module_with_main([*arrays, *bounds, loop, func.ReturnOp()])


def _outlined_loops(module: ModuleOp) -> dict[str, ForOp]:
    """Map each outlined sequence's port id to the shot loop in its body."""
    return {
        seq.port_id.data: next(op for op in seq.body.block.ops if isinstance(op, ForOp))
        for seq in module.body.block.ops
        if isinstance(seq, SequenceOp)
    }


def _sequences_by_port(module: ModuleOp) -> dict[str, SequenceOp]:
    return {
        seq.port_id.data: seq
        for seq in module.body.block.ops
        if isinstance(seq, SequenceOp)
    }


class TestPulseToQ1SequenceOutlining:
    def test_single_frame_lowers_to_single_sequence(self):
        """Verify that one logical frame yields one outlined Q1 sequence."""
        freq, frame = _frame(4.8e9, "q0.drive")
        module = _module_with_main([freq, frame, func.ReturnOp()])

        pass_instance = Q1OutliningPass()
        pass_instance.apply(Context(), module)

        [seq] = list(module.body.block.ops)
        assert isinstance(seq, SequenceOp)
        assert seq.channel_id.data == "q0.drive"
        assert seq.port_id.data == "q0.drive"
        assert isinstance(seq.body.block.first_op, SetMrkImmOp)
        assert seq.body.block.first_op.mrk.data == 3
        assert any(isinstance(op, CreateFrameOp) for op in seq.body.block.ops)
        assert isinstance(seq.body.block.last_op, StopOp)
        seq.verify()

    def test_distinct_frames_yield_distinct_sequences(self):
        """Verify that distinct frame lineages remain distinct after outlining."""
        f0_freq, f0 = _frame(4.8e9, "q0.drive")
        f1_freq, f1 = _frame(5.2e9, "q1.drive")
        module = _module_with_main([f0_freq, f0, f1_freq, f1, func.ReturnOp()])

        pass_instance = Q1OutliningPass()
        pass_instance.apply(Context(), module)

        sequences = [op for op in module.body.block.ops if isinstance(op, SequenceOp)]
        assert [seq.channel_id.data for seq in sequences] == ["q0.drive", "q1.drive"]
        assert [seq.port_id.data for seq in sequences] == ["q0.drive", "q1.drive"]
        assert all(isinstance(seq.body.block.first_op, SetMrkImmOp) for seq in sequences)
        assert all(seq.body.block.first_op.mrk.data == 3 for seq in sequences)
        assert all(
            any(isinstance(op, CreateFrameOp) for op in seq.body.block.ops)
            for seq in sequences
        )
        assert all(isinstance(seq.body.block.last_op, StopOp) for seq in sequences)
        assert pass_instance.state.frame_to_port == {
            "frame_0": "q0.drive",
            "frame_1": "q1.drive",
        }
        assert pass_instance.state.frame_to_sequence == {
            "frame_0": "q0.drive",
            "frame_1": "q1.drive",
        }

    def test_sequence_symbol_normalizes_channel_slashes(self):
        """Verify that channel tokens are normalised before use as sequence symbols."""
        freq, frame = _frame(4.8e9, "q0/drive")
        module = _module_with_main([freq, frame, func.ReturnOp()])

        pass_instance = Q1OutliningPass()
        pass_instance.apply(Context(), module)

        [seq] = [op for op in module.body.block.ops if isinstance(op, SequenceOp)]
        assert seq.channel_id.data == "q0_drive"
        assert seq.port_id.data == "q0/drive"
        assert pass_instance.state.frame_to_port == {"frame_0": "q0/drive"}
        assert pass_instance.state.frame_to_sequence == {"frame_0": "q0_drive"}

    def test_frames_sharing_channel_id_remain_separate_sequences(self):
        """Verify that shared physical ports do not merge distinct logical frames."""
        f0_freq, f0 = _frame(4.8e9, "shared.port")
        f1_freq, f1 = _frame(5.2e9, "shared.port")
        module = _module_with_main([f0_freq, f0, f1_freq, f1, func.ReturnOp()])

        pass_instance = Q1OutliningPass()
        pass_instance.apply(Context(), module)

        sequences = [op for op in module.body.block.ops if isinstance(op, SequenceOp)]
        assert len(sequences) == 2
        assert [seq.channel_id.data for seq in sequences] == ["frame_0", "frame_1"]
        assert [seq.port_id.data for seq in sequences] == ["shared.port", "shared.port"]
        assert pass_instance.state.frame_to_port == {
            "frame_0": "shared.port",
            "frame_1": "shared.port",
        }
        assert pass_instance.state.frame_to_sequence == {
            "frame_0": "frame_0",
            "frame_1": "frame_1",
        }

    def test_empty_module_lowers_to_no_sequences(self):
        """Verify that outlining preserves an empty module as empty."""
        module = ModuleOp([])

        pass_instance = Q1OutliningPass()
        pass_instance.apply(Context(), module)

        assert list(module.body.block.ops) == []
        assert pass_instance.state.frame_to_port == {}
        assert pass_instance.state.frame_to_sequence == {}

    def test_emit_sequence_ops_rejects_partition_not_starting_with_create_frame(self):
        """Verify that malformed frame partitions are rejected during sequence emission."""
        freq, frame = _frame(4.8e9, "q0.drive")
        malformed_root = FrameNode(op=frame, parent=None)
        malformed_lin = FrameLineage(
            create_frame=frame,
            port="q0.drive",
            related_ops=[FrameNode(op=func.ReturnOp(), parent=malformed_root)],
        )
        malformed = FrameLineageAnalysis.from_lineages([malformed_lin])
        with pytest.raises(ValueError, match="does not contain pulse.create_frame"):
            Q1OutliningPass()._emit_sequence_ops(ModuleOp([]), malformed)

    def test_scf_container_with_frame_op_is_included_in_sequence(self):
        """Verify that a region-bearing op referencing a frame is placed in its outlined
        sequence."""
        freq, frame_op = _frame(4.8e9, "q0.drive")
        duration = ConstantOp(TimeAttr(16e-9))
        wait = WaitOp(frame_op, duration)
        container = _ContainerOp(Region(Block([wait])))
        module = _module_with_main([freq, frame_op, duration, container, func.ReturnOp()])

        pass_instance = Q1OutliningPass()
        pass_instance.apply(Context(), module)

        [seq] = [op for op in module.body.block.ops if isinstance(op, SequenceOp)]
        body_ops = list(seq.body.block.ops)
        assert any(isinstance(op, _ContainerOp) for op in body_ops)
        assert any(isinstance(op, CreateFrameOp) for op in body_ops)
        assert isinstance(seq.body.block.first_op, SetMrkImmOp)
        assert isinstance(seq.body.block.last_op, StopOp)

    def test_outer_constant_free_var_in_container_body_is_pulled_in(self):
        """Verify that entry-block constants used as free variables inside a nested region
        are included in the outlined sequence."""
        freq, frame_op = _frame(4.8e9, "q0.drive")
        duration = ConstantOp(TimeAttr(16e-9))
        wait = WaitOp(frame_op, duration)
        container = _ContainerOp(Region(Block([wait])))
        module = _module_with_main([freq, frame_op, duration, container, func.ReturnOp()])

        pass_instance = Q1OutliningPass()
        pass_instance.apply(Context(), module)

        [seq] = [op for op in module.body.block.ops if isinstance(op, SequenceOp)]
        body_ops = list(seq.body.block.ops)
        # duration is a free variable in the container body, not a direct operand.
        assert any(
            isinstance(op, ConstantOp) and op.value == duration.value for op in body_ops
        )

    def test_nested_region_bearing_ops_are_captured_transitively(self):
        """Verify that a frame referenced inside a doubly-nested region causes the outermost
        container to be included in the sequence."""
        freq, frame_op = _frame(4.8e9, "q0.drive")
        duration = ConstantOp(TimeAttr(16e-9))
        wait = WaitOp(frame_op, duration)
        inner = _ContainerOp(Region(Block([wait])))
        outer = _ContainerOp(Region(Block([inner])))
        module = _module_with_main([freq, frame_op, duration, outer, func.ReturnOp()])

        pass_instance = Q1OutliningPass()
        pass_instance.apply(Context(), module)

        [seq] = [op for op in module.body.block.ops if isinstance(op, SequenceOp)]
        body_ops = list(seq.body.block.ops)
        assert any(isinstance(op, _ContainerOp) for op in body_ops)
        assert any(isinstance(op, CreateFrameOp) for op in body_ops)

    def test_unused_constant_is_not_pulled_into_sequence(self):
        """Verify that a constant not referenced by any lineage is excluded from the
        outlined sequence."""
        freq, frame_op = _frame(4.8e9, "q0.drive")
        duration = ConstantOp(TimeAttr(16e-9))
        wait = WaitOp(frame_op, duration)
        unused = ConstantOp(TimeAttr(32e-9))
        module = _module_with_main(
            [freq, frame_op, duration, wait, unused, func.ReturnOp()]
        )

        pass_instance = Q1OutliningPass()
        pass_instance.apply(Context(), module)

        [seq] = [op for op in module.body.block.ops if isinstance(op, SequenceOp)]
        body_ops = list(seq.body.block.ops)
        assert not any(
            isinstance(op, ConstantOp) and op.value == unused.value for op in body_ops
        )

    def test_constant_shared_by_two_lineages_is_cloned_into_both_sequences(self):
        """Verify that a duration constant shared by two frame lineages is cloned into each
        outlined sequence body, keeping each sequence self-contained."""
        f0_freq, f0 = _frame(4.8e9, "q0.drive")
        f1_freq, f1 = _frame(5.2e9, "q1.drive")
        duration = ConstantOp(TimeAttr(16e-9))
        wait_0 = WaitOp(f0, duration)
        wait_1 = WaitOp(f1, duration)
        module = _module_with_main(
            [f0_freq, f0, f1_freq, f1, duration, wait_0, wait_1, func.ReturnOp()]
        )

        pass_instance = Q1OutliningPass()
        pass_instance.apply(Context(), module)

        sequences = [op for op in module.body.block.ops if isinstance(op, SequenceOp)]
        assert len(sequences) == 2
        for seq in sequences:
            body_ops = list(seq.body.block.ops)
            assert any(
                isinstance(op, ConstantOp) and op.value == duration.value for op in body_ops
            )

    def test_chain_of_dependencies_are_all_pulled_into_sequence(self):
        """Verify that a multi-level dependency chain feeding a lineage op is pulled in
        transitively, not just its immediate operand."""
        freq, frame_op = _frame(4.8e9, "q0.drive")
        time_0 = ConstantOp(TimeAttr(16e-9))
        time_1 = ConstantOp(TimeAttr(32e-9))
        combined = MaxTimeOp(time_0, time_1)
        wait = WaitOp(frame_op, combined)
        module = _module_with_main(
            [freq, frame_op, time_0, time_1, combined, wait, func.ReturnOp()]
        )

        pass_instance = Q1OutliningPass()
        pass_instance.apply(Context(), module)

        [seq] = [op for op in module.body.block.ops if isinstance(op, SequenceOp)]
        body_ops = list(seq.body.block.ops)
        assert any(
            isinstance(op, ConstantOp) and op.value == time_0.value for op in body_ops
        )
        assert any(
            isinstance(op, ConstantOp) and op.value == time_1.value for op in body_ops
        )
        assert any(isinstance(op, MaxTimeOp) for op in body_ops)

    def test_nested_create_frame_op_is_outlined_into_self_contained_sequence(self):
        """Verify that a CreateFrameOp created inside a nested region is still outlined into
        its own self-contained sequence."""
        freq = ConstantOp(FrequencyAttr(4.8e9))
        frame_op = CreateFrameOp(freq, StringAttr("q0.drive"))
        container = _ContainerOp(Region(Block([freq, frame_op])))
        module = _module_with_main([container, func.ReturnOp()])

        pass_instance = Q1OutliningPass()
        pass_instance.apply(Context(), module)

        [seq] = [op for op in module.body.block.ops if isinstance(op, SequenceOp)]
        assert seq.channel_id.data == "q0.drive"
        body_ops = list(seq.body.block.ops)
        assert any(isinstance(op, _ContainerOp) for op in body_ops)
        assert any(
            isinstance(nested, CreateFrameOp) for op in body_ops for nested in op.walk()
        )

    def test_container_enclosing_two_lineages_is_rejected(self):
        """A non-loop region enclosing two frames cannot be split.

        Only the shot loop is split into per-sequencer copies. Splitting an arbitrary region
        would need rules for its regions and block arguments, and no Pulse producer emits
        one enclosing more than one frame.
        """
        f0_freq, f0 = _frame(4.8e9, "q0.drive")
        f1_freq, f1 = _frame(5.2e9, "q1.drive")
        container = _ContainerOp(Region(Block([f0_freq, f0, f1_freq, f1])))
        module = _module_with_main([container, func.ReturnOp()])

        with pytest.raises(PassFailedException, match="Only scf.for shot loops are split"):
            Q1OutliningPass().apply(Context(), module)

    def test_synchronisation_across_frames_is_rejected(self):
        """Cross-frame synchronisation must be lowered before outlining: fission cannot
        preserve it, since each partition becomes an independent program."""
        f0_freq, f0 = _frame(4.8e9, "q0.drive")
        f1_freq, f1 = _frame(5.2e9, "q1.drive")
        sync = SynchronizeOp(f0, f1)
        module = _module_with_main([f0_freq, f0, f1_freq, f1, sync, func.ReturnOp()])

        with pytest.raises(PassFailedException, match="spans multiple frame lineages"):
            Q1OutliningPass().apply(Context(), module)

    def test_dependency_owned_by_another_lineage_is_rejected(self):
        """A region containing one lineage cannot feed a different sequence."""
        f0_freq, f0 = _frame(4.8e9, "q0.drive")
        f1_freq, f1 = _frame(5.2e9, "q1.drive")
        producer = _ResultContainerOp(Region(Block([f1_freq, f1])))
        wait = WaitOp(f0, producer.result)
        module = _module_with_main([f0_freq, f0, producer, wait, func.ReturnOp()])

        with pytest.raises(PassFailedException, match="owned by another frame lineage"):
            Q1OutliningPass().apply(Context(), module)


class TestNormalizeSequenceSymbol:
    @pytest.mark.parametrize(
        "channel_token, expected",
        [
            ("q0/drive", "q0_drive"),
            ("q0.drive", "q0.drive"),
            ("q0$drive", "q0$drive"),
            ("q0_drive", "q0_drive"),
            ("1drive", "_1drive"),
            ("///", "sequence"),
            ("__q0__", "q0"),
            ("q0//drive", "q0_drive"),
            # All-underscore input: collapse + strip yields empty → fallback.
            ("_______", "sequence"),
        ],
    )
    def test_normalises_channel_token_to_valid_symbol(self, channel_token, expected):
        """Verify normalisation rules for a range of representative channel tokens."""
        assert _normalize_sequence_symbol(channel_token) == expected


class TestSequenceSymbolAllocator:
    def test_collision_between_unique_channels_falls_back_to_frame_id(self):
        """Verify that normalisation collision across distinct channels uses frame_i names.

        When two channel tokens normalise to the same symbol, the second
        allocation must not claim the already-emitted symbol and must fall back
        to its ``frame_i`` identifier instead.
        """
        freq_0, frame_0 = _frame(4.8e9, "q0/drive")
        freq_1, frame_1 = _frame(5.2e9, "q0_drive")

        lin_0 = FrameLineage(create_frame=frame_0, port="q0/drive")
        lin_1 = FrameLineage(create_frame=frame_1, port="q0_drive")
        reserved = {"frame_0", "frame_1"}
        allocator = _SymbolAllocator(
            symbol_counts={"q0/drive": 1, "q0_drive": 1},
            used_sequence_symbols=reserved,
        )

        _, sym_0 = allocator.allocate("frame_0", lin_0)
        _, sym_1 = allocator.allocate("frame_1", lin_1)

        assert sym_0 == "q0_drive"
        assert sym_1 == "frame_1"

    def test_normalised_symbol_cannot_steal_frame_i_fallback_name(self):
        """Verify that pre-reserving frame_i names prevents a normalised symbol from
        claiming them.

        A channel token that normalises to ``frame_1`` must not be emitted as
        ``frame_1`` because that name is reserved as the fallback for the
        second partition. The allocator must fall back to the next available
        symbol instead.
        """
        freq_0, frame_0 = _frame(4.8e9, "frame_1")
        freq_1, frame_1 = _frame(5.2e9, "q1/drive")

        lin_0 = FrameLineage(create_frame=frame_0, port="frame_1")
        lin_1 = FrameLineage(create_frame=frame_1, port="q1/drive")
        reserved = {"frame_0", "frame_1"}
        allocator = _SymbolAllocator(
            symbol_counts={"frame_1": 1, "q1/drive": 1},
            used_sequence_symbols=reserved,
        )

        _, sym_0 = allocator.allocate("frame_0", lin_0)
        _, sym_1 = allocator.allocate("frame_1", lin_1)

        assert sym_0 == "frame_0"
        assert sym_1 == "q1_drive"


class TestPulseToQ1ShotLoopFission:
    """Outlining a shot loop that spans every frame.

    The loop belongs to no single lineage, so each partition receives a private copy pruned
    to its own work rather than a share of one loop.
    """

    def test_shot_loop_is_fissioned_into_one_loop_per_frame(self):
        """Verify that a loop spanning two frames becomes one loop per sequence."""
        module = _shot_loop_module()

        Q1OutliningPass().apply(Context(), module)
        module.verify()

        assert sorted(_outlined_loops(module)) == ["q0/drive", "q0/measure"]

    def test_fissioned_drive_loop_drops_the_acquisition_plumbing(self):
        """Verify that a partition with no acquisition keeps neither the results array nor
        the loop-carried argument threading it."""
        module = _shot_loop_module()

        Q1OutliningPass().apply(Context(), module)

        sequence = _sequences_by_port(module)["q0/drive"]
        loop = _outlined_loops(module)["q0/drive"]
        body = [op.name for op in loop.body.block.ops]
        assert len(loop.iter_args) == 0
        assert "pulse.pulse" in body
        assert "pulse.acquire" not in body
        assert "results.store" not in body
        assert not any(isinstance(op, CreateOp) for op in sequence.body.block.ops)

    def test_fissioned_measure_loop_drops_its_results_bookkeeping(self):
        """Verify that an acquiring partition keeps its acquisition but not the results
        array recording it, nor the other frame's waveform work.

        A hardware sequence records acquisitions into bins: the bin index is re-derived
        from the loop induction variable during acquire lowering, and no Q1 pass lowers
        results arrays, so carrying them into a sequence fails downstream.
        """
        module = _shot_loop_module()

        Q1OutliningPass().apply(Context(), module)
        module.verify()

        sequence = _sequences_by_port(module)["q0/measure"]
        loop = _outlined_loops(module)["q0/measure"]
        body = [op.name for op in loop.body.block.ops]
        assert len(loop.iter_args) == 0
        assert not any(isinstance(op, CreateOp) for op in sequence.body.block.ops)
        assert "pulse.acquire" in body
        assert "results.store" not in body
        assert "pulse.pulse" not in body
        assert "pulse.square_waveform" not in body

    def test_two_acquire_channels_are_outlined_independently(self):
        """Verify that two acquiring frames fission into independent sequences, each
        carrying exactly its own acquisition and nothing of the other's."""
        module = _shot_loop_module(("q0/measure", "q1/measure"))

        Q1OutliningPass().apply(Context(), module)
        module.verify()

        loops = _outlined_loops(module)
        assert sorted(loops) == ["q0/drive", "q0/measure", "q1/measure"]
        for channel, loop in loops.items():
            body = [op.name for op in loop.body.block.ops]
            assert len(loop.iter_args) == 0
            assert "results.store" not in body
            frames = [
                op.port.data for op in loop.body.block.ops if isinstance(op, CreateFrameOp)
            ]
            assert frames == [channel]
        for channel in ("q0/measure", "q1/measure"):
            body = [op.name for op in loops[channel].body.block.ops]
            assert body.count("pulse.acquire") == 1

    def test_frames_chained_through_one_results_array_are_outlined(self):
        """Verify that acquisitions chained through a single shared results array still
        partition cleanly.

        Chained stores could not be split between partitions, but stripping removes the
        whole chain before pruning, so the shape never reaches lineage attribution.
        """
        module = _shot_loop_module(("q0/measure", "q1/measure"), shared_array=True)

        Q1OutliningPass().apply(Context(), module)
        module.verify()

        loops = _outlined_loops(module)
        assert sorted(loops) == ["q0/drive", "q0/measure", "q1/measure"]
        assert all(len(loop.iter_args) == 0 for loop in loops.values())


class TestPulseToQ1ShotLoopFissionRejections:
    """Shot loop shapes fission refuses rather than splitting.

    Each would otherwise produce valid IR describing a different program, so each is
    rejected at the point it is recognised rather than silently dropped.
    """

    def test_frame_carried_between_iterations_is_rejected(self):
        """A frame arriving as a block argument has no resolvable lineage, so the loop body
        would be attributed to no partition and silently vanish."""
        freq, frame = _frame(4.8e9, "q0/drive")
        width, amplitude = ConstantOp(TimeAttr(64e-9)), ConstantOp(AmplitudeAttr(1.0))
        waveform = SquareWaveformOp(width, amplitude)
        bounds = [
            ArithConstantOp.from_int_and_width(value, IndexType())
            for value in (0, _SHOTS, 1)
        ]
        body = Block(arg_types=(IndexType(), frame.result.type))
        pulse = PulseOp(body.args[1], waveform)
        body.add_ops([pulse, YieldOp(pulse.result)])
        loop = ForOp(*bounds, [frame.result], body)
        module = _module_with_main(
            [freq, frame, width, amplitude, waveform, *bounds, loop, func.ReturnOp()]
        )

        with pytest.raises(PassFailedException, match="takes a frame as a block argument"):
            Q1OutliningPass().apply(Context(), module)

    def test_region_bearing_op_inside_the_shot_loop_is_rejected(self):
        """Retained operations are copied from the loop body only, so work nested inside a
        region there would be dropped rather than copied."""
        freq, frame = _frame(4.8e9, "q0/drive")
        width, amplitude = ConstantOp(TimeAttr(64e-9)), ConstantOp(AmplitudeAttr(1.0))
        waveform = SquareWaveformOp(width, amplitude)
        bounds = [
            ArithConstantOp.from_int_and_width(value, IndexType())
            for value in (0, _SHOTS, 1)
        ]
        body = Block(arg_types=(IndexType(),))
        body.add_ops(
            [
                freq,
                frame,
                width,
                amplitude,
                waveform,
                _ContainerOp(Region(Block([PulseOp(frame, waveform)]))),
                YieldOp(),
            ]
        )
        loop = ForOp(*bounds, [], body)
        module = _module_with_main([*bounds, loop, func.ReturnOp()])

        with pytest.raises(PassFailedException, match="is region-bearing"):
            Q1OutliningPass().apply(Context(), module)

    def test_acquisition_consumer_that_would_be_lost_is_rejected(self):
        """Only pulse.integrate and results bookkeeping are accounted for; any other
        consumer would be dropped and the acquisition would mean something else."""
        freq, frame = _frame(7.1e9, "q0/measure")
        duration = ConstantOp(TimeAttr(1e-6))
        acquire = AcquireOp(frame, duration)
        bounds = [
            ArithConstantOp.from_int_and_width(value, IndexType())
            for value in (0, _SHOTS, 1)
        ]
        body = Block(arg_types=(IndexType(),))
        body.add_ops(
            [
                freq,
                frame,
                duration,
                acquire,
                _AcquisitionConsumerOp(acquire.acquisition_result),
                YieldOp(),
            ]
        )
        loop = ForOp(*bounds, [], body)
        module = _module_with_main([*bounds, loop, func.ReturnOp()])

        with pytest.raises(PassFailedException, match="carries only pulse.integrate"):
            Q1OutliningPass().apply(Context(), module)

    def test_partition_consuming_a_rebuilt_loop_result_is_rejected(self):
        """A rebuilt loop carries nothing out, so an operation still needing one of its
        results would be left holding a value from the original loop."""
        freq, frame = _frame(4.8e9, "q0/drive")
        width, amplitude = ConstantOp(TimeAttr(64e-9)), ConstantOp(AmplitudeAttr(1.0))
        waveform = SquareWaveformOp(width, amplitude)
        bounds = [
            ArithConstantOp.from_int_and_width(value, IndexType())
            for value in (0, _SHOTS, 1)
        ]
        body = Block(arg_types=(IndexType(), width.result.type))
        body.add_ops([PulseOp(frame, waveform), YieldOp(body.args[1])])
        loop = ForOp(*bounds, [width.result], body)
        module = _module_with_main(
            [
                freq,
                frame,
                width,
                amplitude,
                waveform,
                *bounds,
                loop,
                WaitOp(frame, loop.results[0]),
                func.ReturnOp(),
            ]
        )

        with pytest.raises(PassFailedException, match="rebuilt without loop-carried"):
            Q1OutliningPass().apply(Context(), module)
