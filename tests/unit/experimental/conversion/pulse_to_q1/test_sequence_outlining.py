# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest
from xdsl.context import Context
from xdsl.dialects import func
from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.ir import Block, Region

from qat.backend.qblox.target_data import QbloxTargetData
from qat.experimental.conversion.pulse_to_q1.sequence_outlining import (
    Q1OutliningPass,
    _normalize_sequence_symbol,
    _SymbolAllocator,
)
from qat.experimental.dialect.pulse.ir import ConstantOp, CreateFrameOp, FrequencyAttr
from qat.experimental.dialect.pulse.transforms.partition_by_frame import (
    FrameLineageAnalysis,
)
from qat.experimental.dialect.q1 import SetMrkImmOp, StopOp
from qat.experimental.dialect.q1_sequence import SequenceOp


def _module_with_main(ops) -> ModuleOp:
    return ModuleOp([func.FuncOp("main", ((), ()), Region(Block(ops)))])


def _frame(frequency: float, channel_id: str) -> tuple[ConstantOp, CreateFrameOp]:
    freq = ConstantOp(FrequencyAttr(frequency))
    return freq, CreateFrameOp(freq, StringAttr(channel_id))


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

    def test_target_data_is_reachable_from_pass(self):
        """Verify that target data supplied at construction remains accessible on the
        pass."""
        target_data = QbloxTargetData()
        pass_instance = Q1OutliningPass(target_data=target_data)
        assert pass_instance.target_data is target_data
        assert pass_instance.target_data.CONTROL_SEQUENCER_DATA.grid_time == 4

    def test_emit_sequence_ops_rejects_partition_not_starting_with_create_frame(self):
        """Verify that malformed frame partitions are rejected during sequence emission."""
        freq, frame = _frame(4.8e9, "q0.drive")
        malformed = FrameLineageAnalysis(
            frame_to_operations={frame.result: (func.ReturnOp(),)},
            frame_to_port={frame.result: "q0.drive"},
            port_to_frames={"q0.drive": (frame.result,)},
            value_to_frame={frame.result: frame.result},
        )
        with pytest.raises(ValueError, match="does not contain pulse.create_frame"):
            Q1OutliningPass()._emit_sequence_ops(ModuleOp([]), malformed)


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

        analysis = FrameLineageAnalysis(
            frame_to_operations={
                frame_0.result: (frame_0,),
                frame_1.result: (frame_1,),
            },
            frame_to_port={
                frame_0.result: "q0/drive",
                frame_1.result: "q0_drive",
            },
            port_to_frames={
                "q0/drive": (frame_0.result,),
                "q0_drive": (frame_1.result,),
            },
            value_to_frame={
                frame_0.result: frame_0.result,
                frame_1.result: frame_1.result,
            },
        )
        reserved = {"frame_0", "frame_1"}
        allocator = _SymbolAllocator(
            symbol_counts={"q0/drive": 1, "q0_drive": 1},
            used_sequence_symbols=reserved,
        )

        _, sym_0 = allocator.allocate("frame_0", frame_0.result, analysis)
        _, sym_1 = allocator.allocate("frame_1", frame_1.result, analysis)

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

        analysis = FrameLineageAnalysis(
            frame_to_operations={
                frame_0.result: (frame_0,),
                frame_1.result: (frame_1,),
            },
            frame_to_port={
                frame_0.result: "frame_1",
                frame_1.result: "q1/drive",
            },
            port_to_frames={
                "frame_1": (frame_0.result,),
                "q1/drive": (frame_1.result,),
            },
            value_to_frame={
                frame_0.result: frame_0.result,
                frame_1.result: frame_1.result,
            },
        )
        reserved = {"frame_0", "frame_1"}
        allocator = _SymbolAllocator(
            symbol_counts={"frame_1": 1, "q1/drive": 1},
            used_sequence_symbols=reserved,
        )

        _, sym_0 = allocator.allocate("frame_0", frame_0.result, analysis)
        _, sym_1 = allocator.allocate("frame_1", frame_1.result, analysis)

        assert sym_0 == "frame_0"
        assert sym_1 == "q1_drive"
