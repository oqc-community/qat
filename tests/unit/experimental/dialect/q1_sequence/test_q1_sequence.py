# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025-2026 Oxford Quantum Circuits Ltd

from xdsl.dialects.builtin import ModuleOp

from qat.experimental.dialect.q1 import StopOp
from qat.experimental.dialect.q1_sequence import Q1_sequence, SequenceOp


class TestDialectRegistration:
    def test_dialect_ops(self):
        op_names = {op.name for op in Q1_sequence.operations}
        assert op_names == {"q1_sequence.sequence"}

    def test_dialect_attrs(self):
        attr_names = {attr.name for attr in Q1_sequence.attributes}
        assert attr_names == {
            "q1_sequence.waveform",
            "q1_sequence.weight",
            "q1_sequence.acquisition",
            "q1_sequence.acquisition_path_connection",
            "q1_sequence.connection",
            "q1_sequence.direction_kind",
            "q1_sequence.nco_config",
            "q1_sequence.awg_config",
            "q1_sequence.unweighted_acquisition_config",
            "q1_sequence.thresholded_acq_config",
            "q1_sequence.acquire_config",
            "q1_sequence.marker_override_config",
            "q1_sequence.sequencer_config",
            "q1_sequence.mixer_correction_config",
            "q1_sequence.real_time_predistortion_config",
            "q1_sequence.sequencer_path",
            "q1_sequence.output_signal_config",
            "q1_sequence.input_signal_config",
            "q1_sequence.scope_acquire_config",
            "q1_sequence.local_oscillator_config",
            "q1_sequence.output_config",
            "q1_sequence.output_path_connection",
            "q1_sequence.input_config",
            "q1_sequence.module_config",
            "q1_sequence.module_kind",
            "q1_sequence.waveform_table_index",
            "q1_sequence.weight_table_index",
            "q1_sequence.acq_table_index",
            "q1_sequence.bin_count_imm",
            "q1_sequence.integration_length_imm",
            "q1_sequence.slot_index",
            "q1_sequence.sequencer_index",
        }

    def test_sequence_in_module(self):
        seq = SequenceOp("ch0", [StopOp()])
        module = ModuleOp([seq])
        found = [op for op in module.body.block.ops if isinstance(op, SequenceOp)]
        assert len(found) == 1
        assert found[0].channel_id.data == "ch0"

    def test_multiple_sequences_in_module(self):
        s0 = SequenceOp("drive", [StopOp()])
        s1 = SequenceOp("readout", [StopOp()])
        module = ModuleOp([s0, s1])
        names = [
            op.channel_id.data for op in module.body.block.ops if isinstance(op, SequenceOp)
        ]
        assert names == ["drive", "readout"]
