# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import json
from io import StringIO

import pytest
from xdsl.context import Context
from xdsl.dialects.builtin import ArrayAttr, ModuleOp

from qat.experimental.dialect.q1.ir.ops import NopOp, StopOp
from qat.experimental.dialect.q1.target import emit_program
from qat.experimental.dialect.q1_sequence.attrs import (
    make_acquisition,
    make_waveform,
    make_weight,
)
from qat.experimental.dialect.q1_sequence.ops import SequenceOp
from qat.experimental.dialect.q1_sequence.target import (
    Q1SequenceTarget,
    emit_module,
    emit_sequence,
)


class TestEmitProgram:
    def test_stop_only(self):
        seq = SequenceOp("ch0", [StopOp()])
        stream = StringIO()
        emit_program(seq.body, stream)
        assert stream.getvalue() == "stop\n"

    def test_nop_stop(self):
        seq = SequenceOp("ch0", [NopOp(), StopOp()])
        stream = StringIO()
        emit_program(seq.body, stream)
        assert stream.getvalue() == "nop\nstop\n"


class TestEmitSequence:
    def test_minimal(self):
        seq = SequenceOp("ch0", [StopOp()])
        result = emit_sequence(seq)
        assert result["program"] == "stop\n"
        assert result["waveforms"] == {}
        assert result["weights"] == {}
        assert result["acquisitions"] == {}

    def test_with_waveform(self):
        wf = make_waveform("wf0", 0, [0.5, 1.0])
        seq = SequenceOp("ch0", [StopOp()], waveforms=ArrayAttr([wf]))
        result = emit_sequence(seq)
        assert "wf0" in result["waveforms"]
        entry = result["waveforms"]["wf0"]
        assert entry["index"] == 0
        assert len(entry["data"]) == 2

    def test_with_weight(self):
        w = make_weight("w0", 0, [1.0])
        seq = SequenceOp("ch0", [StopOp()], weights=ArrayAttr([w]))
        result = emit_sequence(seq)
        assert "w0" in result["weights"]
        assert result["weights"]["w0"]["index"] == 0

    def test_with_acquisition(self):
        acq = make_acquisition("acq0", 0, 4)
        seq = SequenceOp("ch0", [StopOp()], acquisitions=ArrayAttr([acq]))
        result = emit_sequence(seq)
        assert "acq0" in result["acquisitions"]
        entry = result["acquisitions"]["acq0"]
        assert entry["num_bins"] == 4
        assert entry["index"] == 0

    def test_full_sequence(self):
        wf = make_waveform("wf0", 0, [0.1, 0.2])
        w = make_weight("w0", 0, [1.0, 0.0])
        acq = make_acquisition("acq0", 0, 1)
        seq = SequenceOp(
            "ch0",
            [NopOp(), StopOp()],
            waveforms=ArrayAttr([wf]),
            weights=ArrayAttr([w]),
            acquisitions=ArrayAttr([acq]),
        )
        result = emit_sequence(seq)
        assert result["program"] == "nop\nstop\n"
        assert len(result["waveforms"]) == 1
        assert len(result["weights"]) == 1
        assert len(result["acquisitions"]) == 1


class TestEmitModule:
    def test_single_sequence(self):
        seq = SequenceOp("drive", [StopOp()])
        module = ModuleOp([seq])
        result = emit_module(module)
        assert "drive" in result
        assert result["drive"]["program"] == "stop\n"

    def test_multiple_sequences(self):
        s0 = SequenceOp("drive", [NopOp(), StopOp()])
        s1 = SequenceOp("readout", [StopOp()])
        module = ModuleOp([s0, s1])
        result = emit_module(module)
        assert set(result.keys()) == {"drive", "readout"}
        assert result["drive"]["program"] == "nop\nstop\n"
        assert result["readout"]["program"] == "stop\n"

    def test_empty_module(self):
        module = ModuleOp([])
        result = emit_module(module)
        assert result == {}

    def test_sequences_with_data(self):
        wf = make_waveform("wf0", 0, [0.5])
        acq = make_acquisition("acq0", 0, 1)
        s0 = SequenceOp(
            "drive",
            [StopOp()],
            waveforms=ArrayAttr([wf]),
        )
        s1 = SequenceOp(
            "readout",
            [StopOp()],
            acquisitions=ArrayAttr([acq]),
        )
        module = ModuleOp([s0, s1])
        result = emit_module(module)
        assert len(result["drive"]["waveforms"]) == 1
        assert len(result["readout"]["acquisitions"]) == 1


class TestQ1SequenceTarget:
    def test_emit_json(self):
        seq = SequenceOp("drive", [NopOp(), StopOp()])
        module = ModuleOp([seq])

        stream = StringIO()
        Q1SequenceTarget().emit(Context(), module, stream)
        parsed = json.loads(stream.getvalue())

        assert "drive" in parsed
        assert parsed["drive"]["program"] == "nop\nstop\n"

    def test_emit_matches_emit_module(self):
        s0 = SequenceOp("drive", [StopOp()])
        s1 = SequenceOp("readout", [StopOp()])
        module = ModuleOp([s0, s1])

        stream = StringIO()
        Q1SequenceTarget().emit(Context(), module, stream)
        target_result = json.loads(stream.getvalue())

        direct_result = emit_module(module)
        assert target_result == direct_result

    def test_duplicate_channel_id_fails(self):
        s0 = SequenceOp("drive", [StopOp()])
        s1 = SequenceOp("drive", [StopOp()])
        module = ModuleOp([s0, s1])
        with pytest.raises(ValueError, match="Duplicate channel_id 'drive'"):
            emit_module(module)
