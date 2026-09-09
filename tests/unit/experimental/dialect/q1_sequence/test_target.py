# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025-2026 Oxford Quantum Circuits Ltd

import json
from io import StringIO

import pytest
from xdsl.context import Context
from xdsl.dialects.builtin import ArrayAttr, ModuleOp
from xdsl.utils.exceptions import VerifyException

from qat.experimental.dialect.q1 import NopOp, StopOp, emit_program
from qat.experimental.dialect.q1_sequence import Q1SequenceTarget, SequenceOp
from qat.experimental.dialect.q1_sequence.ir.attrs import (
    AcquireConfigAttr,
    AcquisitionPathConnectionAttr,
    AwgConfigAttr,
    ConnectionAttr,
    InputConfigAttr,
    InputSignalConfigAttr,
    LocalOscillatorConfigAttr,
    ModuleConfigAttr,
    NcoConfigAttr,
    OutputConfigAttr,
    OutputPathConnectionAttr,
    ScopeAcquireConfigAttr,
    SequencerConfigAttr,
    ThresholdedAcquireConfigAttr,
    UnweightedAcquireConfigAttr,
    make_acquisition,
    make_waveform,
    make_weight,
)
from qat.experimental.dialect.q1_sequence.target import (
    emit_config,
    emit_module,
    emit_sequence,
)
from qat.experimental.system_data.hardware.qblox.models import (
    DirectionKind,
    QbloxModuleKind,
    SignalPath,
)


class TestEmitConfig:
    def test_recursively_emits_sequencer_configuration(self):
        config = SequencerConfigAttr(
            port_id="q0.readout",
            carrier_frequency=6.2e9,
            connections=[
                ConnectionAttr(DirectionKind.output, [0, 1]),
                ConnectionAttr(DirectionKind.input, [0]),
            ],
            output_path_connections=[
                OutputPathConnectionAttr(0, SignalPath.i),
                OutputPathConnectionAttr(1, SignalPath.q),
            ],
            acquisition_path_connections=[
                AcquisitionPathConnectionAttr(0, SignalPath.i),
            ],
            acquisition_enabled=True,
            disabled_outputs=[2],
            disabled_acquisition_paths=[SignalPath.q],
            acquisition_disabled=False,
            local_oscillator_id="lo0",
            enable_sync=True,
            nco=NcoConfigAttr(frequency=200e6, prop_delay_comp_en=True),
            awg=AwgConfigAttr(gain_path0=0.8, offset_path1=-0.01),
            unweighted_acquire=UnweightedAcquireConfigAttr(16),
            acquire=AcquireConfigAttr(auto_bin_incr_en=True, demod_en_acq=True),
            thresholded_acquire=ThresholdedAcquireConfigAttr(rotation=45.0, threshold=0.25),
        )

        emitted = emit_config(config)

        assert emitted["connections"] == [
            {"direction": "out", "port_ids": [0, 1]},
            {"direction": "in", "port_ids": [0]},
        ]
        assert emitted["output_path_connections"] == [
            {"output_id": 0, "path": "I"},
            {"output_id": 1, "path": "Q"},
        ]
        assert emitted["acquisition_path_connections"] == [{"input_id": 0, "path": "I"}]
        assert emitted["nco"] == {
            "frequency": 200e6,
            "phase_offs": None,
            "prop_delay_comp": None,
            "prop_delay_comp_en": True,
        }
        assert emitted["awg"]["offset_path1"] == -0.01
        assert emitted["unweighted_acquire"] == {"integration_length": 16}
        assert emitted["acquire"]["demod_en_acq"] is True
        assert emitted["thresholded_acquire"]["rotation"] == 45.0

    def test_emits_current_module_schema_and_canonical_lane_order(self):
        config = ModuleConfigAttr(
            2,
            "cluster0",
            QbloxModuleKind.qrm,
            outputs=[OutputConfigAttr(1), OutputConfigAttr(0)],
            inputs=[
                InputConfigAttr(
                    1,
                    InputSignalConfigAttr(gain=6.0),
                    ScopeAcquireConfigAttr(sequencer_select=2, enable_average_mode=False),
                ),
                InputConfigAttr(0),
            ],
            local_oscillators=[
                LocalOscillatorConfigAttr("lo1", 5_000_000_000),
                LocalOscillatorConfigAttr("lo0", 4_000_000_000),
            ],
        )

        emitted = emit_config(config)

        assert emitted["instrument_id"] == "cluster0"
        assert emitted["slot_idx"] == 2
        assert emitted["kind"] == "qrm"
        assert [output["output_id"] for output in emitted["outputs"]] == [0, 1]
        assert [module_input["input_id"] for module_input in emitted["inputs"]] == [0, 1]
        assert [
            oscillator["oscillator_id"] for oscillator in emitted["local_oscillators"]
        ] == ["lo0", "lo1"]
        assert emitted["inputs"][1]["scope_acquire"] == {
            "sequencer_select": 2,
            "enable_average_mode": False,
        }

    @pytest.mark.parametrize(
        "field",
        [
            "connections",
            "output_path_connections",
            "acquisition_path_connections",
            "disabled_outputs",
            "disabled_acquisition_paths",
        ],
    )
    def test_rejects_unresolved_sequencer_configuration(self, field):
        resolved = {
            "connections": [],
            "output_path_connections": [],
            "acquisition_path_connections": [],
            "disabled_outputs": [],
            "disabled_acquisition_paths": [],
        }
        resolved[field] = None

        with pytest.raises(
            VerifyException,
            match=rf"SequencerConfigAttr\.{field} must be resolved before emission",
        ):
            emit_config(SequencerConfigAttr(**resolved))


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

    def test_rejects_non_sequence_ops(self):
        module = ModuleOp([NopOp(), SequenceOp("drive", [StopOp()])])
        with pytest.raises(TypeError, match="top-level SequenceOps"):
            emit_module(module)

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
