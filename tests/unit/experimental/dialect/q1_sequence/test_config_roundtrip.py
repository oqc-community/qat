# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from io import StringIO

from xdsl.context import Context
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.parser import Parser
from xdsl.printer import Printer

from qat.experimental.dialect.q1 import Q1, StopOp
from qat.experimental.dialect.q1_sequence import Q1_sequence
from qat.experimental.dialect.q1_sequence.ir.attrs import (
    AcquireConfigAttr,
    AcquisitionPathConnectionAttr,
    AwgConfigAttr,
    ConnectionAttr,
    InputConfigAttr,
    InputSignalConfigAttr,
    LocalOscillatorConfigAttr,
    MarkerOverrideConfigAttr,
    MixerCorrectionConfigAttr,
    ModuleConfigAttr,
    NcoConfigAttr,
    OutputConfigAttr,
    OutputPathConnectionAttr,
    OutputSignalConfigAttr,
    RealTimePredistortionConfigAttr,
    ScopeAcquireConfigAttr,
    SequencerConfigAttr,
    ThresholdedAcquireConfigAttr,
    UnweightedAcquireConfigAttr,
)
from qat.experimental.dialect.q1_sequence.ir.ops import SequenceOp
from qat.experimental.system_data.hardware.qblox.models import (
    DirectionKind,
    QbloxModuleKind,
    SignalPath,
)


def test_physical_configuration_round_trips():
    module_config = ModuleConfigAttr(
        4,
        "cluster0",
        QbloxModuleKind.qrm_rf,
        [
            OutputConfigAttr(
                0,
                RealTimePredistortionConfigAttr(fir_out="bypassed"),
                OutputSignalConfigAttr(attenuation=12.0),
            )
        ],
        [
            InputConfigAttr(
                0,
                InputSignalConfigAttr(gain=6.0),
                ScopeAcquireConfigAttr(enable_average_mode=True),
            )
        ],
        [LocalOscillatorConfigAttr("lo0", 4_400_000_000, enable=True)],
    )
    sequencer_config = SequencerConfigAttr(
        port_id="q0/measure",
        carrier_frequency=4_640_000_000.0,
        connections=[
            ConnectionAttr(DirectionKind.output, [0]),
            ConnectionAttr(DirectionKind.input, [0]),
        ],
        output_path_connections=[OutputPathConnectionAttr(0, SignalPath.iq)],
        acquisition_path_connections=[AcquisitionPathConnectionAttr(0, SignalPath.iq)],
        acquisition_enabled=True,
        disabled_outputs=[],
        disabled_acquisition_paths=[],
        acquisition_disabled=False,
        local_oscillator_id="lo0",
        enable_sync=True,
        nco=NcoConfigAttr(frequency=240_000_000.0, phase_offs=0.0),
        awg=AwgConfigAttr(mod_en=True),
        mixer=MixerCorrectionConfigAttr(phase_offset=0.5, gain_ratio=1.0),
        marker_switch=MarkerOverrideConfigAttr(marker_ovr_value=3),
        unweighted_acquire=UnweightedAcquireConfigAttr(1024),
        acquire=AcquireConfigAttr(demod_en_acq=True),
        thresholded_acquire=ThresholdedAcquireConfigAttr(threshold=0.25),
    )
    sequence = SequenceOp(
        "q0_measure",
        [StopOp()],
        port_id="q0/measure",
        instrument_id="cluster0",
        slot_idx=4,
        seq_idx=2,
        sequencer_config=sequencer_config,
        module_config=module_config,
    )
    module = ModuleOp([sequence])

    printed = StringIO()
    Printer(stream=printed).print_op(module)

    context = Context()
    context.load_dialect(Builtin)
    context.load_dialect(Q1)
    context.load_dialect(Q1_sequence)
    parsed = Parser(context, printed.getvalue()).parse_op()
    parsed.verify()

    reprinted = StringIO()
    Printer(stream=reprinted).print_op(parsed)
    assert reprinted.getvalue() == printed.getvalue()


def test_absent_configuration_round_trips_as_none():
    sequence = SequenceOp("q0_drive", [StopOp()])
    module = ModuleOp([sequence])

    printed = StringIO()
    Printer(stream=printed).print_op(module)

    context = Context()
    context.load_dialect(Builtin)
    context.load_dialect(Q1)
    context.load_dialect(Q1_sequence)
    parsed = Parser(context, printed.getvalue()).parse_op()
    parsed.verify()

    reprinted = StringIO()
    Printer(stream=reprinted).print_op(parsed)
    assert reprinted.getvalue() == printed.getvalue()
