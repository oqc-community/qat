# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest
from xdsl.utils.exceptions import VerifyException

from qat.backend.qblox.execution import ModuleConfig
from qat.experimental.backend.qblox.translation import (
    translate_module_config,
    translate_package,
    translate_sequencer_config,
)
from qat.experimental.dialect.q1 import StopOp
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


def _sequencer_config(
    *,
    connections=(),
    acquisition_path_connections=(),
    disabled_acquisition_paths=(),
    acquisition_disabled=False,
    local_oscillator_id=None,
) -> SequencerConfigAttr:
    return SequencerConfigAttr(
        port_id="test",
        connections=list(connections),
        output_path_connections=[],
        acquisition_path_connections=list(acquisition_path_connections),
        disabled_outputs=[],
        disabled_acquisition_paths=list(disabled_acquisition_paths),
        acquisition_disabled=acquisition_disabled,
        local_oscillator_id=local_oscillator_id,
    )


def test_translates_complete_sequencer_configuration():
    config = SequencerConfigAttr(
        port_id="q0/measure",
        carrier_frequency=4_640_000_000,
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
        nco=NcoConfigAttr(frequency=240_000_000, phase_offs=0.5),
        awg=AwgConfigAttr(gain_path0=0.8, offset_path1=-0.01, mod_en=True),
        mixer=MixerCorrectionConfigAttr(phase_offset=0.25, gain_ratio=1.1),
        marker_switch=MarkerOverrideConfigAttr(marker_ovr_en=True, marker_ovr_value=3),
        unweighted_acquire=UnweightedAcquireConfigAttr(16),
        acquire=AcquireConfigAttr(auto_bin_incr_en=True, demod_en_acq=True),
        thresholded_acquire=ThresholdedAcquireConfigAttr(rotation=45.0, threshold=0.5),
    )

    translated = translate_sequencer_config(config)

    assert translated.sync_en is True
    assert translated.connection.bulk_value == ["out0_1", "in0"]
    assert translated.connection.out0 == "I"
    assert translated.connection.out1 == "Q"
    assert translated.connection.out2 == "off"
    assert translated.connection.acq_I == "in0"
    assert translated.connection.acq_Q == "off"
    assert translated.nco.freq == 240_000_000
    assert translated.awg.gain_path0 == 0.8
    assert translated.mixer.gain_ratio == 1.1
    assert translated.marker_ovr_value == 3
    assert translated.square_weight_acq.integration_length == 16
    assert translated.demod_en_acq is True
    assert translated.ttl_acq.auto_bin_incr_en is True
    assert translated.thresholded_acq.threshold == 0.5


@pytest.mark.parametrize(
    ("acquisition_paths", "disabled_paths", "acquisition_disabled", "expected"),
    [
        pytest.param(
            [AcquisitionPathConnectionAttr(1, SignalPath.iq)],
            [],
            False,
            ("in1", "in1"),
            id="combined-path",
        ),
        pytest.param(
            [AcquisitionPathConnectionAttr(1, SignalPath.q)],
            [],
            False,
            (None, "in1"),
            id="q-path",
        ),
        pytest.param(
            [],
            [SignalPath.iq],
            False,
            ("off", "off"),
            id="combined-path-disabled",
        ),
        pytest.param(
            [],
            [SignalPath.i],
            False,
            ("off", None),
            id="i-path-disabled",
        ),
        pytest.param(
            [],
            [],
            True,
            ("off", "off"),
            id="acquisition-disabled",
        ),
    ],
)
def test_translates_combined_and_disabled_acquisition_connections(
    acquisition_paths,
    disabled_paths,
    acquisition_disabled,
    expected,
):
    config = _sequencer_config(
        acquisition_path_connections=acquisition_paths,
        disabled_acquisition_paths=disabled_paths,
        acquisition_disabled=acquisition_disabled,
    )

    connection = translate_sequencer_config(config).connection

    assert (connection.acq_I, connection.acq_Q) == expected


def test_translates_lane_oriented_module_configuration():
    module_config = ModuleConfigAttr(
        4,
        "cluster0",
        QbloxModuleKind.qrm_rf,
        outputs=[
            OutputConfigAttr(
                0,
                RealTimePredistortionConfigAttr(fir_out="bypassed"),
                OutputSignalConfigAttr(
                    attenuation=12.0,
                    offset_path_0=0.1,
                    offset_path_1=-0.1,
                ),
            )
        ],
        inputs=[
            InputConfigAttr(
                0,
                InputSignalConfigAttr(attenuation=6.0, gain=4.0),
                ScopeAcquireConfigAttr(sequencer_select=2, enable_average_mode=True),
            )
        ],
        local_oscillators=[LocalOscillatorConfigAttr("lo0", 4_400_000_000, enable=True)],
    )
    sequencer_config = SequencerConfigAttr(
        port_id="q0/measure",
        connections=[
            ConnectionAttr(DirectionKind.output, [0]),
            ConnectionAttr(DirectionKind.input, [0]),
        ],
        output_path_connections=[],
        acquisition_path_connections=[],
        disabled_outputs=[],
        disabled_acquisition_paths=[],
        local_oscillator_id="lo0",
    )

    translated = translate_module_config(module_config, [sequencer_config])

    assert translated.attenuation.out0 == 12.0
    assert translated.attenuation.in0 == 6.0
    assert translated.offset.out0_path0 == 0.1
    assert translated.offset.out0_path1 == -0.1
    assert translated.gain.in0 == 4
    assert translated.scope_acq.sequencer_select == 2
    assert translated.scope_acq.avg_mode_en_path0 is True
    assert translated.fir.out0 == "bypassed"
    assert translated.lo.out0_in0_freq == 4_400_000_000
    assert translated.lo.out0_in0_en is True


def test_translates_qrc_paired_lane_local_oscillator():
    module_config = ModuleConfigAttr(
        2,
        "cluster0",
        QbloxModuleKind.qrc,
        local_oscillators=[LocalOscillatorConfigAttr("lo1", 5_400_000_000, enable=None)],
    )
    sequencer_config = _sequencer_config(
        connections=[ConnectionAttr(DirectionKind.input, [1])],
        local_oscillator_id="lo1",
    )

    translated = translate_module_config(module_config, [sequencer_config])

    assert translated.lo.out1_in1_freq == 5_400_000_000


def test_rejects_conflicting_flattened_scope_configuration():
    module_config = ModuleConfigAttr(
        2,
        "cluster0",
        QbloxModuleKind.qrm,
        inputs=[
            InputConfigAttr(0, scope_acquire=ScopeAcquireConfigAttr(sequencer_select=1)),
            InputConfigAttr(1, scope_acquire=ScopeAcquireConfigAttr(sequencer_select=2)),
        ],
    )

    with pytest.raises(ValueError, match="Conflicting values.*sequencer_select"):
        translate_module_config(module_config, [])


def test_rejects_module_configuration_that_legacy_model_cannot_represent():
    module_config = ModuleConfigAttr(
        2,
        "cluster0",
        QbloxModuleKind.qrm,
        inputs=[InputConfigAttr(0, InputSignalConfigAttr(gain=1.5))],
    )

    with pytest.raises(ValueError, match="cannot be represented"):
        translate_module_config(module_config, [])


def test_rejects_qrc_path_offset_absent_from_legacy_model():
    module_config = ModuleConfigAttr(
        2,
        "cluster0",
        QbloxModuleKind.qrc,
        outputs=[
            OutputConfigAttr(
                2,
                output_signal=OutputSignalConfigAttr(offset_path_0=0.5),
            )
        ],
    )

    with pytest.raises(ValueError, match="out2_path0"):
        translate_module_config(module_config, [])


def test_rejects_sequencer_reference_to_missing_local_oscillator():
    module_config = ModuleConfigAttr(4, "cluster0", QbloxModuleKind.qcm_rf)
    sequencer_config = _sequencer_config(
        connections=[ConnectionAttr(DirectionKind.output, [0])],
        local_oscillator_id="missing",
    )

    with pytest.raises(ValueError, match="missing local oscillator 'missing'"):
        translate_module_config(module_config, [sequencer_config])


def test_rejects_local_oscillator_on_baseband_module():
    with pytest.raises(VerifyException, match="does not support a local oscillator"):
        ModuleConfigAttr(
            2,
            "cluster0",
            QbloxModuleKind.qcm,
            local_oscillators=[LocalOscillatorConfigAttr("lo0", 4_400_000_000)],
        )


def test_rejects_local_oscillator_enable_without_legacy_field():
    module_config = ModuleConfigAttr(
        2,
        "cluster0",
        QbloxModuleKind.qrc,
        local_oscillators=[LocalOscillatorConfigAttr("lo0", 4_400_000_000, enable=True)],
    )
    sequencer_config = _sequencer_config(
        connections=[ConnectionAttr(DirectionKind.output, [2])],
        local_oscillator_id="lo0",
    )

    with pytest.raises(ValueError, match="enable state cannot be represented"):
        translate_module_config(module_config, [sequencer_config])


def test_rejects_package_translation_without_allocation():
    sequence = SequenceOp("drive", [StopOp()], port_id="drive")

    with pytest.raises(ValueError, match="fully configured and allocated"):
        translate_package(sequence, ModuleConfig())
