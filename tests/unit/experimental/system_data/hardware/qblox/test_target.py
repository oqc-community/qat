# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import subprocess
import sys
from dataclasses import replace

import pytest
from frozendict import frozendict

from qat.experimental.system_data.hardware.qblox.models import (
    QbloxModuleKind,
    QbloxModuleLocation,
)
from qat.experimental.system_data.hardware.qblox.target import (
    DEFAULT_QBLOX_TARGET,
    ModuleSpec,
    Q1SequencerFeature,
    Q1SequencerType,
    QbloxTargetDescription,
)


@pytest.mark.parametrize(
    ("kind", "sequencers", "outputs", "inputs", "readout"),
    [
        (QbloxModuleKind.qcm, 6, 4, 0, ()),
        (QbloxModuleKind.qcm_rf, 6, 2, 0, ()),
        (QbloxModuleKind.qrm, 6, 2, 2, tuple(range(6))),
        (QbloxModuleKind.qrm_rf, 6, 1, 1, tuple(range(6))),
        (QbloxModuleKind.qrc, 12, 6, 2, tuple(range(8))),
    ],
)
def test_module_queries_cover_all_kinds(kind, sequencers, outputs, inputs, readout):
    module_spec = DEFAULT_QBLOX_TARGET.module_spec(kind)

    assert module_spec.sequencer_count == sequencers
    assert module_spec.output_count == outputs
    assert module_spec.input_count == inputs
    assert module_spec.sequencer_indices(Q1SequencerType.readout) == readout
    if kind is QbloxModuleKind.qrc:
        assert module_spec.acquisition_memory_bins == 7_000_000


def test_routing_and_sequencer_types_come_from_target_description():
    assert DEFAULT_QBLOX_TARGET.output_sequencers(QbloxModuleKind.qrc, 2) == (
        0,
        4,
        8,
        9,
        10,
        11,
    )
    assert DEFAULT_QBLOX_TARGET.input_sequencers(QbloxModuleKind.qrc, 0) == tuple(range(8))
    readout = DEFAULT_QBLOX_TARGET.sequencer(QbloxModuleKind.qrc, 7)
    control = DEFAULT_QBLOX_TARGET.sequencer(QbloxModuleKind.qrc, 8)
    assert readout.sequencer_spec.type is Q1SequencerType.readout
    assert readout.sequencer_spec.supports(Q1SequencerFeature.awg)
    assert readout.sequencer_spec.supports(Q1SequencerFeature.acquisition)
    assert control.sequencer_spec.type is Q1SequencerType.control
    assert control.sequencer_spec.supports(Q1SequencerFeature.awg)
    assert not control.sequencer_spec.supports(Q1SequencerFeature.acquisition)


@pytest.mark.parametrize(
    ("kind", "is_rf"),
    [
        (QbloxModuleKind.qcm, False),
        (QbloxModuleKind.qcm_rf, True),
        (QbloxModuleKind.qrm, False),
        (QbloxModuleKind.qrm_rf, True),
        (QbloxModuleKind.qrc, True),
    ],
)
def test_rf_classification_matches_qblox_driver(kind, is_rf):
    assert DEFAULT_QBLOX_TARGET.module_spec(kind).is_rf is is_rf


def test_target_rejects_illegal_indices():
    with pytest.raises(ValueError, match="Sequencer index 12"):
        DEFAULT_QBLOX_TARGET.sequencer(QbloxModuleKind.qrc, 12)
    with pytest.raises(ValueError, match="Output channel out6"):
        DEFAULT_QBLOX_TARGET.output_sequencers(QbloxModuleKind.qrc, 6)
    with pytest.raises(ValueError, match="Input channel in2"):
        DEFAULT_QBLOX_TARGET.input_sequencers(QbloxModuleKind.qrc, 2)


def test_target_rejects_module_locations_outside_the_cluster():
    with pytest.raises(ValueError, match=r"module slot must be in \[1, 20\]"):
        DEFAULT_QBLOX_TARGET.validate_module_location(QbloxModuleLocation("cluster", 21))


@pytest.mark.parametrize(
    ("kind", "supports_mixer_correction"),
    [
        (QbloxModuleKind.qcm, True),
        (QbloxModuleKind.qcm_rf, True),
        (QbloxModuleKind.qrm, True),
        (QbloxModuleKind.qrm_rf, True),
        (QbloxModuleKind.qrc, False),
    ],
)
def test_module_and_sequencer_limits(kind, supports_mixer_correction):
    module_spec = DEFAULT_QBLOX_TARGET.module_spec(kind)
    control = DEFAULT_QBLOX_TARGET.sequencer_spec(Q1SequencerType.control)
    readout = DEFAULT_QBLOX_TARGET.sequencer_spec(Q1SequencerType.readout)

    assert module_spec.supports_mixer_correction is supports_mixer_correction
    assert (
        control.nco_min_frequency_hz,
        control.nco_max_frequency_hz,
    ) == (-500_000_000.0, 500_000_000.0)
    assert control.waveform_sample_capacity == 16_384
    assert readout.readout.weight_sample_capacity == 16_384


def test_target_import_does_not_load_xdsl():
    result = subprocess.run(  # noqa: S603 - interpreter and script are fixed test inputs
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import qat.experimental.system_data.hardware.qblox.target; "
                "assert not any(name == 'xdsl' or name.startswith('xdsl.') "
                "for name in sys.modules)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_target_requires_complete_matching_specifications():
    with pytest.raises(ValueError, match="every Q1 sequencer type"):
        QbloxTargetDescription(
            q1asm=DEFAULT_QBLOX_TARGET.q1asm,
            sequencer_specs=frozendict(
                {
                    Q1SequencerType.control: DEFAULT_QBLOX_TARGET.sequencer_spec(
                        Q1SequencerType.control
                    )
                }
            ),
            module_specs=DEFAULT_QBLOX_TARGET.module_specs,
        )


def test_readout_sequencer_uses_readout_specification():
    readout = replace(
        DEFAULT_QBLOX_TARGET.sequencer_spec(Q1SequencerType.readout),
        nco_min_frequency_hz=-400_000_000,
        nco_max_frequency_hz=400_000_000,
        waveform_sample_capacity=8192,
    )
    target = replace(
        DEFAULT_QBLOX_TARGET,
        sequencer_specs=frozendict(
            {
                **DEFAULT_QBLOX_TARGET.sequencer_specs,
                Q1SequencerType.readout: readout,
            }
        ),
    )

    sequencer = target.sequencer(QbloxModuleKind.qrm, 0)
    assert (
        sequencer.sequencer_spec.nco_min_frequency_hz,
        sequencer.sequencer_spec.nco_max_frequency_hz,
    ) == (-400_000_000.0, 400_000_000.0)
    assert sequencer.sequencer_spec.waveform_sample_capacity == 8192


def test_module_channel_map_is_immutable():
    module_spec = DEFAULT_QBLOX_TARGET.module_spec(QbloxModuleKind.qcm)

    with pytest.raises(TypeError):
        module_spec.output_channel_map[0] = ()


def test_module_rejects_control_sequencer_on_input_channel_map():
    with pytest.raises(ValueError, match="control sequencer"):
        ModuleSpec(
            kind=QbloxModuleKind.qcm,
            sequencers=(Q1SequencerType.control,),
            output_count=1,
            input_count=1,
            output_channel_map=frozendict({0: (0,)}),
            input_channel_map=frozendict({0: (0,)}),
            acquisition_memory_bins=1,
        )


def test_target_rejects_mutable_specification_maps():
    with pytest.raises(TypeError, match="immutable frozendict"):
        replace(DEFAULT_QBLOX_TARGET, module_specs=dict(DEFAULT_QBLOX_TARGET.module_specs))
