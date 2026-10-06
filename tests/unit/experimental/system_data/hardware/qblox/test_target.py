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
    SignalPath,
)
from qat.experimental.system_data.hardware.qblox.target import (
    DEFAULT_QBLOX_TARGET,
    AcquisitionConnectionMode,
    LocalOscillatorSpec,
    ModuleSpec,
    Q1SequencerFeature,
    Q1SequencerSpec,
    Q1SequencerType,
    QbloxTargetDescription,
    ReadoutSpec,
)


@pytest.mark.parametrize(
    ("kind", "output_paths", "acquisition_mode", "acquisition_paths"),
    [
        (
            QbloxModuleKind.qcm,
            frozenset({SignalPath.i, SignalPath.q}),
            AcquisitionConnectionMode.none,
            frozenset(),
        ),
        (
            QbloxModuleKind.qrm,
            frozenset({SignalPath.i, SignalPath.q}),
            AcquisitionConnectionMode.components,
            frozenset({SignalPath.i, SignalPath.q, SignalPath.iq}),
        ),
        (
            QbloxModuleKind.qcm_rf,
            frozenset({SignalPath.iq}),
            AcquisitionConnectionMode.none,
            frozenset(),
        ),
        (
            QbloxModuleKind.qrm_rf,
            frozenset({SignalPath.iq}),
            AcquisitionConnectionMode.combined,
            frozenset({SignalPath.iq}),
        ),
        (
            QbloxModuleKind.qrc,
            frozenset({SignalPath.iq}),
            AcquisitionConnectionMode.combined,
            frozenset({SignalPath.iq}),
        ),
    ],
)
def test_module_connection_capabilities_match_qblox_api(
    kind, output_paths, acquisition_mode, acquisition_paths
):
    module_spec = DEFAULT_QBLOX_TARGET.module_spec(kind)

    assert module_spec.output_path_components == output_paths
    assert module_spec.acquisition_connection_mode is acquisition_mode
    assert module_spec.acquisition_path_components == acquisition_paths


@pytest.mark.parametrize(
    ("kind", "path", "components"),
    [
        (QbloxModuleKind.qrm, SignalPath.i, (SignalPath.i,)),
        (QbloxModuleKind.qrm, SignalPath.q, (SignalPath.q,)),
        (QbloxModuleKind.qrm, SignalPath.iq, (SignalPath.i, SignalPath.q)),
        (QbloxModuleKind.qrm_rf, SignalPath.iq, (SignalPath.iq,)),
        (QbloxModuleKind.qrc, SignalPath.iq, (SignalPath.iq,)),
    ],
)
def test_acquisition_paths_expand_to_module_api_components(kind, path, components):
    assert DEFAULT_QBLOX_TARGET.module_spec(kind).acquisition_components(path) == components


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
    assert DEFAULT_QBLOX_TARGET.is_readout_sequencer(QbloxModuleKind.qrc, 7)
    assert not DEFAULT_QBLOX_TARGET.is_readout_sequencer(QbloxModuleKind.qrc, 8)
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


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (QbloxModuleKind.qcm, None),
        (
            QbloxModuleKind.qcm_rf,
            LocalOscillatorSpec(2_000_000_000, 18_000_000_000),
        ),
        (QbloxModuleKind.qrm, None),
        (
            QbloxModuleKind.qrm_rf,
            LocalOscillatorSpec(2_000_000_000, 18_000_000_000),
        ),
        (
            QbloxModuleKind.qrc,
            LocalOscillatorSpec(500_000_000, 10_100_000_000, 100_000_000),
        ),
    ],
)
def test_module_local_oscillator_limits_match_qblox_driver(kind, expected):
    assert DEFAULT_QBLOX_TARGET.module_spec(kind).local_oscillator == expected


@pytest.mark.parametrize(
    "frequency",
    [500_000_000, 10_100_000_000],
)
def test_qrc_local_oscillator_accepts_documented_boundaries(frequency):
    oscillator = DEFAULT_QBLOX_TARGET.module_spec(QbloxModuleKind.qrc).local_oscillator

    assert oscillator is not None
    assert oscillator.supports(frequency)
    assert not oscillator.supports(frequency + 50_000_000)


@pytest.mark.parametrize(
    ("minimum", "maximum", "step", "message"),
    [
        (-1, 1, 1, "minimum frequency must be non-negative"),
        (2, 1, 1, "maximum frequency must not be below its minimum"),
        (0, 1, 0, "frequency step must be positive"),
    ],
)
def test_local_oscillator_rejects_invalid_limits(minimum, maximum, step, message):
    with pytest.raises(ValueError, match=message):
        LocalOscillatorSpec(minimum, maximum, step)


def test_module_rf_classification_requires_a_local_oscillator_specification():
    qcm = DEFAULT_QBLOX_TARGET.module_spec(QbloxModuleKind.qcm)

    with pytest.raises(ValueError, match="RF module classification"):
        replace(qcm, is_rf=True)


@pytest.mark.parametrize(
    ("type_", "readout"),
    [
        (
            Q1SequencerType.control,
            DEFAULT_QBLOX_TARGET.sequencer_spec(Q1SequencerType.readout).readout,
        ),
        (Q1SequencerType.readout, None),
    ],
)
def test_sequencer_spec_requires_readout_limits_only_for_readout(type_, readout):
    with pytest.raises(ValueError, match="require readout-path limits"):
        Q1SequencerSpec(type_, instruction_capacity=1, readout=readout)


def _qcm_spec(**changes):
    return replace(DEFAULT_QBLOX_TARGET.module_spec(QbloxModuleKind.qcm), **changes)


@pytest.mark.parametrize(
    ("changes", "exception", "message"),
    [
        (
            {"sequencers": [Q1SequencerType.control]},
            TypeError,
            "sequencers must be an immutable tuple",
        ),
        (
            {"output_channel_map": {0: (0,), 1: (0,), 2: (0,), 3: (0,)}},
            TypeError,
            "channel maps must be immutable frozendict",
        ),
        (
            {"output_path_components": {SignalPath.i}},
            TypeError,
            "output path components must be an immutable frozenset",
        ),
        (
            {"output_channel_map": frozendict({0: [0], 1: (0,), 2: (0,), 3: (0,)})},
            TypeError,
            "channel-map sequencer indices must be tuples",
        ),
        (
            {"output_channel_map": frozendict()},
            ValueError,
            "output channel map is incomplete",
        ),
        (
            {"input_count": 1},
            ValueError,
            "input channel map is incomplete",
        ),
        (
            {"output_channel_map": frozendict({0: (6,), 1: (0,), 2: (0,), 3: (0,)})},
            ValueError,
            "channel map references an invalid sequencer",
        ),
        (
            {"acquisition_memory_bins": 1},
            ValueError,
            "acquisition memory must match its sequencer types",
        ),
        (
            {"acquisition_connection_mode": AcquisitionConnectionMode.components},
            ValueError,
            "acquisition connection mode must match its sequencer types",
        ),
    ],
)
def test_module_spec_rejects_inconsistent_capabilities(changes, exception, message):
    with pytest.raises(exception, match=message):
        _qcm_spec(**changes)


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


def test_qrc_excludes_unsupported_runtime_configuration():
    module = DEFAULT_QBLOX_TARGET.module_spec(QbloxModuleKind.qrc)

    assert not module.supports_local_oscillator_enable
    assert not module.supports_awg_modulation
    assert not module.supports_acquisition_demodulation


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


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {
                "sequencer_specs": frozendict(
                    {
                        Q1SequencerType.control: DEFAULT_QBLOX_TARGET.sequencer_spec(
                            Q1SequencerType.readout
                        ),
                        Q1SequencerType.readout: DEFAULT_QBLOX_TARGET.sequencer_spec(
                            Q1SequencerType.control
                        ),
                    }
                )
            },
            "Sequencer specification keys must match their types",
        ),
        (
            {
                "module_specs": frozendict(
                    {
                        kind: spec
                        for kind, spec in DEFAULT_QBLOX_TARGET.module_specs.items()
                        if kind is not QbloxModuleKind.qrc
                    }
                )
            },
            "Target description must define every Qblox module kind",
        ),
        (
            {
                "module_specs": frozendict(
                    {
                        **DEFAULT_QBLOX_TARGET.module_specs,
                        QbloxModuleKind.qcm: DEFAULT_QBLOX_TARGET.module_spec(
                            QbloxModuleKind.qrm
                        ),
                    }
                )
            },
            "Module specification keys must match their kinds",
        ),
        (
            {"min_module_slot": 0},
            "Target module slot range is invalid",
        ),
    ],
)
def test_target_rejects_inconsistent_capability_tables(changes, message):
    with pytest.raises(ValueError, match=message):
        replace(DEFAULT_QBLOX_TARGET, **changes)


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


@pytest.mark.parametrize(
    ("sequencer_type", "readout"),
    [
        (Q1SequencerType.control, ReadoutSpec()),
        (Q1SequencerType.readout, None),
    ],
)
def test_sequencer_spec_requires_readout_limits_to_match_type(sequencer_type, readout):
    with pytest.raises(ValueError, match="require readout-path limits"):
        replace(DEFAULT_QBLOX_TARGET.sequencer_spec(sequencer_type), readout=readout)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"sequencers": [Q1SequencerType.control]}, "sequencers must be an immutable"),
        ({"output_channel_map": {0: (0,)}}, "channel maps must be immutable"),
        (
            {"output_channel_map": frozendict({0: [0], 1: (), 2: (), 3: ()})},
            "sequencer indices must be tuples",
        ),
        (
            {"output_channel_map": frozendict({1: (), 2: (), 3: ()})},
            "output channel map is incomplete",
        ),
        (
            {"output_channel_map": frozendict({0: (6,), 1: (), 2: (), 3: ()})},
            "references an invalid sequencer",
        ),
        ({"acquisition_memory_bins": 1}, "acquisition memory must match"),
        ({"is_rf": True}, "RF module classification"),
    ],
)
def test_module_spec_rejects_invalid_configuration(updates, message):
    module_spec = DEFAULT_QBLOX_TARGET.module_spec(QbloxModuleKind.qcm)

    with pytest.raises((TypeError, ValueError), match=message):
        replace(module_spec, **updates)


def test_module_spec_requires_input_map_for_declared_inputs():
    module_spec = DEFAULT_QBLOX_TARGET.module_spec(QbloxModuleKind.qrm)

    with pytest.raises(ValueError, match="input channel map is incomplete"):
        replace(module_spec, input_channel_map=frozendict({0: ()}))


def test_rf_module_requires_local_oscillator_specification():
    module_spec = DEFAULT_QBLOX_TARGET.module_spec(QbloxModuleKind.qcm_rf)

    with pytest.raises(ValueError, match="RF module classification"):
        replace(module_spec, local_oscillator=None)


def test_target_rejects_mutable_specification_maps():
    with pytest.raises(TypeError, match="immutable frozendict"):
        replace(DEFAULT_QBLOX_TARGET, module_specs=dict(DEFAULT_QBLOX_TARGET.module_specs))


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"min_module_slot": 0}, "slot range is invalid"),
        ({"min_module_slot": 5, "max_module_slot": 4}, "slot range is invalid"),
        (
            {
                "sequencer_specs": frozendict(
                    {
                        Q1SequencerType.control: DEFAULT_QBLOX_TARGET.sequencer_spec(
                            Q1SequencerType.readout
                        ),
                        Q1SequencerType.readout: DEFAULT_QBLOX_TARGET.sequencer_spec(
                            Q1SequencerType.control
                        ),
                    }
                )
            },
            "keys must match their types",
        ),
        (
            {
                "module_specs": frozendict(
                    {
                        **DEFAULT_QBLOX_TARGET.module_specs,
                        QbloxModuleKind.qcm: DEFAULT_QBLOX_TARGET.module_spec(
                            QbloxModuleKind.qcm_rf
                        ),
                    }
                )
            },
            "keys must match their kinds",
        ),
    ],
)
def test_target_rejects_inconsistent_specifications(updates, message):
    with pytest.raises(ValueError, match=message):
        replace(DEFAULT_QBLOX_TARGET, **updates)
