# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import re

import pytest
from xdsl.dialects.builtin import NoneAttr

from qat.experimental.conversion.pulse_to_q1.qblox_configuration.resolution import (
    resolve_sequencer_bindings,
)
from qat.experimental.dialect.q1_sequence.ir.attrs import (
    AcquireConfigAttr,
    AcquisitionPathConnectionAttr,
    AwgConfigAttr,
    ConnectionAttr,
    MarkerOverrideConfigAttr,
    MixerCorrectionConfigAttr,
    OutputPathConnectionAttr,
    ThresholdedAcquireConfigAttr,
    UnweightedAcquireConfigAttr,
)
from qat.experimental.system_data.hardware.qblox.configuration import (
    AcquisitionPathConnection,
    OutputPathConnection,
    PortConnection,
    QbloxSequencerConfiguration,
    SequencerConnection,
    supplied_configurations,
)
from qat.experimental.system_data.hardware.qblox.models import (
    DirectionKind,
    QbloxModuleKind,
    SignalPath,
)
from qat.experimental.system_data.hardware.qblox.view import QbloxHardwareView

from tests.unit.experimental.conversion.pulse_to_q1.qblox_configuration.helpers import (
    canonical_data,
    sequencer,
    supplied,
)


def _resolve(configurations, **kwargs):
    data = canonical_data(configurations=configurations, **kwargs)
    return resolve_sequencer_bindings(
        QbloxHardwareView.derive(data), supplied_configurations(data)
    )


def test_absent_source_values_stay_absent():
    binding = _resolve([supplied([sequencer(0)])])["port-0-channel-0"]

    config = binding.sequencer_config
    assert isinstance(config.enable_sync, NoneAttr)
    assert isinstance(config.awg, NoneAttr)
    assert isinstance(config.marker_switch, NoneAttr)
    assert isinstance(config.unweighted_acquire, NoneAttr)
    assert isinstance(config.acquire, NoneAttr)
    assert isinstance(config.thresholded_acquire, NoneAttr)
    output = binding.module_config.outputs.data[0]
    assert isinstance(output.output_signal, NoneAttr)
    assert isinstance(output.pulse_shaping, NoneAttr)


def test_nco_frequency_is_the_carrier_minus_the_oscillator():
    binding = _resolve([supplied([sequencer(0)])])["port-0-channel-0"]

    assert binding.sequencer_config.nco.frequency.value.data == 200_000_000.0
    assert binding.sequencer_config.carrier_frequency.value.data == 4_200_000_000.0


def test_supplied_nco_frequency_must_match_the_calibration():
    with pytest.raises(ValueError, match="supplies NCO frequency 1.0 Hz"):
        _resolve([supplied([sequencer(0, values={"nco": {"freq": 1.0}})])])


def test_supplied_sequencer_values_reach_their_q1_attributes():
    binding = _resolve(
        [
            supplied(
                [
                    sequencer(
                        0,
                        values={
                            "sync_en": True,
                            "marker_ovr_en": True,
                            "marker_ovr_value": 3,
                            "demod_en_acq": True,
                            "nco": {"prop_delay_comp": 12, "prop_delay_comp_en": True},
                            "awg": {"gain_path0": 0.5, "mod_en": True},
                            "square_weight_acq": {"integration_length": 1024},
                            "thresholded_acq": {"rotation": 30.0, "threshold": 0.25},
                            "ttl_acq": {"auto_bin_incr_en": True},
                        },
                    )
                ]
            )
        ],
        kind=QbloxModuleKind.qrm_rf,
    )["port-0-channel-0"]

    config = binding.sequencer_config
    assert bool(config.enable_sync.value.data)
    assert config.nco.prop_delay_comp.data == 12
    assert config.awg == AwgConfigAttr(gain_path0=0.5, mod_en=True)
    assert config.marker_switch == MarkerOverrideConfigAttr(
        marker_ovr_en=True, marker_ovr_value=3
    )
    assert config.unweighted_acquire == UnweightedAcquireConfigAttr(1024)
    assert config.acquire == AcquireConfigAttr(auto_bin_incr_en=True, demod_en_acq=True)
    assert config.thresholded_acquire == ThresholdedAcquireConfigAttr(
        rotation=30.0, threshold=0.25
    )


def test_supplied_routing_is_preserved_in_full():
    connection = SequencerConnection(
        connections=(
            PortConnection(direction=DirectionKind.output, port_ids=(0,)),
            PortConnection(direction=DirectionKind.input, port_ids=(0,)),
        ),
        output_path_connections=(OutputPathConnection(output_id=0, path=SignalPath.i),),
        acquisition_path_connections=(
            AcquisitionPathConnection(input_id=0, path=SignalPath.q),
        ),
        acquisition_enabled=True,
        disabled_acquisition_paths=frozenset({SignalPath.i}),
    )
    binding = _resolve(
        [supplied([QbloxSequencerConfiguration(index=0, connection=connection)])],
        kind=QbloxModuleKind.qrm,
    )["port-0-channel-0"]

    config = binding.sequencer_config
    assert list(config.connections) == [
        ConnectionAttr(DirectionKind.output, [0]),
        ConnectionAttr(DirectionKind.input, [0]),
    ]
    assert list(config.output_path_connections) == [
        OutputPathConnectionAttr(0, SignalPath.i)
    ]
    assert list(config.acquisition_path_connections) == [
        AcquisitionPathConnectionAttr(0, SignalPath.q)
    ]
    assert bool(config.acquisition_enabled.value.data)
    assert [path.data for path in config.disabled_acquisition_paths] == [SignalPath.i]


def test_module_lanes_follow_the_allocated_sequencer_connections():
    binding = _resolve(
        [
            supplied(
                [sequencer(0, outputs=[0], inputs=[0])],
                module_values={
                    "attenuation": {"out0": 6, "in0": 3},
                    "offset": {"out0_path0": 0.1, "out0_path1": 0.2, "in0": 0.4},
                    "gain": {"in0": 2},
                    "scope_acq": {"sequencer_select": 0, "avg_mode_en_path0": True},
                },
            )
        ],
        kind=QbloxModuleKind.qrm_rf,
    )["port-0-channel-0"]

    module = binding.module_config
    output = module.outputs.data[0]
    assert output.output_id.data == 0
    assert output.output_signal.attenuation.value.data == 6.0
    assert output.output_signal.offset_path_0.value.data == 0.1
    assert isinstance(output.output_signal.offset, NoneAttr)
    module_input = module.inputs.data[0]
    assert module_input.input_signal.attenuation.value.data == 3.0
    assert module_input.input_signal.gain.value.data == 2.0
    assert module_input.input_signal.offset.value.data == 0.4
    assert module_input.scope_acquire.sequencer_select.data == 0
    assert bool(module_input.scope_acquire.enable_average_mode.value.data)


def test_scope_selection_preserves_the_selected_sequencer_index():
    binding = _resolve(
        [
            supplied(
                [sequencer(0, outputs=[0], inputs=[0])],
                module_values={"scope_acq": {"sequencer_select": 4}},
            )
        ],
        kind=QbloxModuleKind.qrm_rf,
    )["port-0-channel-0"]

    scope = binding.module_config.inputs.data[0].scope_acquire
    assert scope.sequencer_select.data == 4


def test_scope_selection_is_shared_faithfully_by_every_binding_of_a_module():
    bindings = _resolve(
        [
            supplied(
                [sequencer(1, outputs=[0], inputs=[0]), sequencer(3, outputs=[0])],
                module_values={"scope_acq": {"sequencer_select": 3}},
            )
        ],
        kind=QbloxModuleKind.qrm_rf,
        channels_per_port=2,
    )

    for binding in bindings.values():
        scope = binding.module_config.inputs.data[0].scope_acquire
        assert scope.sequencer_select.data == 3


@pytest.mark.parametrize(
    ("kind", "selected"),
    [
        (QbloxModuleKind.qrm_rf, 9),
        pytest.param(QbloxModuleKind.qrc, 8, id="qrc-control-sequencer"),
        pytest.param(QbloxModuleKind.qrc, 11, id="qrc-last-control-sequencer"),
    ],
)
def test_a_scope_selection_naming_a_non_acquiring_sequencer_is_rejected(kind, selected):
    """Only the acquisition-capable sequencers can trigger a scope acquisition."""

    with pytest.raises(ValueError, match="not one of the acquisition-capable sequencers"):
        _resolve(
            [
                supplied(
                    [sequencer(0, outputs=[0], inputs=[0])],
                    module_values={"scope_acq": {"sequencer_select": selected}},
                )
            ],
            kind=kind,
            oscillator_frequency=None if kind is QbloxModuleKind.qrc else 4_000_000_000,
            carrier_frequency=(
                200_000_000 if kind is QbloxModuleKind.qrc else 4_200_000_000
            ),
        )


def test_scope_acquisition_is_rejected_on_a_module_without_an_acquisition_path():
    with pytest.raises(ValueError, match="no acquisition path to apply"):
        _resolve(
            [
                supplied(
                    [sequencer(0)],
                    module_values={"scope_acq": {"sequencer_select": 0}},
                )
            ],
            kind=QbloxModuleKind.qcm_rf,
        )


def test_local_oscillator_enable_follows_the_lanes_it_feeds():
    binding = _resolve(
        [
            supplied(
                [sequencer(0, outputs=[0], inputs=[0])],
                module_values={"lo": {"out0_in0_en": True, "out0_in0_freq": 4.0e9}},
            )
        ],
        kind=QbloxModuleKind.qrm_rf,
    )["port-0-channel-0"]

    oscillator = binding.module_config.local_oscillators.data[0]
    assert oscillator.oscillator_id.data == "lo-0"
    assert oscillator.frequency.data == 4_000_000_000
    assert bool(oscillator.enable.value.data)


def test_supplied_oscillator_frequency_must_match_the_calibration():
    with pytest.raises(ValueError, match="supplied at 5000000000.0 Hz"):
        _resolve(
            [supplied([sequencer(0)], module_values={"lo": {"out0_freq": 5.0e9}})],
        )


def test_mixer_correction_comes_from_the_channel_calibration():
    binding = _resolve([supplied([sequencer(0)])])["port-0-channel-0"]

    assert binding.sequencer_config.mixer == MixerCorrectionConfigAttr(
        phase_offset=0.0, gain_ratio=0.9
    )


def test_mixer_correction_is_absent_where_the_module_has_no_mixer():
    binding = _resolve(
        [supplied([sequencer(0)])],
        kind=QbloxModuleKind.qrc,
        oscillator_frequency=None,
        carrier_frequency=200_000_000,
    )["port-0-channel-0"]

    assert isinstance(binding.sequencer_config.mixer, NoneAttr)


def test_supplied_mixer_correction_must_match_the_calibration():
    with pytest.raises(ValueError, match="supplies mixer gain_ratio 0.5"):
        _resolve([supplied([sequencer(0, values={"mixer": {"gain_ratio": 0.5}})])])


@pytest.mark.parametrize(
    ("kind", "outputs", "inputs", "index", "expected"),
    [
        (QbloxModuleKind.qcm_rf, [2], [], 0, "routes to output 2, absent on qcm_rf"),
        (QbloxModuleKind.qcm_rf, [0], [0], 0, "routes from input 0, absent on qcm_rf"),
        (QbloxModuleKind.qrc, [2], [], 1, "cannot drive output 2 on qrc"),
        (QbloxModuleKind.qrc, [2], [0], 8, "cannot read input 0 on qrc"),
        (QbloxModuleKind.qcm_rf, [0], [], 6, "is outside the 6 sequencers"),
    ],
)
def test_supplied_routing_is_validated_against_the_target(
    kind, outputs, inputs, index, expected
):
    with pytest.raises(ValueError, match=expected):
        _resolve(
            [supplied([sequencer(index, outputs=outputs, inputs=inputs)])],
            kind=kind,
            oscillator_frequency=None if kind is QbloxModuleKind.qrc else 4_000_000_000,
            carrier_frequency=(
                200_000_000 if kind is QbloxModuleKind.qrc else 4_200_000_000
            ),
        )


def test_acquisition_configuration_requires_an_acquisition_sequencer():
    with pytest.raises(ValueError, match="is not acquisition-capable"):
        _resolve(
            [
                supplied(
                    [
                        sequencer(
                            8,
                            outputs=[2],
                            values={"square_weight_acq": {"integration_length": 4}},
                        )
                    ]
                )
            ],
            kind=QbloxModuleKind.qrc,
            oscillator_frequency=None,
            carrier_frequency=200_000_000,
        )


def test_sequencer_without_supplied_routing_is_rejected():
    with pytest.raises(ValueError, match="supplies no routing"):
        _resolve([supplied([QbloxSequencerConfiguration(index=0)])])


def test_ports_sharing_a_module_share_one_module_configuration():
    bindings = _resolve(
        [
            supplied([sequencer(0)], module_values={"attenuation": {"out0": 6}}),
            supplied([sequencer(1)], module_values={"attenuation": {"out0": 6}}),
        ]
    )

    configs = {binding.module_config for binding in bindings.values()}
    assert len(configs) == 1
    assert {binding.port_id for binding in bindings.values()} == {"port-0", "port-1"}


def test_unallocated_bank_entries_are_validated_too():
    """An unusable bank entry is a source defect even when nothing allocates it."""

    with pytest.raises(ValueError, match="Supplied sequencer 999 .* is outside the"):
        _resolve([supplied([sequencer(0), sequencer(999)])])


def test_unallocated_bank_entries_are_validated_against_the_target_routing():
    with pytest.raises(
        ValueError, match="Supplied sequencer 1 .* routes to output 9, absent on qcm_rf"
    ):
        _resolve([supplied([sequencer(0), sequencer(1, outputs=[9])])])


def test_unallocated_bank_entries_must_supply_routing():
    bank = [sequencer(0), QbloxSequencerConfiguration(index=1)]

    with pytest.raises(ValueError, match="Supplied sequencer 1 .* supplies no routing"):
        _resolve([supplied(bank)])


@pytest.mark.parametrize(
    "connection",
    [
        SequencerConnection(
            connections=(PortConnection(direction=DirectionKind.output, port_ids=(0,)),),
            acquisition_enabled=True,
        ),
        SequencerConnection(
            connections=(PortConnection(direction=DirectionKind.output, port_ids=(0,)),),
            acquisition_disabled=True,
        ),
        SequencerConnection(
            connections=(PortConnection(direction=DirectionKind.output, port_ids=(0,)),),
            disabled_acquisition_paths=frozenset({SignalPath.q}),
        ),
    ],
)
def test_connection_side_acquisition_state_requires_an_acquisition_capable_module(
    connection,
):
    bank = [QbloxSequencerConfiguration(index=0, connection=connection)]

    with pytest.raises(ValueError, match="is not acquisition-capable"):
        _resolve([supplied(bank)], kind=QbloxModuleKind.qcm)


def test_connection_side_acquisition_state_is_accepted_on_an_acquiring_module():
    bank = [
        QbloxSequencerConfiguration(
            index=0,
            connection=SequencerConnection(
                connections=(
                    PortConnection(direction=DirectionKind.output, port_ids=(0,)),
                ),
                acquisition_enabled=True,
            ),
        )
    ]

    binding = _resolve([supplied(bank)], kind=QbloxModuleKind.qrm_rf)["port-0-channel-0"]

    assert bool(binding.sequencer_config.acquisition_enabled.value.data)


def test_an_io_connection_binds_both_lanes_in_both_directions():
    """``io0_1`` is complex mode: outputs 0 and 1 drive, inputs 0 and 1 acquire."""

    bank = [
        QbloxSequencerConfiguration(
            index=0,
            connection=SequencerConnection(
                connections=(PortConnection(direction=DirectionKind.io, port_ids=(0, 1)),)
            ),
        )
    ]

    binding = _resolve([supplied(bank)], kind=QbloxModuleKind.qrm)["port-0-channel-0"]

    assert [output.output_id.data for output in binding.module_config.outputs] == [0, 1]
    assert [entry.input_id.data for entry in binding.module_config.inputs] == [0, 1]


def test_a_complex_input_connection_binds_both_acquisition_lanes():
    """``in0_1`` puts the acquisition path of a baseband module in complex mode."""

    bank = [
        QbloxSequencerConfiguration(
            index=0,
            connection=SequencerConnection(
                connections=(
                    PortConnection(direction=DirectionKind.output, port_ids=(0, 1)),
                    PortConnection(direction=DirectionKind.input, port_ids=(0, 1)),
                )
            ),
        )
    ]

    binding = _resolve([supplied(bank)], kind=QbloxModuleKind.qrm)["port-0-channel-0"]

    assert [entry.input_id.data for entry in binding.module_config.inputs] == [0, 1]
    assert list(binding.sequencer_config.connections) == [
        ConnectionAttr(DirectionKind.output, [0, 1]),
        ConnectionAttr(DirectionKind.input, [0, 1]),
    ]


@pytest.mark.parametrize(
    ("connection", "expected"),
    [
        pytest.param(
            SequencerConnection(
                connections=(
                    PortConnection(direction=DirectionKind.io, port_ids=(0, 1)),
                    PortConnection(direction=DirectionKind.output, port_ids=(1,)),
                )
            ),
            "connects outputs [1] more than once",
            id="duplicate-output-lane",
        ),
        pytest.param(
            SequencerConnection(
                connections=(
                    PortConnection(direction=DirectionKind.io, port_ids=(0, 1)),
                    PortConnection(direction=DirectionKind.input, port_ids=(0,)),
                )
            ),
            "connects inputs [0] more than once",
            id="duplicate-input-lane",
        ),
        pytest.param(
            SequencerConnection(
                connections=(
                    PortConnection(direction=DirectionKind.output, port_ids=(0,)),
                ),
                disabled_outputs=frozenset({0}),
            ),
            "both connects and disables outputs [0]",
            id="output-connected-and-disabled",
        ),
        pytest.param(
            SequencerConnection(
                connections=(
                    PortConnection(direction=DirectionKind.output, port_ids=(0,)),
                ),
                acquisition_path_connections=(
                    AcquisitionPathConnection(input_id=0, path=SignalPath.i),
                ),
                disabled_acquisition_paths=frozenset({SignalPath.i}),
            ),
            "both connects and disables acquisition paths ['I']",
            id="acquisition-path-connected-and-disabled",
        ),
        pytest.param(
            SequencerConnection(
                connections=(
                    PortConnection(direction=DirectionKind.output, port_ids=(0,)),
                ),
                acquisition_enabled=True,
                acquisition_disabled=True,
            ),
            "both enables and disables acquisition",
            id="acquisition-enabled-and-disabled",
        ),
        pytest.param(
            SequencerConnection(
                connections=(PortConnection(direction=DirectionKind.input, port_ids=(0,)),),
                acquisition_disabled=True,
            ),
            "disables acquisition while connecting inputs",
            id="acquisition-disabled-with-inputs",
        ),
    ],
)
def test_conflicting_routing_is_rejected_in_every_bank_entry(connection, expected):
    """An unallocated entry claiming a lane twice is a source defect all the same."""

    bank = [sequencer(0), QbloxSequencerConfiguration(index=1, connection=connection)]

    with pytest.raises(ValueError, match=re.escape("Supplied sequencer 1 ")):
        _resolve([supplied(bank)], kind=QbloxModuleKind.qrm)
    with pytest.raises(ValueError, match=re.escape(expected)):
        _resolve([supplied(bank)], kind=QbloxModuleKind.qrm)


def test_unrepresentable_sequencer_values_are_rejected_in_every_bank_entry():
    bank = [sequencer(0), sequencer(1, values={"colour": "green"})]

    with pytest.raises(
        ValueError, match=r"Supplied sequencer 1 .* \['colour'\], which the Q1 dialect"
    ):
        _resolve([supplied(bank)], kind=QbloxModuleKind.qrm)


@pytest.mark.parametrize("group", ["awg", "nco", "thresholded_acq"])
def test_a_malformed_sequencer_configuration_group_is_rejected(group):
    bank = [sequencer(0, values={group: 1.0})]

    with pytest.raises(ValueError, match="configuration group is a mapping of fields"):
        _resolve([supplied(bank)], kind=QbloxModuleKind.qrm)


@pytest.mark.parametrize("group", ["offset", "lo", "scope_acq"])
def test_a_malformed_module_configuration_group_is_rejected(group):
    with pytest.raises(ValueError, match="configuration group is a mapping of fields"):
        _resolve(
            [supplied([sequencer(0)], module_values={group: ["out0"]})],
            kind=QbloxModuleKind.qrm,
        )


def test_unrepresentable_module_values_are_rejected():
    with pytest.raises(ValueError, match=r"\['colour'\], which the Q1 dialect"):
        _resolve(
            [supplied([sequencer(0)], module_values={"colour": "green"})],
            kind=QbloxModuleKind.qrm,
        )


@pytest.mark.parametrize(
    ("connection", "expected"),
    [
        pytest.param(
            SequencerConnection(
                connections=(PortConnection(direction=DirectionKind.io, port_ids=(0, 1)),)
            ),
            "accepts only one I/O port per connection",
            id="complex-io",
        ),
        pytest.param(
            SequencerConnection(
                connections=(
                    PortConnection(direction=DirectionKind.input, port_ids=(0, 1)),
                )
            ),
            "accepts only one I/O port per connection",
            id="complex-input",
        ),
        pytest.param(
            SequencerConnection(
                output_path_connections=(
                    OutputPathConnection(output_id=0, path=SignalPath.i),
                )
            ),
            "drives an output with the combined 'IQ' path only",
            id="output-i-path",
        ),
        pytest.param(
            SequencerConnection(
                output_path_connections=(
                    OutputPathConnection(output_id=0, path=SignalPath.q),
                )
            ),
            "drives an output with the combined 'IQ' path only",
            id="output-q-path",
        ),
        pytest.param(
            SequencerConnection(
                acquisition_path_connections=(
                    AcquisitionPathConnection(input_id=0, path=SignalPath.i),
                )
            ),
            "selects one input for its whole acquisition path",
            id="acquisition-i-path",
        ),
        pytest.param(
            SequencerConnection(
                connections=(
                    PortConnection(direction=DirectionKind.output, port_ids=(0,)),
                ),
                disabled_acquisition_paths=frozenset({SignalPath.q}),
            ),
            "selects one input for its whole acquisition path",
            id="disabled-q-path",
        ),
    ],
)
@pytest.mark.parametrize("kind", [QbloxModuleKind.qrm_rf, QbloxModuleKind.qrc])
def test_rf_modules_reject_component_specific_routing(connection, expected, kind):
    bank = [QbloxSequencerConfiguration(index=0, connection=connection)]

    with pytest.raises(ValueError, match=re.escape(expected)):
        _resolve(
            [supplied(bank)],
            kind=kind,
            oscillator_frequency=(None if kind is QbloxModuleKind.qrc else 4_000_000_000),
            carrier_frequency=(
                200_000_000 if kind is QbloxModuleKind.qrc else 4_200_000_000
            ),
        )


def test_rf_modules_accept_combined_iq_and_acquisition_routing():
    bank = [
        QbloxSequencerConfiguration(
            index=0,
            connection=SequencerConnection(
                output_path_connections=(
                    OutputPathConnection(output_id=0, path=SignalPath.iq),
                ),
                acquisition_path_connections=(
                    AcquisitionPathConnection(input_id=0, path=SignalPath.iq),
                ),
            ),
        )
    ]

    binding = _resolve([supplied(bank)], kind=QbloxModuleKind.qrm_rf)["port-0-channel-0"]

    assert list(binding.sequencer_config.output_path_connections) == [
        OutputPathConnectionAttr(0, SignalPath.iq)
    ]
    assert list(binding.sequencer_config.acquisition_path_connections) == [
        AcquisitionPathConnectionAttr(0, SignalPath.iq)
    ]
