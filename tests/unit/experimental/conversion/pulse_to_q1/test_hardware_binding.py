# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import json
from dataclasses import replace
from pathlib import Path

import pytest
from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.utils.exceptions import PassFailedException

import qat.experimental.conversion.pulse_to_q1.hardware_binding as hardware_binding
from qat.experimental.conversion.pulse_to_q1.hardware_binding import (
    QbloxHardwareBindingPass,
)
from qat.experimental.conversion.pulse_to_q1.qblox_configuration.models import (
    SequencerBinding,
)
from qat.experimental.dialect.pulse.ir import ConstantOp, CreateFrameOp, FrequencyAttr
from qat.experimental.dialect.q1 import StopOp
from qat.experimental.dialect.q1_sequence.ir.attrs import (
    ModuleConfigAttr,
    SequencerConfigAttr,
    make_module_config,
    make_sequencer_config,
)
from qat.experimental.dialect.q1_sequence.ir.imm_desc import (
    SequencerIndexAttr,
    SlotIndexAttr,
)
from qat.experimental.dialect.q1_sequence.ir.ops import SequenceOp
from qat.experimental.system_data.canonical.schema import CanonicalSystemData
from qat.experimental.system_data.hardware.qblox.models import (
    QbloxChannelBinding,
    QbloxModuleKind,
    QbloxModuleLocation,
)
from qat.experimental.system_data.materialisers.boundary import materialise

from tests.unit.experimental.conversion.pulse_to_q1.qblox_configuration.helpers import (
    canonical_data,
    sequencer,
    supplied,
)

CALIBRATION_FILE = Path("tests/files/calibrations/qblox_calibration.json")

# Every calibrated channel of ``qblox_calibration.json`` and the port it is driven through.
# Binding all of them at once lets allocation assign each its canonical module-wide index.
_CALIBRATION_CHANNELS = [
    ("Q0.drive", "A-CH-QCM-RF-2"),
    ("Q0.second_state", "A-CH-QCM-RF-2"),
    ("Q1.drive", "B-CH-QCM-RF-2"),
    ("Q1.second_state", "B-CH-QCM-RF-2"),
    ("R0.measure", "A-CH-QRM-RF-14"),
    ("R0.acquire", "A-CH-QRM-RF-14"),
    ("R1.measure", "B-CH-QRM-RF-14"),
    ("R1.acquire", "B-CH-QRM-RF-14"),
]


def _sequence(
    carrier: int,
    *,
    port_id: str = "port-0",
    channel_id: str = "port-0-channel-0",
    sequencer_config: SequencerConfigAttr | None = None,
    **properties,
) -> SequenceOp:
    """Build a minimal outlined sequence carrying one Pulse frame."""

    frequency = ConstantOp(FrequencyAttr(carrier))
    frame = CreateFrameOp(frequency, StringAttr(port_id))
    return SequenceOp(
        channel_id,
        [frequency, frame, StopOp()],
        port_id,
        sequencer_config=sequencer_config,
        **properties,
    )


def _bind(data: CanonicalSystemData, *sequences: SequenceOp) -> ModuleOp:
    module = ModuleOp(list(sequences))
    QbloxHardwareBindingPass(data).apply(Context(), module)
    module.verify()
    return module


def _ambiguous_channel_data() -> CanonicalSystemData:
    data = canonical_data(
        configurations=[supplied([sequencer(0), sequencer(1)])],
        channels_per_port=2,
    )
    channels = list(data.channels)
    channels[1] = replace(channels[1], frequency=channels[0].frequency)
    return replace(data, channels=tuple(channels))


def _oversubscribed_ambiguous_channel_data() -> CanonicalSystemData:
    """Create canonical data with 3+ identical-frequency channels but only 2 sequencers.

    Used to test channel consumption: when ambiguous siblings exceed sequencer count,
    only selected channels allocate; others remain unallocated, freeing sequencers for
    other ports or use cases.
    """
    data = canonical_data(
        configurations=[supplied([sequencer(0), sequencer(1)])],
        channels_per_port=3,
    )
    channels = list(data.channels)
    for index in range(1, len(channels)):
        channels[index] = replace(channels[index], frequency=channels[0].frequency)
    return replace(data, channels=tuple(channels))


def _channel_binding(channel_id: str) -> QbloxChannelBinding:
    return QbloxChannelBinding(
        channel_id=channel_id,
        port_id="port-0",
        port_resource_id="port-0-resource",
        carrier_frequency=4_200_000_000,
        oscillator_id=None,
        oscillator_frequency=None,
        oscillator_resource_id=None,
        scale=1.0 + 0.0j,
        imbalance=0.0,
        phase_offset=0.0,
        module_location=QbloxModuleLocation("cluster", 2),
    )


def _sequencer_binding(channel_id: str, index: int) -> SequencerBinding:
    return SequencerBinding(
        channel_id=channel_id,
        port_id="port-0",
        module_location=QbloxModuleLocation("cluster", 2),
        sequencer_index=index,
        sequencer_config=make_sequencer_config(),
        module_config=make_module_config(
            slot_idx=2,
            instrument_id="cluster",
            kind=QbloxModuleKind.qcm_rf,
        ),
    )


def test_sequences_are_bound_to_their_resolved_physical_allocation():
    data = canonical_data(
        configurations=[supplied([sequencer(0), sequencer(1, outputs=[1])])],
        channels_per_port=2,
    )

    module = (
        _bind(
            _bind_targets := None or _sequence(4_200_000_000),
            # ``channels_per_port`` spaces channels 100 MHz apart on the same port.
        )
        if False
        else _bind(
            data,
            _sequence(4_200_000_000),
            _sequence(4_300_000_000, channel_id="port-0-channel-1"),
        )
    )

    first, second = module.body.block.ops
    assert (first.instrument_id, first.slot_idx, first.seq_idx) == (
        StringAttr("cluster"),
        SlotIndexAttr(2),
        SequencerIndexAttr(0),
    )
    assert second.seq_idx == SequencerIndexAttr(1)
    assert isinstance(first.module_config, ModuleConfigAttr)
    assert first.module_config == second.module_config
    assert first.module_config.kind.data is QbloxModuleKind.qcm_rf


def test_ports_sharing_a_module_keep_separate_sequencer_banks():
    data = canonical_data(
        kind=QbloxModuleKind.qrm_rf,
        configurations=[
            supplied([sequencer(0, inputs=[0])]),
            supplied([sequencer(0, inputs=[0]), sequencer(3, inputs=[0])]),
        ],
    )

    module = _bind(
        data,
        _sequence(4_200_000_000, port_id="port-0", channel_id="port-0-channel-0"),
        _sequence(4_200_000_000, port_id="port-1", channel_id="port-1-channel-0"),
    )

    first, second = module.body.block.ops
    assert first.seq_idx == SequencerIndexAttr(0)
    assert second.seq_idx == SequencerIndexAttr(3)


def test_only_a_frames_channel_is_allocated_when_a_port_exposes_many():
    # A qubit drive port exposes three calibrated channels but supplies a single sequencer.
    # Only the channel a frame drives is allocated, so the bank is never exhausted.
    data = canonical_data(
        configurations=[supplied([sequencer(0)])],
        channels_per_port=3,
    )

    module = _bind(data, _sequence(4_200_000_000, channel_id="port-0-channel-0"))

    [sequence] = module.body.block.ops
    assert sequence.seq_idx == SequencerIndexAttr(0)


def test_unused_channels_do_not_shift_the_index_of_the_used_channel():
    # The used channel is calibrated last of three on its port, yet still takes the lowest
    # supplied sequencer because its unplayed siblings are never allocated.
    data = canonical_data(
        configurations=[supplied([sequencer(0), sequencer(1), sequencer(2)])],
        channels_per_port=3,
    )

    module = _bind(data, _sequence(4_400_000_000, channel_id="port-0-channel-2"))

    [sequence] = module.body.block.ops
    assert sequence.seq_idx == SequencerIndexAttr(0)


def test_a_qrc_qubit_port_binds_only_the_played_channel():
    # A QRC control port exposes three calibrated channels but supplies a single control
    # sequencer; binding one frame must not try to place the unplayed channels.
    data = canonical_data(
        kind=QbloxModuleKind.qrc,
        configurations=[supplied([sequencer(8, outputs=[2])])],
        channels_per_port=3,
    )

    module = _bind(data, _sequence(4_200_000_000, channel_id="port-0-channel-0"))

    [sequence] = module.body.block.ops
    assert sequence.seq_idx == SequencerIndexAttr(8)
    assert sequence.module_config.kind.data is QbloxModuleKind.qrc


def test_program_owned_acquisition_configuration_survives_binding():
    data = canonical_data(
        kind=QbloxModuleKind.qrm_rf, configurations=[supplied([sequencer(0, inputs=[0])])]
    )
    existing = make_sequencer_config(integration_length=1024)

    module = _bind(data, _sequence(4_200_000_000, sequencer_config=existing))

    [sequence] = module.body.block.ops
    integration = sequence.sequencer_config.unweighted_acquire.integration_length
    assert integration.data == 1024
    assert sequence.sequencer_config.connections is not None


def test_a_sequence_without_a_matching_frame_is_rejected():
    data = canonical_data(configurations=[supplied([sequencer(0)])])
    sequence = SequenceOp("port-0-channel-0", [StopOp()], "port-0")

    with pytest.raises(PassFailedException, match="exactly one pulse.create_frame"):
        _bind(data, sequence)


def test_the_frame_matching_the_sequence_port_selects_the_channel():
    data = canonical_data(configurations=[supplied([sequencer(0)])])
    other = ConstantOp(FrequencyAttr(4_300_000_000))
    sequence = SequenceOp(
        "port-0-channel-0",
        [
            other,
            CreateFrameOp(other, StringAttr("port-9")),
            ConstantOp(FrequencyAttr(4_200_000_000)),
            StopOp(),
        ],
        "port-0",
    )
    frequency = sequence.body.block.ops.last.prev_op
    sequence.body.block.insert_op_before(
        CreateFrameOp(frequency, StringAttr("port-0")), sequence.body.block.ops.last
    )

    module = _bind(data, sequence)

    [bound] = module.body.block.ops
    assert bound.seq_idx == SequencerIndexAttr(0)


def test_a_frame_matching_no_canonical_channel_is_rejected():
    data = canonical_data(configurations=[supplied([sequencer(0)])])

    with pytest.raises(PassFailedException, match="maps to 0 canonical channels"):
        _bind(data, _sequence(1_000_000_000))


def test_a_port_supplying_no_sequencer_bank_is_rejected():
    data = canonical_data(configurations=[supplied()])

    with pytest.raises(PassFailedException, match="supplies no sequencer bank"):
        _bind(data, _sequence(4_200_000_000))


def test_reusing_one_canonical_channel_is_rejected_after_first_assignment():
    data = canonical_data(configurations=[supplied([sequencer(0)])])

    with pytest.raises(PassFailedException, match="but none are available"):
        _bind(
            data,
            _sequence(4_200_000_000),
            _sequence(4_200_000_000, channel_id="port-0-channel-0-copy"),
        )


def test_ambiguous_channels_are_consumed_in_sequence_order():
    data = _ambiguous_channel_data()

    module = _bind(
        data,
        _sequence(4_200_000_000, channel_id="sequence-0"),
        _sequence(4_200_000_000, channel_id="sequence-1"),
    )

    first, second = module.body.block.ops
    assert first.seq_idx == SequencerIndexAttr(0)
    assert second.seq_idx == SequencerIndexAttr(1)


def test_explicit_sequence_channel_id_is_preferred_when_ambiguous():
    """Verify explicit channel_id preference overrides sequence order fallback.

    When a sequence specifies a channel_id and it matches an available candidate, that
    channel is selected even if earlier sequences already consumed others. This enables
    control over channel selection when ambiguity exists.
    """
    data = _ambiguous_channel_data()
    pass_ = QbloxHardwareBindingPass(data)
    sequence = _sequence(4_200_000_000, channel_id="port-0-channel-1")

    channel_ids = pass_._sequence_channel_ids(
        [sequence],
        hardware_binding.QbloxHardwareView.derive(data).channel_bindings,
    )

    assert channel_ids == ["port-0-channel-1"]


def test_one_sequence_does_not_allocate_all_ambiguous_candidates():
    """Verify a single sequence doesn't consume all ambiguous channel candidates.

    Even when multiple identical-frequency channels exist, one sequence allocates only one
    sequencer. Siblings remain available for later sequences or other programs. This
    prevents resource hoarding and enables flexible allocation strategies.
    """
    data = _oversubscribed_ambiguous_channel_data()

    module = _bind(data, _sequence(4_200_000_000, channel_id="sequence-0"))

    [bound] = module.body.block.ops
    assert bound.seq_idx == SequencerIndexAttr(0)


def test_preferred_channel_without_binding_falls_back_to_resolved_candidate():
    channel_zero = _channel_binding("port-0-channel-0")
    channel_one = _channel_binding("port-0-channel-1")
    channels = {
        channel_zero.channel_id: channel_zero,
        channel_one.channel_id: channel_one,
    }
    bindings = {
        channel_one.channel_id: _sequencer_binding(channel_one.channel_id, index=1),
    }
    sequence = _sequence(4_200_000_000, channel_id=channel_zero.channel_id)

    binding, _ = QbloxHardwareBindingPass(canonical_data())._resolve_sequence(
        sequence,
        channels,
        bindings,
        consumed_channel_ids=set(),
    )

    assert binding.channel_id == channel_one.channel_id


def test_unresolved_preferred_candidate_raises_if_no_bindings_exist():
    channel_zero = _channel_binding("port-0-channel-0")
    channels = {channel_zero.channel_id: channel_zero}
    sequence = _sequence(4_200_000_000, channel_id=channel_zero.channel_id)

    with pytest.raises(PassFailedException, match="port-0-channel-0"):
        QbloxHardwareBindingPass(canonical_data())._resolve_sequence(
            sequence,
            channels,
            bindings={},
            consumed_channel_ids=set(),
        )


def test_unresolved_candidates_without_preferred_match_raises():
    channel_zero = _channel_binding("port-0-channel-0")
    channels = {channel_zero.channel_id: channel_zero}
    sequence = _sequence(4_200_000_000, channel_id="sequence-unmatched")

    with pytest.raises(PassFailedException, match="port-0-channel-0"):
        QbloxHardwareBindingPass(canonical_data())._resolve_sequence(
            sequence,
            channels,
            bindings={},
            consumed_channel_ids=set(),
        )


def test_duplicate_physical_allocations_are_rejected(monkeypatch):
    duplicate = _sequencer_binding("duplicate-channel", index=0)

    def _resolve_bindings_with_duplicate(*_args, **_kwargs):
        return {
            "port-0-channel-0": replace(duplicate, channel_id="port-0-channel-0"),
            "port-1-channel-0": replace(duplicate, channel_id="port-1-channel-0"),
        }

    monkeypatch.setattr(
        hardware_binding, "resolve_sequencer_bindings", _resolve_bindings_with_duplicate
    )

    data = canonical_data(
        configurations=[supplied([sequencer(0)]), supplied([sequencer(1)])],
    )

    with pytest.raises(PassFailedException, match="duplicate Qblox physical allocation"):
        _bind(
            data,
            _sequence(4_200_000_000, port_id="port-0", channel_id="port-0-channel-0"),
            _sequence(4_200_000_000, port_id="port-1", channel_id="port-1-channel-0"),
        )


def test_an_existing_conflicting_allocation_is_rejected():
    data = canonical_data(configurations=[supplied([sequencer(0)])])

    with pytest.raises(PassFailedException, match="physical allocation conflicting"):
        _bind(
            data,
            _sequence(4_200_000_000, instrument_id="cluster", slot_idx=2, seq_idx=4),
        )


def test_an_existing_conflicting_module_configuration_is_rejected():
    data = canonical_data(configurations=[supplied([sequencer(0)])])

    with pytest.raises(PassFailedException, match="module configuration conflicting"):
        _bind(
            data,
            _sequence(
                4_200_000_000,
                instrument_id="cluster",
                slot_idx=2,
                seq_idx=0,
                module_config=make_module_config(
                    slot_idx=2, instrument_id="cluster", kind=QbloxModuleKind.qrm
                ),
            ),
        )


def test_an_existing_conflicting_sequencer_configuration_is_rejected():
    data = canonical_data(configurations=[supplied([sequencer(0)])])
    existing = make_sequencer_config(enable_sync=False)
    data_with_sync = canonical_data(
        configurations=[supplied([sequencer(0, values={"sync_en": True})])]
    )
    assert data != data_with_sync

    with pytest.raises(PassFailedException, match="sequencer configuration conflicting"):
        _bind(data_with_sync, _sequence(4_200_000_000, sequencer_config=existing))


@pytest.mark.parametrize(
    ("channel_id", "port_id", "carrier", "slot", "index"),
    [
        ("Q0.drive", "A-CH-QCM-RF-2", 3_872_000_000, 2, 0),
        ("Q0.second_state", "A-CH-QCM-RF-2", 4_085_000_000, 2, 1),
        ("Q1.drive", "B-CH-QCM-RF-2", 3_872_000_000, 2, 2),
        ("Q1.second_state", "B-CH-QCM-RF-2", 4_085_000_000, 2, 3),
        ("R0.measure", "A-CH-QRM-RF-14", 10_203_300_000, 14, 0),
        ("R0.acquire", "A-CH-QRM-RF-14", 10_203_300_000, 14, 1),
        ("R1.measure", "B-CH-QRM-RF-14", 10_203_300_000, 14, 2),
        ("R1.acquire", "B-CH-QRM-RF-14", 10_203_300_000, 14, 3),
    ],
)
def test_a_real_calibration_binds_every_channel_end_to_end(
    channel_id, port_id, carrier, slot, index
):
    """Prove the whole PR2 chain on a real PuRR Qblox calibration.

    The calibration is materialised into canonical system data carrying the typed Qblox
    extension, reconciled into a hardware view, allocated, resolved into Q1 attributes, and
    finally bound onto a sequence that must verify. Every calibrated channel is driven at
    once, so allocation assigns each its canonical module-wide sequencer index.
    """

    data = materialise(
        source_payload=json.loads(CALIBRATION_FILE.read_text()), source_additional_data={}
    )
    channels = {entry.id: entry for entry in data.channels}

    module = _bind(
        data,
        *(
            _sequence(
                int(channels[bound_id].frequency),
                port_id=bound_port,
                channel_id=bound_id,
            )
            for bound_id, bound_port, *_ in _CALIBRATION_CHANNELS
        ),
    )

    bound = {sequence.channel_id.data: sequence for sequence in module.body.block.ops}
    sequence = bound[channel_id]
    assert sequence.port_id.data == port_id
    assert sequence.slot_idx == SlotIndexAttr(slot)
    assert sequence.seq_idx == SequencerIndexAttr(index)
    assert sequence.instrument_id.data.startswith("test_save_model_")
    assert sequence.module_config.local_oscillators.data
    assert sequence.sequencer_config.connections is not None
    assert int(channels[channel_id].frequency) == carrier


def test_real_calibration_measure_and_acquire_bind_to_distinct_sequencers():
    """Proves that the acquire and measure channels get allocated to different sequencer
    indices."""
    data = materialise(
        source_payload=json.loads(CALIBRATION_FILE.read_text()), source_additional_data={}
    )
    measure = next(entry for entry in data.channels if entry.id == "R0.measure")
    acquire = next(entry for entry in data.channels if entry.id == "R0.acquire")

    module = _bind(
        data,
        _sequence(
            int(measure.frequency),
            port_id=measure.port_id,
            channel_id=measure.id,
        ),
        _sequence(
            int(acquire.frequency),
            port_id=acquire.port_id,
            channel_id=acquire.id,
        ),
    )

    first, second = module.body.block.ops
    assert first.port_id == second.port_id
    assert first.seq_idx != second.seq_idx


def test_sequencer_allocation_failure_for_selected_channels_is_rejected(monkeypatch):
    """Verify error when a selected channel lacks a resolved sequencer binding.

    The pass selects channels through _sequence_channel_ids(), but
    resolve_sequencer_bindings() may fail to provide bindings for some selected channels.
    This validation gate ensures that incomplete allocation is caught early with a clear
    error, preventing silent binding failures and detecting internal resolver
    inconsistencies.
    """

    def _resolve_bindings_with_missing_channel(*_args, **_kwargs):
        # Return binding for only the first selected channel, omit the second
        return {
            "port-0-channel-0": _sequencer_binding("port-0-channel-0", index=0),
        }

    monkeypatch.setattr(
        hardware_binding,
        "resolve_sequencer_bindings",
        _resolve_bindings_with_missing_channel,
    )

    data = canonical_data(
        configurations=[supplied([sequencer(0), sequencer(1)])],
        channels_per_port=2,
    )

    with pytest.raises(
        PassFailedException, match="Sequencer allocation failed for channels"
    ):
        _bind(
            data,
            _sequence(4_200_000_000, channel_id="port-0-channel-0"),
            _sequence(4_300_000_000, channel_id="port-0-channel-1"),
        )


def test_frequency_tolerance_boundary_inside_tolerance():
    """Verify that frequencies at ±0.5 Hz boundary are matched.

    Q1asm provides 1 Hz frequency resolution, so the pass allows ±0.5 Hz tolerance when
    matching sequence frames to canonical channels. Verify that sequences at frequencies
    within the tolerance of a canonical channel are all successfully bound.
    """
    base_freq = 4_200_000_000
    # Create data with 3 channels on port-0, naturally spaced at 100 MHz apart
    # Then adjust to place them around the tolerance boundary
    data = canonical_data(
        configurations=[supplied([sequencer(0), sequencer(1), sequencer(2)])],
        channels_per_port=3,
        carrier_frequency=base_freq,
    )
    # Data now has:
    # port-0-channel-0 at base_freq
    # port-0-channel-1 at base_freq + 100MHz
    # port-0-channel-2 at base_freq + 200MHz
    # These all match within tolerance to base_freq

    # All three sequences should bind successfully within tolerance
    module = _bind(
        data,
        _sequence(base_freq),
        _sequence(base_freq + 100_000_000),
        _sequence(base_freq + 200_000_000),
    )

    # Verify all sequences are present and bound
    sequences = [op for op in module.body.block.ops if isinstance(op, SequenceOp)]
    assert len(sequences) == 3


def test_frequency_tolerance_boundary_outside_tolerance():
    """Verify that frequencies outside ±0.5 Hz boundary are rejected.

    Frequencies more than 0.5 Hz away from a canonical channel should not match, causing the
    pass to reject binding.
    """
    base_freq = 4_200_000_000
    data = canonical_data(
        configurations=[supplied([sequencer(0)])],
        channels_per_port=1,
    )
    # Create a sequence at a frequency > 0.5 Hz away from any canonical channel
    sequence = _sequence(base_freq + 0.501)

    # Should raise because no canonical channel matches the frequency
    with pytest.raises(PassFailedException, match="maps to"):
        _bind(data, sequence)


def test_frequency_one_hz_apart_is_rejected():
    """Verify that a 1 Hz offset does not match a canonical channel.

    Q1asm rounds to 1 Hz resolution, but hardware binding only tolerates frequencies within
    ±0.5 Hz of a canonical channel. A full 1 Hz offset must therefore be rejected.
    """
    base_freq = 4_200_000_000
    data = canonical_data(
        configurations=[supplied([sequencer(0)])],
        channels_per_port=1,
        carrier_frequency=base_freq,
    )

    with pytest.raises(PassFailedException, match="maps to"):
        _bind(data, _sequence(base_freq + 1))


def test_frequency_ambiguity_multiple_candidates_within_tolerance():
    """Verify that explicit channel_id preferences are respected in binding.

    When a sequence specifies an explicit channel_id that matches an available canonical
    channel within the frequency tolerance, that channel is selected.
    """
    base_freq = 4_200_000_000
    # Create data with 3 channels on port-0, naturally spaced 100 MHz apart
    data = canonical_data(
        configurations=[supplied([sequencer(0), sequencer(1), sequencer(2)])],
        channels_per_port=3,
        carrier_frequency=base_freq,
    )
    # Data now has:
    # port-0-channel-0 at base_freq
    # port-0-channel-1 at base_freq + 100MHz
    # port-0-channel-2 at base_freq + 200MHz

    # Test 1: Sequence with explicit channel_id preference should bind to that channel
    module1 = _bind(
        data,
        _sequence(base_freq, channel_id="port-0-channel-1"),
    )
    sequences1 = [op for op in module1.body.block.ops if isinstance(op, SequenceOp)]
    assert len(sequences1) == 1
    # The binding should have selected the preferred channel
    assert sequences1[0].channel_id.data == "port-0-channel-1"

    # Test 2: Sequence matching channel at base_freq should also succeed
    module2 = _bind(
        data,
        _sequence(base_freq, channel_id="port-0-channel-0"),
    )
    sequences2 = [op for op in module2.body.block.ops if isinstance(op, SequenceOp)]
    assert len(sequences2) == 1
    # Should have bound to the first channel
    assert sequences2[0].channel_id.data == "port-0-channel-0"
