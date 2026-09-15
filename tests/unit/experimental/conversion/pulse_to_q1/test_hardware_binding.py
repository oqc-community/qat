# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import json
from dataclasses import replace
from pathlib import Path

import pytest
from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.utils.exceptions import PassFailedException

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
    data = _ambiguous_channel_data()

    module = _bind(data, _sequence(4_200_000_000, channel_id="port-0-channel-1"))

    [bound] = module.body.block.ops
    assert bound.seq_idx == SequencerIndexAttr(1)


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

    binding = QbloxHardwareBindingPass(canonical_data())._resolve_sequence(
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

    def _resolve_sequence_with_duplicate(*_args, **_kwargs):
        return duplicate

    monkeypatch.setattr(
        QbloxHardwareBindingPass,
        "_resolve_sequence",
        _resolve_sequence_with_duplicate,
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
    finally bound onto a sequence that must verify.
    """

    data = materialise(
        source_payload=json.loads(CALIBRATION_FILE.read_text()), source_additional_data={}
    )
    channel = next(entry for entry in data.channels if entry.id == channel_id)

    module = _bind(
        data, _sequence(int(channel.frequency), port_id=port_id, channel_id=channel_id)
    )

    [sequence] = module.body.block.ops
    assert sequence.port_id.data == port_id
    assert sequence.slot_idx == SlotIndexAttr(slot)
    assert sequence.seq_idx == SequencerIndexAttr(index)
    assert sequence.instrument_id.data.startswith("test_save_model_")
    assert sequence.module_config.local_oscillators.data
    assert sequence.sequencer_config.connections is not None
    assert int(channel.frequency) == carrier


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
