# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
import numpy as np
import pytest

from qat.ir.waveforms import (
    GaussianWaveform,
    SofterSquareWaveform,
    SoftSquareWaveform,
    SquareWaveform,
)
from qat.model.convert_purr import convert_purr_echo_hw_to_pydantic
from qat.purr.compiler.devices import ChannelType, PulseShapeType
from qat.utils.hardware_model import (
    apply_setup_to_echo_hardware,
    random_connectivity,
)

channel_type_mapping = {
    "drive": ChannelType.drive,
    "second_state": ChannelType.second_state,
    "freq_shirt": ChannelType.freq_shift,
    "measure": ChannelType.measure,
    "acquire": ChannelType.acquire,
    "cross_resonance_channels": ChannelType.cross_resonance,
    "cross_resonance_cancellation_channels": ChannelType.cross_resonance_cancellation,
}


def get_echo_hw_pair(n_qubits, seed=42):
    physical_connectivity = random_connectivity(n_qubits, seed=seed)
    physical_connectivity = [
        (q1_index, q2_index)
        for q1_index in physical_connectivity
        for q2_index in physical_connectivity[q1_index]
    ]

    hw_legacy_echo = apply_setup_to_echo_hardware(
        qubit_count=n_qubits, connectivity=physical_connectivity
    )
    hw_pyd_echo = convert_purr_echo_hw_to_pydantic(hw_legacy_echo)
    return (hw_pyd_echo, hw_legacy_echo)


def validate_pulse_channel(pyd_pulse_channel, legacy_pulse_channel):
    assert pyd_pulse_channel.frequency == legacy_pulse_channel.frequency
    assert pyd_pulse_channel.imbalance == legacy_pulse_channel.imbalance
    assert (
        pyd_pulse_channel.phase_iq_offset
        == legacy_pulse_channel.physical_channel.phase_offset
    )
    assert pyd_pulse_channel.scale == legacy_pulse_channel.scale


def validate_pulse_shapes(pyd_pulse, legacy_pulse):
    waveform_lookup = {
        PulseShapeType.GAUSSIAN: GaussianWaveform,
        PulseShapeType.SQUARE: SquareWaveform,
        PulseShapeType.SOFT_SQUARE: SoftSquareWaveform,
        PulseShapeType.SOFTER_SQUARE: SofterSquareWaveform,
    }
    assert pyd_pulse.waveform_type == waveform_lookup[legacy_pulse["shape"]]
    assert pyd_pulse.width == legacy_pulse["width"]
    assert pyd_pulse.amp == legacy_pulse["amp"]
    assert pyd_pulse.drag == legacy_pulse["drag"]
    assert pyd_pulse.rise == legacy_pulse["rise"]
    assert pyd_pulse.phase == legacy_pulse["phase"]


@pytest.mark.parametrize("n_qubits", [0, 1, 2, 4, 32, 64])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
class TestEchoHardwareModelConversion:
    def test_physical_baseband(self, n_qubits, seed):
        pyd_hw, leg_hw = get_echo_hw_pair(n_qubits, seed=seed)

        for qubit_index, qubit in pyd_hw.qubits.items():
            # Compare qubit physical basebands.
            assert (
                qubit.physical_channel.baseband.frequency
                == leg_hw.get_qubit(qubit_index).physical_channel.baseband_frequency
            )
            assert (
                qubit.physical_channel.baseband.if_frequency
                == leg_hw.get_qubit(qubit_index).physical_channel.baseband_if_frequency
            )
            # Compare resonator physical basebands.
            assert (
                qubit.resonator.physical_channel.baseband.frequency
                == leg_hw.get_qubit(
                    qubit_index
                ).measure_device.physical_channel.baseband_frequency
            )
            assert (
                qubit.resonator.physical_channel.baseband.if_frequency
                == leg_hw.get_qubit(
                    qubit_index
                ).measure_device.physical_channel.baseband_if_frequency
            )

    def test_physical_channel(self, n_qubits, seed):
        pyd_hw, leg_hw = get_echo_hw_pair(n_qubits, seed=seed)

        for qubit_index, qubit in pyd_hw.qubits.items():
            pyd_physical_channels = [
                qubit.physical_channel,
                qubit.resonator.physical_channel,
            ]
            leg_physical_channels = [
                leg_hw.get_qubit(qubit_index).physical_channel,
                leg_hw.get_qubit(qubit_index).measure_device.physical_channel,
            ]

            for pyd_phys_ch, leg_phys_ch in zip(
                pyd_physical_channels, leg_physical_channels
            ):
                assert pyd_phys_ch.block_size == leg_phys_ch.block_size
                assert f"CH{pyd_phys_ch.name_index}" == leg_phys_ch.id

    def test_1q_pulse_channels(self, n_qubits, seed):
        pyd_hw, leg_hw = get_echo_hw_pair(n_qubits, seed=seed)

        for qubit_index, qubit in pyd_hw.qubits.items():
            for pulse_channel_name in ["drive", "freq_shift", "second_state"]:
                pyd_pulse_ch = getattr(qubit.pulse_channels, pulse_channel_name)

                try:
                    legacy_pulse_channel = leg_hw.get_qubit(qubit_index).get_pulse_channel(
                        channel_type_mapping[pulse_channel_name]
                    )
                    validate_pulse_channel(pyd_pulse_ch, legacy_pulse_channel)
                except KeyError:
                    assert np.isnan(pyd_pulse_ch.frequency)

    def test_1q_pulse_shapes(self, n_qubits, seed):
        pyd_hw, leg_hw = get_echo_hw_pair(n_qubits, seed=seed)
        for qubit_index, qubit in pyd_hw.qubits.items():
            pyd_pulse = qubit.pulse_channels.drive.pulse
            legacy_pulse = leg_hw.get_qubit(qubit_index).pulse_hw_x_pi_2
            validate_pulse_shapes(pyd_pulse, legacy_pulse)

    def test_2q_pulse_channels(self, n_qubits, seed):
        pyd_hw, leg_hw = get_echo_hw_pair(n_qubits, seed=seed)

        for qubit_index, qubit in pyd_hw.qubits.items():
            for pulse_channel_name in [
                "cross_resonance_channels",
                "cross_resonance_cancellation_channels",
            ]:
                pyd_pulse_channel = getattr(qubit.pulse_channels, pulse_channel_name)

                for resonance_pulse_channel in pyd_pulse_channel.values():
                    legacy_resonance_pulse_channel = leg_hw.get_qubit(
                        qubit_index
                    ).get_pulse_channel(
                        channel_type_mapping[pulse_channel_name],
                        auxiliary_devices=[
                            leg_hw.get_qubit(resonance_pulse_channel.auxiliary_qubit)
                        ],
                    )
                    validate_pulse_channel(
                        resonance_pulse_channel, legacy_resonance_pulse_channel
                    )

    def test_2q_pulse_shapes(self, n_qubits, seed):
        pyd_hw, leg_hw = get_echo_hw_pair(n_qubits, seed=seed)
        for qubit_index, qubit in pyd_hw.qubits.items():
            pyd_pulse_channel = getattr(qubit.pulse_channels, "cross_resonance_channels")
            for resonance_pulse_channel in pyd_pulse_channel.values():
                legacy_pulse = leg_hw.get_qubit(qubit_index).pulse_hw_zx_pi_4[
                    f"Q{resonance_pulse_channel.auxiliary_qubit}"
                ]

                validate_pulse_shapes(resonance_pulse_channel.zx_pi_4_pulse, legacy_pulse)

    def test_resonator_pulse_channels(self, n_qubits, seed):
        pyd_hw, leg_hw = get_echo_hw_pair(n_qubits, seed=seed)

        for qubit_index, qubit in pyd_hw.qubits.items():
            for pulse_channel_name in ["measure", "acquire"]:
                pyd_pulse_channel = getattr(
                    qubit.resonator.pulse_channels, pulse_channel_name
                )
                legacy_pulse_channel = leg_hw.get_qubit(
                    qubit_index
                ).measure_device.get_pulse_channel(channel_type_mapping[pulse_channel_name])
                validate_pulse_channel(pyd_pulse_channel, legacy_pulse_channel)

    def test_connectivity(self, n_qubits, seed):
        pyd_hw, leg_hw = get_echo_hw_pair(n_qubits, seed=seed)

        assert len(pyd_hw.qubits) == len(leg_hw.qubits)
        assert len(pyd_hw.qubits) == len(leg_hw.resonators)

        leg_coupling_directions = [
            qubit_coupling.direction for qubit_coupling in leg_hw.qubit_direction_couplings
        ]
        assert set(pyd_hw.logical_connectivity_quality.keys()) == set(
            leg_coupling_directions
        )

        for leg_coupling_direction in leg_hw.qubit_direction_couplings:
            assert (
                leg_coupling_direction.quality / 100.0
                == pyd_hw.logical_connectivity_quality[leg_coupling_direction.direction]
            )
