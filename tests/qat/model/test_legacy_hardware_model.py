# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
import numpy as np
import pytest

from qat.model.convert_legacy import convert_legacy_echo_hw_to_pydantic
from qat.purr.compiler.devices import ChannelType

from tests.qat.utils.hardware_models import (
    apply_setup_to_echo_hardware,
    random_directed_connectivity,
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
    logical_connectivity = random_directed_connectivity(n_qubits, seed=seed)
    logical_connectivity = [
        (q1_index, q2_index)
        for q1_index in logical_connectivity
        for q2_index in logical_connectivity[q1_index]
    ]

    hw_legacy_echo = apply_setup_to_echo_hardware(
        qubit_count=n_qubits, connectivity=logical_connectivity
    )
    hw_pyd_echo = convert_legacy_echo_hw_to_pydantic(hw_legacy_echo)
    return (hw_pyd_echo, hw_legacy_echo)


def validate_pulse_channel(pyd_pulse_channel, legacy_pulse_channel):
    assert pyd_pulse_channel.frequency == legacy_pulse_channel.frequency
    assert pyd_pulse_channel.imbalance == legacy_pulse_channel.physical_channel.imbalance
    assert (
        pyd_pulse_channel.phase_iq_offset
        == legacy_pulse_channel.physical_channel.phase_offset
    )
    assert pyd_pulse_channel.scale == legacy_pulse_channel.scale
    assert pyd_pulse_channel.fixed_if == legacy_pulse_channel.fixed_if


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
                assert pyd_phys_ch.sample_time == leg_phys_ch.sample_time
                assert pyd_phys_ch.block_size == leg_phys_ch.block_size

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
                leg_coupling_direction.quality
                == pyd_hw.logical_connectivity_quality[leg_coupling_direction.direction]
            )
