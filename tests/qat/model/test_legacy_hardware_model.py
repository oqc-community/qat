import numpy as np
import pytest

from qat.model.device import FrozenDict
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


@pytest.mark.parametrize("n_qubits", [0, 1, 2, 4, 32, 64])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
class TestHardwareModelConversion:
    def test_echo(self, n_qubits, seed):
        logical_connectivity = random_directed_connectivity(n_qubits, seed=seed)
        logical_connectivity = [
            (q1_index, q2_index)
            for q1_index in logical_connectivity
            for q2_index in logical_connectivity[q1_index]
        ]

        hw_echo = apply_setup_to_echo_hardware(
            qubit_count=n_qubits, connectivity=logical_connectivity
        )
        hw_pyd_echo = hw_echo.export_pydantic()

        for qubit_index, qubit in hw_pyd_echo.qubits.items():
            # Compare qubit physical basebands.
            assert (
                qubit.physical_channel.baseband.frequency
                == hw_echo.get_qubit(qubit_index).physical_channel.baseband_frequency
            )
            assert (
                qubit.physical_channel.baseband.if_frequency
                == hw_echo.get_qubit(qubit_index).physical_channel.baseband_if_frequency
            )
            # Compare resonator physical basebands.
            assert (
                qubit.resonator.physical_channel.baseband.frequency
                == hw_echo.get_qubit(
                    qubit_index
                ).measure_device.physical_channel.baseband_frequency
            )
            assert (
                qubit.resonator.physical_channel.baseband.if_frequency
                == hw_echo.get_qubit(
                    qubit_index
                ).measure_device.physical_channel.baseband_if_frequency
            )

            # Compare qubit physical channels.
            assert (
                qubit.physical_channel.sample_time
                == hw_echo.get_qubit(qubit_index).physical_channel.sample_time
            )
            assert (
                qubit.physical_channel.block_size
                == hw_echo.get_qubit(qubit_index).physical_channel.block_size
            )
            assert (
                qubit.physical_channel.phase_iq_offset
                == hw_echo.get_qubit(qubit_index).physical_channel.phase_offset
            )
            assert (
                qubit.physical_channel.bias
                == hw_echo.get_qubit(qubit_index).physical_channel.imbalance
            )
            # Compare resonator physical channels.
            assert (
                qubit.resonator.physical_channel.sample_time
                == hw_echo.get_qubit(
                    qubit_index
                ).measure_device.physical_channel.sample_time
            )
            assert (
                qubit.resonator.physical_channel.block_size
                == hw_echo.get_qubit(qubit_index).measure_device.physical_channel.block_size
            )
            assert (
                qubit.resonator.physical_channel.phase_iq_offset
                == hw_echo.get_qubit(
                    qubit_index
                ).measure_device.physical_channel.phase_offset
            )
            assert (
                qubit.resonator.physical_channel.bias
                == hw_echo.get_qubit(qubit_index).measure_device.physical_channel.imbalance
            )

            # Compare qubit pulse channels.
            def validate_pulse_channel(pyd_pulse_channel, legacy_pulse_channel):
                assert pyd_pulse_channel.frequency == legacy_pulse_channel.frequency
                assert pyd_pulse_channel.bias == legacy_pulse_channel.bias
                assert pyd_pulse_channel.scale == legacy_pulse_channel.scale
                assert pyd_pulse_channel.fixed_if == legacy_pulse_channel.fixed_if

            for pulse_channel_name in [
                "drive",
                "freq_shift",
                "second_state",
                "cross_resonance_channels",
                "cross_resonance_cancellation_channels",
            ]:
                pyd_pulse_channel = getattr(qubit.pulse_channels, pulse_channel_name)
                if isinstance(pyd_pulse_channel, FrozenDict):
                    for resonance_pulse_channel in pyd_pulse_channel.values():
                        legacy_resonance_pulse_channel = hw_echo.get_qubit(
                            qubit_index
                        ).get_pulse_channel(
                            channel_type_mapping[pulse_channel_name],
                            auxiliary_devices=[
                                hw_echo.get_qubit(resonance_pulse_channel.auxiliary_qubit)
                            ],
                        )
                        validate_pulse_channel(
                            resonance_pulse_channel, legacy_resonance_pulse_channel
                        )
                else:
                    try:
                        legacy_pulse_channel = hw_echo.get_qubit(
                            qubit_index
                        ).get_pulse_channel(channel_type_mapping[pulse_channel_name])
                        validate_pulse_channel(pyd_pulse_channel, legacy_pulse_channel)
                    except KeyError:
                        assert np.isnan(pyd_pulse_channel.frequency)

            # Compare resonator pulse channels.
            for pulse_channel_name in ["measure", "acquire"]:
                pyd_pulse_channel = getattr(
                    qubit.resonator.pulse_channels, pulse_channel_name
                )
                legacy_pulse_channel = hw_echo.get_qubit(
                    qubit_index
                ).measure_device.get_pulse_channel(channel_type_mapping[pulse_channel_name])
                validate_pulse_channel(pyd_pulse_channel, legacy_pulse_channel)
