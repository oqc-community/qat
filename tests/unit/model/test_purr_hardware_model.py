# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
import hashlib
from collections import defaultdict

import numpy as np
import pytest

from qat.frontend.converters.purr import WAVEFORM_MAP
from qat.model.convert_purr import convert_purr_echo_hw_to_pydantic
from qat.model.loaders.purr import EchoModelLoader
from qat.purr.compiler.devices import ChannelType
from qat.purr.compiler.hardware_models import ErrorMitigation, ReadoutMitigation
from qat.purr.compiler.instructions import PulseShapeType
from qat.purr.utils.logger import LoggerLevel
from qat.utils.hardware_model import apply_setup_to_echo_hardware, random_connectivity

channel_type_mapping = {
    "drive": ChannelType.drive,
    "second_state": ChannelType.second_state,
    "freq_shift": ChannelType.freq_shift,
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
    hw_legacy_echo.calibration_id = hashlib.md5().hexdigest()
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
    assert pyd_pulse.waveform_type == WAVEFORM_MAP[legacy_pulse["shape"]]
    assert pyd_pulse.width == legacy_pulse["width"]
    assert pyd_pulse.amp == legacy_pulse["amp"]
    assert pyd_pulse.drag == legacy_pulse.get("drag", 0.0)
    assert pyd_pulse.rise == legacy_pulse.get("rise", 1.0 / 3.0)
    assert pyd_pulse.phase == legacy_pulse.get("phase", 0.0)


class TestEchoHardwareModelConversion:
    _NUM_QUBITS = [2, 11, 32]
    _SEEDS = [254]
    _LINEAR_MATRIX = {
        0: {"0|0": 0.9, "1|0": 0.1, "0|1": 0.2, "1|1": 0.8},
        1: {"0|0": 0.85, "1|0": 0.15, "0|1": 0.25, "1|1": 0.75},
    }
    _PULSE_PARAMS = [
        (PulseShapeType.SQUARE, 32e-9, 0.01),
        (PulseShapeType.GAUSSIAN, 16e-9, 0.02),
    ]

    @pytest.mark.parametrize("num_qubits", _NUM_QUBITS)
    @pytest.mark.parametrize("seed", _SEEDS)
    def test_physical_connectivity(self, num_qubits, seed):
        pyd_hw, leg_hw = get_echo_hw_pair(num_qubits, seed=seed)

        coupling_map = defaultdict(set)
        for qubit in leg_hw.qubits:
            for coupled_qubit in qubit.coupled_qubits:
                coupling_map[qubit.index].add(coupled_qubit.index)

        assert pyd_hw.physical_connectivity == coupling_map

    def test_unidirectional_physical_coupling(self):
        """Test that the physical connectivity is correctly represented as unidirectional."""
        leg_hw = EchoModelLoader(qubit_count=4).load()
        qubit_0 = leg_hw.get_qubit(0)
        qubit_1 = leg_hw.get_qubit(1)
        assert qubit_1 in qubit_0.coupled_qubits
        assert qubit_0 in qubit_1.coupled_qubits

        # Check that we've enforced it to be unidirectional
        qubit_1.coupled_qubits.remove(qubit_0)
        assert qubit_0 not in qubit_1.coupled_qubits
        assert qubit_1 in qubit_0.coupled_qubits
        leg_hw.qubit_direction_couplings = [
            coupling
            for coupling in leg_hw.qubit_direction_couplings
            if coupling.direction != (1, 0)
        ]

        pyd_hw = convert_purr_echo_hw_to_pydantic(leg_hw)
        assert 1 in pyd_hw.physical_connectivity[0]
        assert 0 not in pyd_hw.physical_connectivity[1]

    @pytest.mark.parametrize("num_qubits", _NUM_QUBITS)
    @pytest.mark.parametrize("seed", _SEEDS)
    def test_logical_connectivity(self, num_qubits, seed):
        pyd_hw, leg_hw = get_echo_hw_pair(num_qubits, seed=seed)

        logical_coupling = defaultdict(set)
        quality_mapping = dict()
        for coupling in leg_hw.qubit_direction_couplings:
            logical_coupling[coupling.direction[0]].add(coupling.direction[1])
            quality_mapping[coupling.direction] = coupling.quality / 100.0
        assert pyd_hw.logical_connectivity == logical_coupling
        assert pyd_hw.logical_connectivity_quality == quality_mapping

    def test_coupling_qualities_with_reduced_connectivity(self):
        """Test that we can have coupling qualities with reduced connectivity."""
        leg_hw = EchoModelLoader(qubit_count=4).load()

        # only select directions that point in a given direction
        new_couplings = [
            coupling
            for coupling in leg_hw.qubit_direction_couplings
            if coupling.direction[0] < coupling.direction[1]
        ]
        assert 0 < len(new_couplings) < len(leg_hw.qubit_direction_couplings)
        leg_hw.qubit_direction_couplings = new_couplings
        pyd_hw = convert_purr_echo_hw_to_pydantic(leg_hw)

        # assert we only see the restricted couplings
        assert sum(
            [len(couplings) for couplings in pyd_hw.logical_connectivity.values()]
        ) == len(new_couplings)
        for coupling in new_couplings:
            assert coupling.direction[0] in pyd_hw.logical_connectivity
            assert (
                coupling.direction[1] in pyd_hw.logical_connectivity[coupling.direction[0]]
            )
            assert (
                pyd_hw.logical_connectivity_quality[coupling.direction]
                == coupling.quality / 100.0
            )

    def test_coupling_qualities_without_logical_connectivity(self):
        """Test that we can have coupling qualities without logical connectivity."""
        leg_hw = EchoModelLoader(qubit_count=4).load()
        leg_hw.qubit_direction_couplings = None
        pyd_hw = convert_purr_echo_hw_to_pydantic(leg_hw)

        # Without a defined logical connectivity, we assume the physical connectivity
        assert pyd_hw.logical_connectivity == pyd_hw.physical_connectivity

    def test_error_mitigation(self):
        leg_hw = EchoModelLoader(qubit_count=2).load()
        error_mitigation = ErrorMitigation(
            readout_mitigation=ReadoutMitigation(linear=self._LINEAR_MATRIX)
        )
        leg_hw.error_mitigation = error_mitigation
        pyd_hw = convert_purr_echo_hw_to_pydantic(leg_hw)

        assert pyd_hw.error_mitigation.readout_mitigation is not None
        linear_mitigation = pyd_hw.error_mitigation.readout_mitigation.linear
        assert len(linear_mitigation) == 2
        assert linear_mitigation[0] == np.array([[0.9, 0.2], [0.1, 0.8]])
        assert linear_mitigation[1] == np.array([[0.85, 0.25], [0.15, 0.75]])

    def test_error_mitigation_with_extra_values(self):
        """Regression test for invalid error mitigation configurations."""
        leg_hw = EchoModelLoader(qubit_count=1).load()
        error_mitigation = ErrorMitigation(
            readout_mitigation=ReadoutMitigation(
                linear=self._LINEAR_MATRIX,
            )
        )
        leg_hw.error_mitigation = error_mitigation
        pyd_hw = convert_purr_echo_hw_to_pydantic(leg_hw)

        assert pyd_hw.error_mitigation.readout_mitigation is not None
        linear_mitigation = pyd_hw.error_mitigation.readout_mitigation.linear
        assert len(linear_mitigation) == 1
        assert 0 in linear_mitigation

    def test_error_mitigation_with_string_indices(self):
        """Regression test for invalid error mitigation configurations with string
        indices."""

        leg_hw = EchoModelLoader(qubit_count=2).load()
        error_mitigation = ErrorMitigation(
            readout_mitigation=ReadoutMitigation(
                linear={str(k): v for k, v in self._LINEAR_MATRIX.items()},
            )
        )
        leg_hw.error_mitigation = error_mitigation
        pyd_hw = convert_purr_echo_hw_to_pydantic(leg_hw)

        assert pyd_hw.error_mitigation.readout_mitigation is not None
        linear_mitigation = pyd_hw.error_mitigation.readout_mitigation.linear
        assert len(linear_mitigation) == 2
        assert 0 in linear_mitigation
        assert 1 in linear_mitigation

    @pytest.mark.parametrize("num_qubits", _NUM_QUBITS)
    @pytest.mark.parametrize("seed", _SEEDS)
    def test_1q_pulse_channels(self, num_qubits, seed):
        pyd_hw, leg_hw = get_echo_hw_pair(num_qubits, seed=seed)

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

    @pytest.mark.parametrize("num_qubits", _NUM_QUBITS)
    @pytest.mark.parametrize("seed", _SEEDS)
    def test_2q_pulse_channels(self, num_qubits, seed):
        pyd_hw, leg_hw = get_echo_hw_pair(num_qubits, seed=seed)

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

    @pytest.mark.parametrize("num_qubits", _NUM_QUBITS)
    @pytest.mark.parametrize("seed", _SEEDS)
    def test_resonator_pulse_channels(self, num_qubits, seed):
        pyd_hw, leg_hw = get_echo_hw_pair(num_qubits, seed=seed)

        for qubit_index, qubit in pyd_hw.qubits.items():
            for pulse_channel_name in ["measure", "acquire"]:
                pyd_pulse_channel = getattr(
                    qubit.resonator.pulse_channels, pulse_channel_name
                )
                legacy_pulse_channel = leg_hw.get_qubit(
                    qubit_index
                ).measure_device.get_pulse_channel(channel_type_mapping[pulse_channel_name])
                validate_pulse_channel(pyd_pulse_channel, legacy_pulse_channel)

    @pytest.mark.parametrize("shape, width, amp", _PULSE_PARAMS)
    def test_x_pi_2_pulse(self, shape, width, amp):
        leg_hw = EchoModelLoader(qubit_count=1).load()
        leg_pulse = leg_hw.get_qubit(0).pulse_hw_x_pi_2
        leg_pulse["shape"] = shape
        leg_pulse["width"] = width
        leg_pulse["amp"] = amp
        pyd_hw = convert_purr_echo_hw_to_pydantic(leg_hw)

        pyd_pulse = pyd_hw.qubit_with_index(0).drive_pulse_channel.pulse
        validate_pulse_shapes(pyd_pulse, leg_pulse)

    @pytest.mark.parametrize("shape, width, amp", _PULSE_PARAMS)
    @pytest.mark.parametrize("direct_x_pi", [True, False])
    def test_x_pi_pulse(self, shape, width, amp, direct_x_pi):
        leg_hw = EchoModelLoader(qubit_count=1).load()
        qubit = leg_hw.get_qubit(0)
        leg_pulse = qubit.pulse_hw_x_pi
        leg_pulse["shape"] = shape
        leg_pulse["width"] = width
        leg_pulse["amp"] = amp
        qubit.direct_x_pi = direct_x_pi
        pyd_hw = convert_purr_echo_hw_to_pydantic(leg_hw)

        pyd_qubit = pyd_hw.qubit_with_index(0)
        pyd_pulse = pyd_qubit.drive_pulse_channel.pulse_x_pi
        assert pyd_qubit.direct_x_pi == direct_x_pi
        validate_pulse_shapes(pyd_pulse, leg_pulse)

    @pytest.mark.parametrize("shape, width, amp", _PULSE_PARAMS)
    def test_zx_pi_4_pulse(self, shape, width, amp):
        leg_hw = EchoModelLoader(qubit_count=2).load()
        leg_pulse = leg_hw.get_qubit(0).pulse_hw_zx_pi_4["Q1"]
        leg_pulse["shape"] = shape
        leg_pulse["width"] = width
        leg_pulse["amp"] = amp
        pyd_hw = convert_purr_echo_hw_to_pydantic(leg_hw)

        pyd_pulse = (
            pyd_hw.qubit_with_index(0).cross_resonance_pulse_channels[1].zx_pi_4_pulse
        )
        validate_pulse_shapes(pyd_pulse, leg_pulse)

    @pytest.mark.parametrize("shape, width, amp", _PULSE_PARAMS)
    def test_measure_pulse(self, shape, width, amp):
        leg_hw = EchoModelLoader(qubit_count=1).load()
        leg_pulse = leg_hw.get_qubit(0).pulse_measure
        leg_pulse["shape"] = shape
        leg_pulse["width"] = width
        leg_pulse["amp"] = amp
        pyd_hw = convert_purr_echo_hw_to_pydantic(leg_hw)

        pyd_pulse = pyd_hw.qubit_with_index(0).resonator.measure_pulse_channel.pulse
        validate_pulse_shapes(pyd_pulse, leg_pulse)

    @pytest.mark.parametrize(
        "delay, width, sync, use_weights, weights",
        [
            (16e-9, 32e-9, True, True, np.random.rand(32 * 8)),
            (0.0, 16e-9, False, False, None),
        ],
    )
    def test_measure_acquire(self, delay, width, sync, use_weights, weights):
        leg_hw = EchoModelLoader(qubit_count=1).load()
        qubit = leg_hw.get_qubit(0)
        qubit.measure_acquire = {
            "delay": delay,
            "width": width,
            "sync": sync,
            "use_weights": use_weights,
            "weights": weights,
        }
        pyd_hw = convert_purr_echo_hw_to_pydantic(leg_hw)

        pyd_acquire = pyd_hw.qubit_with_index(0).resonator.acquire_pulse_channel.acquire
        assert pyd_acquire.delay == delay
        assert pyd_acquire.width == width
        assert pyd_acquire.sync == sync
        assert pyd_acquire.use_weights == use_weights
        if use_weights:
            assert np.array_equal(pyd_acquire.weights, weights)
        else:
            assert np.array_equal(pyd_acquire.weights, np.array([]))

    @pytest.mark.parametrize("num_qubits", _NUM_QUBITS)
    @pytest.mark.parametrize("seed", _SEEDS)
    def test_physical_baseband(self, num_qubits, seed):
        pyd_hw, leg_hw = get_echo_hw_pair(num_qubits, seed=seed)

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

    @pytest.mark.parametrize("num_qubits", _NUM_QUBITS)
    @pytest.mark.parametrize("seed", _SEEDS)
    def test_physical_channel(self, num_qubits, seed):
        pyd_hw, leg_hw = get_echo_hw_pair(num_qubits, seed=seed)

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

    def test_physical_couplings_with_missing_cr_channels_logs_warning(self, caplog):
        # Setup a hardware model with a missing CR channel
        leg_hw = EchoModelLoader(qubit_count=2).load()
        leg_hw.qubit_direction_couplings = [
            coupling
            for coupling in leg_hw.qubit_direction_couplings
            if coupling.direction != (0, 1)
        ]
        qubit = leg_hw.qubits[0]
        cr_chan = leg_hw.qubits[0].get_cross_resonance_channel(leg_hw.qubits[1])
        # Strip away the qubit from the ID
        assert qubit.pulse_channels.pop(cr_chan.partial_id()[3:], None) is not None

        with caplog.at_level(LoggerLevel.WARNING.value):
            pyd_hw = convert_purr_echo_hw_to_pydantic(leg_hw)
            assert "The following physical couplings" in caplog.text
            assert "(0, 1)" in caplog.text
            assert 1 not in pyd_hw.physical_connectivity[0]

    def test_logical_couplings_with_missing_cr_channels_raises_warning(self):
        # Setup a hardware model with a missing CR channel
        leg_hw = EchoModelLoader(qubit_count=2).load()
        qubit = leg_hw.qubits[0]
        cr_chan = leg_hw.qubits[0].get_cross_resonance_channel(leg_hw.qubits[1])
        # Strip away the qubit from the ID
        assert qubit.pulse_channels.pop(cr_chan.partial_id()[3:], None) is not None
        assert (
            len(
                [
                    coupling
                    for coupling in leg_hw.qubit_direction_couplings
                    if coupling.direction == (0, 1)
                ]
            )
            == 1
        )

        with pytest.warns(match="The following logical couplings are present in the PuRR "):
            convert_purr_echo_hw_to_pydantic(leg_hw)

    def test_logical_couplings_with_missing_zx_pi_4_raises_warning(self):
        # Setup a hardware model with a missing ZX pi/4 pulse
        leg_hw = EchoModelLoader(qubit_count=2).load()
        qubit = leg_hw.qubits[0]
        assert qubit.pulse_hw_zx_pi_4.pop("Q1", None) is not None

        with pytest.warns(match="The following logical couplings are present in the PuRR "):
            convert_purr_echo_hw_to_pydantic(leg_hw)

    def test_error_mitigation_on_non_existent_qubits_logs_warning(self, caplog):
        leg_hw = EchoModelLoader(qubit_count=1).load()
        error_mitigation = ErrorMitigation(
            readout_mitigation=ReadoutMitigation(linear=self._LINEAR_MATRIX)
        )
        leg_hw.error_mitigation = error_mitigation

        with caplog.at_level(LoggerLevel.WARNING.value):
            convert_purr_echo_hw_to_pydantic(leg_hw)
            assert "The following qubits have linear readout mitigation" in caplog.text
            assert "1" in caplog.text
