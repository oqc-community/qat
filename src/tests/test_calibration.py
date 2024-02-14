# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd

import os
import pytest

from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.backends.live import LiveHardwareModel
from qat.purr.backends.qblox import QbloxControlHardware
from qat.purr.compiler.devices import Calibratable
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from src.tests.test_qblox_backend import setup_qblox_hardware_model


class TestCalibrations:
    def _get_qb_freq(self, hardware, qb=0):
        return hardware.get_qubit(qb).get_drive_channel().frequency

    def test_basic_calibration(self):
        echo: QuantumHardwareModel = get_default_echo_hardware()
        echo.is_calibrated = False
        before = Calibratable.load_calibration(echo.get_calibration())

        qubit = echo.get_qubit(0)
        drive = qubit.get_drive_channel()
        drive.frequency = drive.frequency + 100

        before_frequency = self._get_qb_freq(before)
        after_frequency = self._get_qb_freq(echo)
        assert before_frequency + 100 == after_frequency

    def custom_calibration_process(self, echo):
        qubit = echo.get_qubit(1)
        calibrated = qubit.is_calibrated
        if not calibrated:
            drive = qubit.get_drive_channel()
            drive.frequency = drive.frequency + 100
            qubit.is_calibrated = True

    def run_calibration_tests(self, hardware_model):
        hardware_model.is_calibrated = False
        before = Calibratable.load_calibration(hardware_model.get_calibration())
        self.custom_calibration_process(hardware_model)
        after = Calibratable.load_calibration(hardware_model.get_calibration())
        self.custom_calibration_process(hardware_model)
        after_again = hardware_model

        qubit = hardware_model.get_qubit(0)
        twobit = hardware_model.get_qubit(1)

        before_frequency = self._get_qb_freq(before, 1)
        after_frequency = self._get_qb_freq(after, 1)

        assert before_frequency + 100 == after_frequency
        assert self._get_qb_freq(after) == self._get_qb_freq(after_again)
        assert not qubit.is_calibrated
        assert twobit.is_calibrated

    def test_echo_calibration(self):
        echo_model: QuantumHardwareModel = get_default_echo_hardware()
        self.run_calibration_tests(echo_model)

    def test_qblox_calibration(self):
        instrument = QbloxControlHardware(name="live_cluster_mm")
        qblox_model = LiveHardwareModel()
        setup_qblox_hardware_model(qblox_model, instrument)
        self.run_calibration_tests(qblox_model)


class TestCalibrationSavingAndLoading:
    def setup_method(self, method) -> None:
        self.calibration_filename = (
            f"{method.__name__}_"
            f"{get_default_echo_hardware().__class__.__name__}.calibration.json"
        )

    def teardown_method(self, method) -> None:
        if os.path.exists(self.calibration_filename):
            os.remove(self.calibration_filename)

    def test_cyclic_references(self):
        echo = get_default_echo_hardware()
        copied_echo: QuantumHardwareModel = Calibratable.load_calibration(
            echo.get_calibration()
        )

        for key, value in copied_echo.pulse_channels.items():
            channel_key = key[:3]
            original_channel = copied_echo.physical_channels[channel_key]
            assert (
                id(value.physical_channel) == id(original_channel),
                "Copied references are different objects.",
            )

        assert len(echo.basebands) == len(copied_echo.basebands)
        assert len(echo.physical_channels) == len(copied_echo.physical_channels)
        assert len(echo.pulse_channels) == len(copied_echo.pulse_channels)
        assert len(echo.quantum_devices) == len(copied_echo.quantum_devices)
        assert len(echo.qubit_direction_couplings) == len(
            copied_echo.qubit_direction_couplings
        )

    @pytest.mark.parametrize("qubit_count", [4, 8, 35])
    def test_load_hardware_definition(self, qubit_count):
        echo = get_default_echo_hardware(qubit_count)
        original_calibration = echo.get_calibration()
        empty_hw = Calibratable.load_calibration(original_calibration)

        assert len(echo.qubits) == len(empty_hw.qubits)
        assert len(echo.resonators) == len(empty_hw.resonators)
        assert len(echo.quantum_devices) == len(empty_hw.quantum_devices)
        assert len(echo.basebands) == len(empty_hw.basebands)
        assert len(echo.physical_channels) == len(empty_hw.physical_channels)
