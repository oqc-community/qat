# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
import os
import unittest

from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler.devices import Calibratable
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.utils.logger import load_object_from_log_folder, save_object_to_log_folder


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

    def test_custom_calibration(self):
        echo: QuantumHardwareModel = get_default_echo_hardware()
        echo.is_calibrated = False
        before = Calibratable.load_calibration(echo.get_calibration())
        self.custom_calibration_process(echo)
        after = Calibratable.load_calibration(echo.get_calibration())
        self.custom_calibration_process(echo)
        after_again = echo

        qubit = echo.get_qubit(0)
        twobit = echo.get_qubit(1)

        before_frequency = self._get_qb_freq(before, 1)
        after_frequency = self._get_qb_freq(after, 1)

        assert before_frequency + 100 == after_frequency
        assert self._get_qb_freq(after) == self._get_qb_freq(after_again)
        assert not qubit.is_calibrated
        assert twobit.is_calibrated


class CalibrationSavingAndLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calibration_filename = (
            f"{self._testMethodName}_"
            f"{get_default_echo_hardware().__class__.__name__}.calibration.json"
        )

    def tearDown(self) -> None:
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
            self.assertEqual(
                id(value.physical_channel),
                id(original_channel),
                "Copied references are different objects."
            )

        assert len(echo.basebands) == len(copied_echo.basebands)
        assert len(echo.physical_channels) == len(copied_echo.physical_channels)
        assert len(echo.pulse_channels) == len(copied_echo.pulse_channels)
        assert len(echo.quantum_devices) == len(copied_echo.quantum_devices)
        assert len(echo.qubit_direction_couplings
                  ) == len(copied_echo.qubit_direction_couplings)

    def test_save_calibration(self):
        saved_path = save_object_to_log_folder(
            get_default_echo_hardware().get_calibration(), self.calibration_filename
        )
        assert os.path.exists(saved_path)

    def test_load_calibration(self):
        echo = get_default_echo_hardware()
        original_calibration = echo.get_calibration()
        saved_path = save_object_to_log_folder(
            original_calibration, self.calibration_filename
        )
        loaded_calibration = load_object_from_log_folder(self.calibration_filename)
        echo.load_calibration(loaded_calibration)
        assert os.path.exists(saved_path)
        assert original_calibration == echo.get_calibration()

    def test_load_hardware_definition(self):
        echo = get_default_echo_hardware()
        original_calibration = echo.get_calibration()
        empty_hw = Calibratable.load_calibration(original_calibration)

        assert len(echo.qubits) == len(empty_hw.qubits)
        assert len(echo.resonators) == len(empty_hw.resonators)
        assert len(echo.quantum_devices) == len(empty_hw.quantum_devices)
        assert len(echo.basebands) == len(empty_hw.basebands)
        assert len(echo.physical_channels) == len(empty_hw.physical_channels)

    def test_load_calibration_for_only_one_qubit(self):
        echo = get_default_echo_hardware()
        qubit1 = echo.get_qubit(0)
        qubit2 = echo.get_qubit(1)

        drive0 = qubit1.get_drive_channel()
        drive0_frequency = drive0.frequency
        drive1 = qubit2.get_drive_channel()
        drive1_frequency = drive1.frequency
        save_object_to_log_folder(
            echo.get_qubit(0).get_calibration(), self.calibration_filename
        )

        drive1.frequency += 200
        loaded_calibration = Calibratable.load_calibration(
            load_object_from_log_folder(self.calibration_filename)
        )
        echo.quantum_devices["Q0"] = loaded_calibration

        assert drive0_frequency == drive0.frequency
        assert drive1_frequency != drive1.frequency
