# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
import os

import pytest

from qat.purr.backends.qblox.live import QbloxLiveHardwareModel
from qat.purr.compiler.devices import Calibratable
from qat.purr.utils.logger import load_object_from_log_folder, save_object_to_log_folder


@pytest.mark.parametrize("cluster_setup", [None], indirect=True)
class TestHardwareModelLifecycle:
    def test_save_calibration(self, cluster_setup, calibration_filename):
        model = QbloxLiveHardwareModel()
        cluster_setup.configure(model)

        saved_path = save_object_to_log_folder(
            model.get_calibration(), calibration_filename
        )
        assert os.path.exists(saved_path)

    def test_load_calibration(self, cluster_setup, calibration_filename):
        model = QbloxLiveHardwareModel()
        cluster_setup.configure(model)

        original_calibration = model.get_calibration()
        saved_path = save_object_to_log_folder(original_calibration, calibration_filename)
        assert os.path.exists(saved_path)

        loaded_calibration = load_object_from_log_folder(calibration_filename)
        assert original_calibration == loaded_calibration
        model.load_calibration(loaded_calibration)
        assert original_calibration == model.get_calibration()

    def _get_qb_freq(self, model, qubit):
        return model.get_qubit(qubit).get_drive_channel().frequency

    def _apply_changes(self, model):
        qubit = model.get_qubit(1)
        calibrated = qubit.is_calibrated
        if not calibrated:
            drive = qubit.get_drive_channel()
            drive.frequency = drive.frequency + 100
            qubit.is_calibrated = True

    def _run_calibration_tests(self, model):
        model.is_calibrated = False
        before = Calibratable.load_calibration(model.get_calibration())
        self._apply_changes(model)
        after = Calibratable.load_calibration(model.get_calibration())

        qubit0 = model.get_qubit(0)
        qubit1 = model.get_qubit(1)

        after_frequency_0 = self._get_qb_freq(after, 0)
        before_frequency_0 = self._get_qb_freq(before, 0)
        assert not qubit0.is_calibrated
        assert after_frequency_0 == before_frequency_0

        before_frequency_1 = self._get_qb_freq(before, 1)
        after_frequency_1 = self._get_qb_freq(after, 1)
        assert qubit1.is_calibrated
        assert before_frequency_1 + 100 == after_frequency_1

    def test_qblox_calibration(self, cluster_setup):
        model = QbloxLiveHardwareModel()
        cluster_setup.configure(model)
        self._run_calibration_tests(model)
