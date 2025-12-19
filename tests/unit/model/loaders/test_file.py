# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import os

import pytest

from qat.model.loaders.purr.file import QbloxFileModelLoader
from qat.purr.backends.qblox.live import QbloxLiveHardwareModel


class TestFileModelLoader:
    def test_load_model(self, testpath):
        path = str(testpath / "files/calibrations/qblox_calibration.json")
        loader = QbloxFileModelLoader(path)
        model = loader.load()
        assert isinstance(model, QbloxLiveHardwareModel)
        assert model.calibration_id == "ec25e471f50c62f437f6e96b44f8d2c5"
        assert len(model.calibration_id) == 32

    @pytest.mark.parametrize("qblox_model", [None], indirect=True)
    def test_save_model(self, qblox_model, tmp_path):
        filename = os.path.join(tmp_path, "qblox_calibration.json")
        assert not os.path.exists(filename)
        qblox_model.save_calibration_to_file(filename)
        assert os.path.exists(filename)
