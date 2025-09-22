# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from pathlib import Path

from qat.model.loaders.purr.base import BaseLegacyModelLoader
from qat.purr.backends.qblox.live import QbloxLiveHardwareModel
from qat.purr.compiler.hardware_models import (
    QuantumHardwareModel as LegacyQuantumHardwareModel,
)
from qat.utils.hardware_model import hash_calibration_file


class FileModelLoader(BaseLegacyModelLoader):
    def __init__(self, path: Path | str):
        self.path: Path = Path(path).expanduser()

    def load(self) -> LegacyQuantumHardwareModel:
        hw = LegacyQuantumHardwareModel.load_calibration_from_file(str(self.path))
        hw.calibration_id = hash_calibration_file(str(self.path))
        return hw


class QbloxFileModelLoader(FileModelLoader):
    """
    File loader for the open source QbloxLiveHardwareModel
    """

    def load(self) -> QbloxLiveHardwareModel:
        model = QbloxLiveHardwareModel().load_calibration_from_file(self.path)
        model.calibration_id = hash_calibration_file(self.path)
        model._reverse_coercion()  ## Required to revert strings into integers
        return model
