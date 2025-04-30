# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from pathlib import Path

from qat.model.hardware_model import PhysicalHardwareModel
from qat.model.loaders.base import BasePhysicalModelLoader


class FileModelLoader(BasePhysicalModelLoader):
    def __init__(self, path: Path | str):
        self.path: Path = Path(path).expanduser()

    def load(self) -> PhysicalHardwareModel:
        blob = self.path.read_text()
        return PhysicalHardwareModel.model_validate_json(blob)
