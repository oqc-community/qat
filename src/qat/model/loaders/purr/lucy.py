# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from qat.model.loaders.purr.base import BaseLegacyModelLoader
from qat.purr.backends.live import build_lucy_hardware
from qat.purr.compiler.hardware_models import QuantumHardwareModel


class LucyModelLoader(BaseLegacyModelLoader):
    def load(self) -> QuantumHardwareModel:
        return build_lucy_hardware(QuantumHardwareModel())
