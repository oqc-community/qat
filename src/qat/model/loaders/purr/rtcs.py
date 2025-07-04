# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from qat.model.loaders.purr.base import BaseLegacyModelLoader
from qat.purr.backends.realtime_chip_simulator import get_default_RTCS_hardware
from qat.purr.compiler.hardware_models import HardwareModel as LegacyHardwareModel


class RTCSModelLoader(BaseLegacyModelLoader):
    def __init__(self, rotating_frame: bool = True):
        self.rotating_frame = rotating_frame

    def load(self) -> LegacyHardwareModel:
        return get_default_RTCS_hardware(rotating_frame=self.rotating_frame)
