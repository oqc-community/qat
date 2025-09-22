# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from abc import ABC

from qat.model.loaders.base import BaseModelLoader
from qat.purr.compiler.hardware_models import LegacyHardwareModel


class BaseLegacyModelLoader(BaseModelLoader[LegacyHardwareModel], ABC): ...
