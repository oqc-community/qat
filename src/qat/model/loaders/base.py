# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import abc

from qat.model.hardware_model import PydLogicalHardwareModel, PydPhysicalHardwareModel
from qat.purr.compiler.hardware_models import LegacyHardwareModel


class BaseModelLoader(abc.ABC):
    """ModelLoaders load HardwareModels from a source configured on initialisation."""

    def __init__(self): ...

    @abc.abstractmethod
    def load(self) -> LegacyHardwareModel | PydLogicalHardwareModel:
        """Load and return the Hardware Model.

        :return: A loaded Hardware Model
        :rtype: LegacyHardwareModel | LogicalHardwareModel
        """
        pass


class BaseLogicalModelLoader(BaseModelLoader):
    @abc.abstractmethod
    def load(self) -> PydPhysicalHardwareModel:
        pass


class BasePhysicalModelLoader(BaseModelLoader):
    @abc.abstractmethod
    def load(self) -> PydPhysicalHardwareModel:
        pass
