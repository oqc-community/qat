# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from qat.model.hardware_model import PydLogicalHardwareModel, PydPhysicalHardwareModel
from qat.purr.compiler.hardware_models import LegacyHardwareModel

LegHwModel = TypeVar("LegHwModel", bound=LegacyHardwareModel)

PydHwModel = TypeVar("PydHwModel", PydPhysicalHardwareModel, PydLogicalHardwareModel)

HwModel = TypeVar(
    "HwModel", LegacyHardwareModel, PydLogicalHardwareModel, PydPhysicalHardwareModel
)


class BaseModelLoader(Generic[HwModel], ABC):
    """ModelLoaders load HardwareModels from a source configured on initialisation."""

    def __init__(self): ...

    @abstractmethod
    def load(self) -> HwModel:
        """Load and return the Hardware Model.

        :return: A loaded Hardware Model
        :rtype: LegacyHardwareModel | LogicalHardwareModel
        """
        pass


class BaseLogicalModelLoader(BaseModelLoader[PydLogicalHardwareModel], ABC): ...


class BasePhysicalModelLoader(BaseModelLoader[PydPhysicalHardwareModel], ABC): ...
