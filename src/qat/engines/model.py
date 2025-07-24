# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from abc import ABC, abstractmethod

from qat.model.hardware_model import PhysicalHardwareModel
from qat.purr.compiler.execution import QuantumExecutionEngine
from qat.purr.compiler.hardware_models import QuantumHardwareModel


class RequiresHardwareModelMixin(ABC):
    """Specifies a hardware model requirement for a :class:`NativeEngine`. To be used as a
    mixin for engines that require a hardware model to be set.

    Engines might require a hardware model to be set to execute. This might be because the
    hardware model contains information about the control hardware of the target device
    (for the case of legacy hardware models), or because the engine is a simulator that
    makes use of the calibration data.
    """

    @property
    def model(self) -> PhysicalHardwareModel | QuantumHardwareModel:
        return self._model

    @model.setter
    def model(self, model: PhysicalHardwareModel | QuantumHardwareModel):
        """Uses indirection for the model setter, so the child class can implement its
        own logic for updating the model."""

        if self._model != model:
            self._update_model(model)

    @abstractmethod
    def _update_model(self, model: PhysicalHardwareModel | QuantumHardwareModel):
        """Method to update the model in the engine. This is to be implemented by the child
        class, as changing the model might have different implications."""
        ...


def requires_hardware_model(engine) -> bool:
    """Checks if an engine requires a hardware model. This is used to allow backwards
    compatibility with purr engines, but should be removed in the future."""

    # TODO: Yeet this function in the bin for v4 (COMPILER-662)

    if isinstance(engine, RequiresHardwareModelMixin):
        return True
    if isinstance(engine, QuantumExecutionEngine) and engine.model is not None:
        return True
    return False
