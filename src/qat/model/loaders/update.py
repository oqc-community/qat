# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from abc import abstractmethod

from qat.model.hardware_model import PhysicalHardwareModel
from qat.purr.compiler.hardware_models import QuantumHardwareModel


class ModelUpdateChecker:
    """A mixin for model loaders that adds an interface for checking if models are
    up-to-date for models that are expected to evolve within its lifetime."""

    @abstractmethod
    def is_up_to_date(self, model: PhysicalHardwareModel | QuantumHardwareModel) -> bool:
        """Used to check if the model is up-to-date. To be implemented by subclasses."""
        ...
