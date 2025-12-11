# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import random

from qat.model.loaders.purr.base import BaseLegacyModelLoader
from qat.model.loaders.purr.echo import EchoModelLoader
from qat.model.loaders.update import ModelUpdateChecker
from qat.purr.compiler.hardware_models import (
    ErrorMitigation,
    QuantumHardwareModel,
    ReadoutMitigation,
)


class BrokenLoader(EchoModelLoader):
    """ModelLoaders load HardwareModels from a source configured on initialisation."""

    def __init__(self, on_init: bool = False, on_load: bool = False, *args, **kwargs):
        self.on_init = on_init
        self.on_load = on_load

        if self.on_init:
            raise ValueError("This loader is broken intentionally on init.")

        super().__init__(*args, **kwargs)

    def load(self) -> QuantumHardwareModel:
        """Load and return the Hardware Model.

        :return: A loaded Hardware Model
        :rtype: LegacyHardwareModel | LogicalHardwareModel
        """
        if self.on_load:
            raise ValueError("This loader is broken intentionally on load.")
        return super().load()


class MockModelLoader(BaseLegacyModelLoader, ModelUpdateChecker):
    """A mock model loader used to test the UpdateablePipeline infrastructure. Each load
    will add an extra qubit."""

    def __init__(self, num_qubits: int = 1):
        self.num_qubits = num_qubits

    def load(self):
        self.num_qubits += 1
        return EchoModelLoader(qubit_count=self.num_qubits).load()

    def is_up_to_date(self, model) -> bool:
        return len(model.qubits) == self.num_qubits


class EchoModelLoaderWithErrorMitigation(EchoModelLoader):
    """A model loader that applies a random error mitigation configuration to the loaded
    model."""

    def load(self, seed=None):
        """Load the Echo model and apply error mitigation."""
        hw = super().load()
        return self.apply_error_mitigation_setup(hw, seed=seed)

    def generate_random_linear(self, qubit_indices, random_data=True, seed=None):
        if seed is not None:
            random.seed(seed)
        output = {}
        for index in qubit_indices:
            random_0 = random.random() if random_data else 1
            random_1 = random.random() if random_data else 1
            output[str(index)] = {
                "0|0": random_0,
                "1|0": 1 - random_0,
                "1|1": random_1,
                "0|1": 1 - random_1,
            }
        return output

    def apply_error_mitigation_setup(self, hw, seed=None):
        lin_mit = self.generate_random_linear([q.index for q in hw.qubits], seed=seed)
        hw.error_mitigation = ErrorMitigation(ReadoutMitigation(linear=lin_mit))
        return hw
