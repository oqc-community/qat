# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from qiskit_aer.noise import NoiseModel

from qat.model.loaders.purr.base import BaseLegacyModelLoader
from qat.purr.backends.echo import Connectivity
from qat.purr.backends.qiskit_simulator import (
    QiskitHardwareModel as LegacyQiskitHardwareModel,
)
from qat.purr.backends.qiskit_simulator import get_default_qiskit_hardware


class QiskitModelLoader(BaseLegacyModelLoader):
    def __init__(
        self,
        qubit_count: int = 20,
        noise_model: NoiseModel | None = None,
        strict_placement: bool = True,
        connectivity: Connectivity | list[(int, int)] | None = None,
    ):
        self.qubit_count = qubit_count
        self.noise_model = noise_model
        self.strict_placement = strict_placement
        self.connectivity = connectivity

    def load(self) -> LegacyQiskitHardwareModel:
        return get_default_qiskit_hardware(
            qubit_count=self.qubit_count,
            noise_model=self.noise_model,
            strict_placement=self.strict_placement,
            connectivity=self.connectivity,
        )
