# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from qat.model.loaders.purr.base import BaseLegacyModelLoader
from qat.purr.backends.echo import Connectivity, get_default_echo_hardware
from qat.purr.compiler.hardware_models import QuantumHardwareModel


class EchoModelLoader(BaseLegacyModelLoader):
    def __init__(
        self,
        qubit_count: int = 4,
        connectivity: Connectivity | list[(int, int)] | None = None,
        add_direction_couplings=True,
    ):
        self.connectivity = connectivity
        self.qubit_count = qubit_count
        self.add_direction_couplings = add_direction_couplings

    def load(self) -> QuantumHardwareModel:
        return get_default_echo_hardware(
            qubit_count=self.qubit_count,
            connectivity=self.connectivity,
            add_direction_couplings=self.add_direction_couplings,
        )
