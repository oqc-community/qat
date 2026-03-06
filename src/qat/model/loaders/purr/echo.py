# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from qat.model.loaders.purr.base import BaseLegacyModelLoader
from qat.purr.backends.echo import Connectivity, get_default_echo_hardware
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.utils.uuid import SeedType


class EchoModelLoader(BaseLegacyModelLoader):
    def __init__(
        self,
        qubit_count: int = 4,
        connectivity: Connectivity | list[(int, int)] | None = None,
        add_direction_couplings=True,
        random_seed: SeedType | bool | None = False,
    ):
        self.connectivity = connectivity
        self.qubit_count = qubit_count
        self.add_direction_couplings = add_direction_couplings
        self._random_seed = random_seed

    def load(self) -> QuantumHardwareModel:
        """Build the default Echo hardware model.

        :param seed: Seed passed to `get_default_echo_hardware`. Use `False` to omit
            the argument and keep that function's default seed. This differs from
            `None`, which is forwarded explicitly as `seed=None`.
        """
        return get_default_echo_hardware(
            qubit_count=self.qubit_count,
            connectivity=self.connectivity,
            add_direction_couplings=self.add_direction_couplings,
            **({"seed": self._random_seed} if self._random_seed is not False else {}),
        )
