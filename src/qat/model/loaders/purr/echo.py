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

        The ``random_seed`` value provided at initialization is forwarded to
        :func:`get_default_echo_hardware` as its ``seed`` argument, except when
        it is the boolean value ``False``. In that case, the ``seed`` argument is
        omitted entirely. This differs from passing ``None``, which is forwarded
        as ``seed=None`` and allows the callee to apply its own default seeding
        behaviour.
        """
        return get_default_echo_hardware(
            qubit_count=self.qubit_count,
            connectivity=self.connectivity,
            add_direction_couplings=self.add_direction_couplings,
            **({"seed": self._random_seed} if self._random_seed is not False else {}),
        )
