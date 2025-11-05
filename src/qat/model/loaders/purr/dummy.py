# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.model.loaders.purr.base import BaseLegacyModelLoader
from qat.purr.backends.echo import Connectivity
from qat.purr.backends.qblox.dummy import (
    get_default_dummy_hardware,
)


class QbloxDummyModelLoader(BaseLegacyModelLoader):
    def __init__(
        self,
        name: str = None,
        dummy_cfg: dict = None,
        qubit_count: int = 4,
        connectivity: Connectivity | list[(int, int)] | None = None,
        add_direction_couplings=True,
    ):
        self.name = name
        self.dummy_cfg = dummy_cfg
        self.connectivity = connectivity
        self.qubit_count = qubit_count
        self.add_direction_couplings = add_direction_couplings

    def load(self):
        return get_default_dummy_hardware(
            qubit_count=self.qubit_count,
            connectivity=self.connectivity,
            add_direction_couplings=self.add_direction_couplings,
            name=self.name,
            dummy_cfg=self.dummy_cfg,
        )
