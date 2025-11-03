# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from collections import defaultdict
from random import Random

from qat.model.convert_purr import convert_purr_echo_hw_to_pydantic
from qat.model.hardware_model import PhysicalHardwareModel
from qat.model.loaders.base import BasePhysicalModelLoader
from qat.purr.backends.echo import Connectivity
from qat.purr.compiler.hardware_models import ErrorMitigation
from qat.utils.hardware_model import (
    apply_setup_to_echo_hardware,
    ensure_connected_connectivity,
    random_connectivity,
    random_quality_map,
)

from .purr.echo import EchoModelLoader as LegacyEchoModelLoader


class EchoModelLoader(BasePhysicalModelLoader):
    def __init__(
        self,
        qubit_count: int = 4,
        connectivity: Connectivity | list[(int, int)] | None = None,
        error_mitigation: ErrorMitigation | None = None,
    ):
        self._legacy = LegacyEchoModelLoader(
            qubit_count=qubit_count, connectivity=connectivity
        )
        self._error_mitigation = error_mitigation

    def load(self) -> PhysicalHardwareModel:
        legacy_model = self._legacy.load()
        legacy_model.error_mitigation = self._error_mitigation
        return convert_purr_echo_hw_to_pydantic(legacy_model)


class JaggedEchoModelLoader(BasePhysicalModelLoader):
    def __init__(
        self,
        qubit_count: int = 4,
        qubit_indices: list[int] = None,
        connectivity: list[(int, int)] | None = None,
        error_mitigation: ErrorMitigation | None = None,
        random_seed: int | None = None,
    ):
        """Load a converted Echo hardware model with non-sequential physical qubit indices.

        :param qubit_count: Number of physical qubits to select.
        :param qubit_indices: Specific indices to select from, if used in combination
            with `connectivity` these indices must all be in the connectivity keys.
            Defaults to a selection of `qubit_count` indices in `connectivity` if
            provided else selected from a generated :func:`random_connectivity`.
        :param connectivity: List of tuples representing the qubit connections. Defaults
            to a genetared :func:`random_connectivity` including `qubit_indices` if
            provided.
        :param error_mitigation: Error mitigation settings to apply to the model.
        :param random_seed: Seed for random connectivity and quality map generation.
        """
        if isinstance(connectivity, Connectivity):
            raise NotImplementedError(
                "JaggedEchoModelLoader does not support Connectivity type."
            )

        random = Random(random_seed)
        if connectivity is None:
            if qubit_indices is None:
                n = qubit_count + 5
            else:
                n = max(qubit_indices) + 1
            connectivity = random_connectivity(n, seed=random)
        else:
            connectivity_dict = defaultdict(set)
            for qubit, coupled in connectivity:
                connectivity_dict[qubit].add(coupled)
            connectivity = connectivity_dict

        available_indices = set(connectivity)

        if qubit_indices is None:
            qubit_indices = random.sample(list(available_indices), qubit_count)
        else:
            if not (qi_set := set(qubit_indices)) <= available_indices:
                missing = qi_set - available_indices
                raise ValueError(
                    f"Incompatible connectivity and qubit_indices, chosen indices {missing}"
                    " are not in connectivity list."
                )

        new_connectivity = ensure_connected_connectivity(connectivity, qubit_indices)
        self._quality_map = random_quality_map(
            new_connectivity, min_quality=1, max_quality=100, seed=random
        )
        self._qubit_indices = set(qubit_indices)
        self._qubit_count = len(qubit_indices)
        self._error_mitigation = error_mitigation

    def load(self) -> PhysicalHardwareModel:
        legacy_model = apply_setup_to_echo_hardware(
            self._qubit_count,
            connectivity=self._quality_map,
            qubit_indices=self._qubit_indices,
        )
        legacy_model.error_mitigation = self._error_mitigation
        return convert_purr_echo_hw_to_pydantic(legacy_model)


PydEchoModelLoader = EchoModelLoader
