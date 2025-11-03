# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from collections import defaultdict
from enum import Enum
from random import Random

from qat.model.builder import PhysicalHardwareModelBuilder
from qat.model.hardware_model import PhysicalHardwareModel
from qat.model.loaders.base import BasePhysicalModelLoader


class LucyCouplingDirection(Enum):
    """Enum for the coupling direction between qubits on a Lucy model."""

    LEFT = "left"
    RIGHT = "right"
    RANDOM = "random"


class LucyCouplingQuality(Enum):
    """Describes the coupling quality settings for a Lucy model."""

    UNIFORM = "uniform"
    RANDOM = "random"


class LucyModelLoader(BasePhysicalModelLoader):
    """Factory for creating mock Lucy hardware models. That is, a model of QPUs that have
    qubits arranged on a ring.

    The direction of couplings can be specified to go in either direction, or have each
    coupling choosen randomly. The coupling quality can be set to uniform, or random, or
    custom qualities can be provided as a dictionary mapping pairs of qubits to their
    coupling quality. Furthermore, some properties of the pulse channels can be provided.

    :param qubit_count: The number of qubits in the model. Default is 8.
    :param coupling_direction: The direction of the coupling between qubits. Default is
        `LucyCouplingDirection.RIGHT`.
    :param coupling_qualities: The coupling quality settings for the model. Default is
        `LucyCouplingQuality.UNIFORM`. Can also be a dictionary mapping pairs of qubits
        to their coupling quality.
    :param qubit_baseband_frequency: The baseband frequency for the qubits. Default is
        5.5 GHz.
    :param resonator_baseband_frequency: The baseband frequency for the resonators. Default
        is 8.5 GHz.
    :param qubit_baseband_if_frequency: The IF frequency for the qubits. Default is 250 MHz.
    :param resonator_baseband_if_frequency: The IF frequency for the resonators. Default
        is 250 MHz.
    :param drive_frequency: The frequency of the drive pulse channels. Default is 5.5 GHz.
    :param measure_frequency: The frequency of the measure pulse channels. Default is
        8.5 GHz.
    :param cr_scale: The scale factor for the cross-resonance pulse channels. Default is
        50.0.
    :param crc_scale: The scale factor for the cross-resonance cancellation pulse channels.
        Default is 0.0.
    :param random_seed: An optional seed for the random number generator to ensure
        reproducibility.
    :param start_index: The starting index for the qubits; useful for testing systems with
        non-standard indexing conventions. Default is 0.
    """

    def __init__(
        self,
        qubit_count: int = 8,
        coupling_direction: LucyCouplingDirection = LucyCouplingDirection.RIGHT,
        coupling_qualities: LucyCouplingQuality
        | dict[tuple[int, int], float] = LucyCouplingQuality.UNIFORM,
        qubit_baseband_frequency: float = 5.5e9,
        resonator_baseband_frequency: float = 8.5e9,
        qubit_baseband_if_frequency: float = 250e6,
        resonator_baseband_if_frequency: float = 250e6,
        drive_frequency: float = 5.5e9,
        measure_frequency: float = 8.5e9,
        cr_scale: float = 50.0,
        crc_scale: float = 0.0,
        random_seed: int | None = None,
        start_index: int = 0,
    ):
        self.qubit_count = qubit_count
        self.coupling_direction = coupling_direction
        self.coupling_qualities = coupling_qualities

        self.qubit_baseband_frequency = qubit_baseband_frequency
        self.resonator_baseband_frequency = resonator_baseband_frequency
        # TODO: remove if frequency? (COMPILER-714)
        self.qubit_baseband_if_frequency = qubit_baseband_if_frequency
        self.resonator_baseband_if_frequency = resonator_baseband_if_frequency
        self.drive_frequency = drive_frequency
        self.measure_frequency = measure_frequency
        self.cr_scale = cr_scale
        self.crc_scale = crc_scale
        self.start_index = start_index

        self.random_seed = random_seed
        self._random = Random(random_seed) if random_seed is not None else Random()

    def load(self) -> PhysicalHardwareModel:
        self._reset_random()
        model = self._make_uncalibrated_model()
        model = self._populate_physical_channels(model)
        return self._populate_pulse_channels(model)

    def _reset_random(self):
        """Resets the random number generator with the provided seed."""
        if self.random_seed is not None:
            self._random = Random(self.random_seed)
        else:
            self._random = Random()

    def _generate_physical_connectivity(self) -> dict[int, set[int]]:
        """Physical connectivity is bidirectional."""

        coupling_map = dict()
        for i in range(self.qubit_count):
            left_neighbor = self.start_index + (i - 1) % self.qubit_count
            right_neighbor = self.start_index + (i + 1) % self.qubit_count
            coupling_map[i + self.start_index] = {left_neighbor, right_neighbor}
        return coupling_map

    def _generate_logical_connectivity(self) -> dict[int, set[int]]:
        """Creates a logical connectivity that is unidirectional, with the direction decided
        by the coupling direction."""

        connectivity: dict[int, set[int]] = defaultdict(set)
        for i in range(self.qubit_count):
            left_neighbor = self.start_index + i
            right_neighbor = self.start_index + (i + 1) % self.qubit_count

            if self.coupling_direction == LucyCouplingDirection.RANDOM:
                direction = self._random.choice(
                    [LucyCouplingDirection.LEFT, LucyCouplingDirection.RIGHT]
                )
            else:
                direction = self.coupling_direction

            if direction == LucyCouplingDirection.LEFT:
                connectivity[right_neighbor].add(left_neighbor)
            elif direction == LucyCouplingDirection.RIGHT:
                connectivity[left_neighbor].add(right_neighbor)
        return connectivity

    def _order_custom_coupling_qualities(
        self, connectivity: dict[int, set[int]], qualities: dict[tuple[int, int], float]
    ) -> dict[tuple[int, int], float]:
        """Orders the coupling qualities according to the connectivity of the model."""

        ordered_qualities: dict[tuple[int, int], float] = {}
        for qubit1, qubit2 in qualities.keys():
            if qubit1 in connectivity and qubit2 in connectivity[qubit1]:
                ordered_qualities[(qubit1, qubit2)] = qualities[(qubit1, qubit2)]
            elif qubit2 in connectivity and qubit1 in connectivity[qubit2]:
                ordered_qualities[(qubit2, qubit1)] = qualities[(qubit1, qubit2)]
            else:
                raise ValueError(
                    f"Coupling quality for ({qubit1}, {qubit2}) does not match the ring "
                    "connectivity in Lucy models."
                )
        return ordered_qualities

    def _generate_coupling_qualities(
        self, connectivity: dict[int, set[int]]
    ) -> dict[tuple[int, int], float]:
        """Generates coupling qualities for the model, given the coupling quality type."""

        qualities = self.coupling_qualities
        if not isinstance(qualities, LucyCouplingQuality):
            return self._order_custom_coupling_qualities(connectivity, qualities)

        if qualities == LucyCouplingQuality.UNIFORM:
            return {
                (qubit1, qubit2): 1.0
                for qubit1, connections in connectivity.items()
                for qubit2 in connections
            }
        elif qualities == LucyCouplingQuality.RANDOM:
            return {
                (qubit1, qubit2): self._random.random()
                for qubit1, connections in connectivity.items()
                for qubit2 in connections
            }

    def _build_model(self, physical_connectivity, logical_connectivity, coupling_qualities):
        """Uses the connectivity and coupling qualities to build the model. Intentionally
        uses some indirection here so this can be adapted to other builders."""

        return PhysicalHardwareModelBuilder(
            physical_connectivity=physical_connectivity,
            logical_connectivity=logical_connectivity,
            logical_connectivity_quality=coupling_qualities,
        ).model

    def _make_uncalibrated_model(self) -> PhysicalHardwareModel:
        """Builds an uncalibrated hardware model based on the Lucy lattice."""

        physical_connectivity = self._generate_physical_connectivity()
        logical_connectivity = self._generate_logical_connectivity()
        coupling_qualities = self._generate_coupling_qualities(logical_connectivity)
        return self._build_model(
            physical_connectivity, logical_connectivity, coupling_qualities
        )

    def _populate_physical_channels(
        self, model: PhysicalHardwareModel
    ) -> PhysicalHardwareModel:
        """Populates the model with baseband frequencies for qubits and resonators."""

        for qubit in model.qubits.values():
            baseband = qubit.physical_channel.baseband
            baseband.frequency = self.qubit_baseband_frequency
            # TODO: remove if frequency? (COMPILER-714)
            baseband.if_frequency = self.qubit_baseband_if_frequency

            baseband = qubit.resonator.physical_channel.baseband
            baseband.frequency = self.resonator_baseband_frequency
            # TODO: remove if frequency? (COMPILER-714)
            baseband.if_frequency = self.resonator_baseband_if_frequency
        return model

    def _populate_pulse_channels(
        self, model: PhysicalHardwareModel
    ) -> PhysicalHardwareModel:
        """Populates the model with pulse channels and their frequencies and sclae factors."""

        for qubit in model.qubits.values():
            qubit.drive_pulse_channel.frequency = self.drive_frequency
            qubit.second_state_pulse_channel.frequency = 0.0
            qubit.freq_shift_pulse_channel.frequency = 0.0
            for channel in qubit.cross_resonance_pulse_channels.values():
                channel.frequency = self.drive_frequency
                channel.scale = self.cr_scale

            for channel in qubit.cross_resonance_cancellation_pulse_channels.values():
                channel.frequency = self.drive_frequency
                channel.scale = self.crc_scale

            qubit.measure_pulse_channel.frequency = self.measure_frequency
            qubit.acquire_pulse_channel.frequency = self.measure_frequency

        return model
