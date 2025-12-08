# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
from __future__ import annotations

from copy import deepcopy

from pydantic import Field, field_validator, model_validator
from pydantic_extra_types.semantic_version import SemanticVersion
from semver import Version

from qat.model.device import Qubit, Resonator
from qat.model.error_mitigation import ErrorMitigation
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.utils.logger import get_default_logger
from qat.utils.pydantic import (
    CalibratableUnitInterval,
    FrozenDict,
    FrozenSet,
    NoExtraFieldsModel,
    QubitCoupling,
    QubitId,
)
from qat.utils.uuid import uuid4

VERSION = Version(0, 0, 1)
log = get_default_logger()


class LogicalHardwareModel(NoExtraFieldsModel):
    """Models a hardware with a given connectivity.

    :param version: Semantic version of the hardware model.
    :param logical_connectivity: Connectivity of the qubits in the hardware model (directed graph).
    """

    version: SemanticVersion = Field(frozen=True, repr=False, default=VERSION)
    logical_connectivity: FrozenDict[QubitId, FrozenSet[QubitId]]

    calibration_id: str = Field(default_factory=lambda: str(uuid4()), frozen=True)

    @field_validator("version")
    def version_compatibility(version: Version):
        assert version.major == VERSION.major, (
            f"Direct instantiation requires major version compatibility (expected {VERSION.major}.Y.Z, found {version})"
        )
        assert VERSION >= version, (
            f"Latest supported hardware model version {VERSION}, found {version}"
        )
        return VERSION

    def __eq__(self, other: LogicalHardwareModel) -> bool:
        if type(self) is not type(other):
            return False

        if self.__class__.model_fields != other.__class__.model_fields:
            return False

        if self.version != other.version:
            return False

        if self.logical_connectivity != other.logical_connectivity:
            return False

        return True

    def __ne__(self, other: LogicalHardwareModel) -> bool:
        return not self.__eq__(other)


class PhysicalHardwareModel(LogicalHardwareModel):
    """Class for calibrating our QPU hardware.

    :param qubits: The superconducting qubits on the chip.
    :param physical_connectivity: The connectivities of the physical qubits on the QPU (undirected graph).
    :param logical_connectivity: The connectivities (directed graph) of the qubits used for compilation, which can be a subgraph of `physical_connectivity`.
    :param logical_connectivity_quality: Quality of the connections between the qubits.
    :param error_mitigation: Error mitigation strategy for this hardware model.
    """

    qubits: FrozenDict[QubitId, Qubit]
    physical_connectivity: FrozenDict[QubitId, FrozenSet[QubitId]] = Field(frozen=True)
    logical_connectivity: FrozenDict[QubitId, FrozenSet[QubitId]] | None = Field(
        default=None
    )
    logical_connectivity_quality: FrozenDict[QubitCoupling, CalibratableUnitInterval]
    error_mitigation: ErrorMitigation = ErrorMitigation()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._generate_mappings()

    def _generate_mappings(self):
        self._ids_to_pulse_channels = {}
        self._ids_to_physical_channels = {}
        self._pulse_channel_ids_to_physical_channel = {}
        self._pulse_channel_ids_to_device = {}
        self._qubits_to_qubit_ids = {}
        self._resonators_to_qubits = {}
        self._physical_channel_map = {}
        self._physical_channel_ids_to_qubit = {}
        self._physical_channel_ids_to_resonator = {}

        for qubit_id, qubit in self.qubits.items():
            self._qubits_to_qubit_ids[qubit] = qubit_id
            self._resonators_to_qubits[qubit.resonator] = qubit
            self._physical_channel_to_device_mapping(qubit)

            for device in [qubit, qubit.resonator]:
                phys_channel = device.physical_channel
                self._ids_to_physical_channels[phys_channel.uuid] = phys_channel
                self._physical_channel_map[phys_channel.name_index] = phys_channel

                for pulse_channel in device.all_pulse_channels:
                    self._ids_to_pulse_channels[pulse_channel.uuid] = pulse_channel
                    self._pulse_channel_ids_to_physical_channel[pulse_channel.uuid] = (
                        phys_channel
                    )
                    self._pulse_channel_ids_to_device[pulse_channel.uuid] = device

    def _physical_channel_to_device_mapping(self, qubit: Qubit):
        """
        populated the mapping from physical channel ids to devices (qubits and resonators)
        in the hardware model.
        The qubit mapping will also map the resonator physical channels to the qubit connected to
        the resonator.
        While the resonator mapping will only map the physical channels to the resonator.
        """
        self._physical_channel_ids_to_qubit.update(
            {
                qubit.physical_channel.uuid: qubit,
                qubit.resonator.physical_channel.uuid: qubit,
            }
        )

        self._physical_channel_ids_to_resonator.update(
            {qubit.resonator.physical_channel.uuid: qubit.resonator}
        )

    @model_validator(mode="before")
    def default_logical_connectivity(cls, data):
        if not data.get("logical_connectivity", None):
            logical_connectivity = deepcopy(data["physical_connectivity"])
            data["logical_connectivity"] = logical_connectivity

        if not data.get("logical_connectivity_quality", None):
            logical_connectivity_quality = {}

            for q1_index, connected_qubits in data["logical_connectivity"].items():
                for q2_index in connected_qubits:
                    logical_connectivity_quality[(q1_index, q2_index)] = 1.0

            data["logical_connectivity_quality"] = logical_connectivity_quality

        return data

    @field_validator("physical_connectivity")
    def validate_physical_connectivity_symmetry(cls, physical_connectivity):
        for q, connected_qs in physical_connectivity.items():
            for connected_q in connected_qs:
                if (
                    physical_connectivity.get(connected_q, None)
                    and q not in physical_connectivity[connected_q]
                ):
                    raise ValueError(
                        f"The topology is not symmetric, qubit {q} not present in connected qubits of {connected_q}."
                    )
        return physical_connectivity

    @model_validator(mode="after")
    def validate_connectivity(self):
        # Check if all qubits exist in physical connectivity.
        if len(self.qubits) > 1:  # 1Q-systems do not have any connectivity.
            assert self.qubits.keys() == self.physical_connectivity.keys(), (
                f"Inconsistent qubit ids for {self.qubits} and {self.physical_connectivity}."
            )

        # Check if logical connectivity is subset of physical connectivity.
        for qubit_index in self.logical_connectivity:
            if (
                self.logical_connectivity[qubit_index]
                and not self.logical_connectivity[qubit_index]
                <= self.physical_connectivity[qubit_index]
            ):
                raise ValueError(
                    "Logical connectivity must be a subgraph of the physical connectivity."
                )

        # Check if qubit cross resonance (cancellation) pulse channels agree with the physical connectivity.
        physical_connectivities = {
            (src, tgt) for (src, tgts) in self.physical_connectivity.items() for tgt in tgts
        }
        cross_resonance_edges = {
            (src, chan.auxiliary_qubit)
            for src, qubit in self.qubits.items()
            for chan in qubit.cross_resonance_pulse_channels.values()
        }
        cross_cancellation_edges = {
            (src, chan.auxiliary_qubit)
            for src, qubit in self.qubits.items()
            for chan in qubit.cross_resonance_cancellation_pulse_channels.values()
        }

        assert cross_resonance_edges == cross_cancellation_edges, (
            "Cross resonance channels mismatch cross resonance cancellation channels."
        )
        assert physical_connectivities == cross_resonance_edges, (
            "Cross resonance channels mismatch physical connectivity."
        )
        assert physical_connectivities == cross_cancellation_edges, (
            "Cross resonance cancellation channels mismatch physical connectivity."
        )

        # Check unique physical channel indices.
        physical_channel_indices = [
            qubit.physical_channel.name_index for qubit in self.quantum_devices
        ]
        assert len(physical_channel_indices) == len(set(physical_channel_indices)), (
            f"Physical channel indices must be unique, found {physical_channel_indices}."
        )

        return self

    @model_validator(mode="after")
    def validate_connectivity_quality(self):
        for q1_index, q2_index in self.logical_connectivity_quality:
            if not self.logical_connectivity.get(q1_index, None):
                raise IndexError(
                    f"The coupling ({q1_index}, {q2_index}) is not present in `logical_connectivity`."
                )

        for q1_index, connected_qubits in self.logical_connectivity.items():
            for q2_index in connected_qubits:
                if (q1_index, q2_index) not in self.logical_connectivity_quality:
                    raise IndexError(
                        f"The coupling ({q1_index}, {q2_index}) is not present in `logical_connectivity_quality`."
                    )

        return self

    @model_validator(mode="after")
    def validate_error_mitigation(self):
        if self.error_mitigation.is_enabled and not all(
            [q_id in self.qubits.keys() for q_id in self.error_mitigation.qubits]
        ):
            raise ValueError(
                "Please provide an error mitigation strategy for all qubits in the hardware model."
            )

        return self

    def __eq__(self, other: PhysicalHardwareModel) -> bool:
        if not super().__eq__(other):
            return False

        s_qubits = list(self.qubits.values())
        o_qubits = list(other.qubits.values())
        if len(s_qubits) != len(o_qubits):
            return False

        for s, o in zip(s_qubits, o_qubits):
            if s != o:
                return False

        return True

    @property
    def is_calibrated(self) -> bool:
        for qubit in self.qubits.values():
            if not qubit.is_calibrated:
                return False
        return True

    @property
    def number_of_qubits(self) -> int:
        return len(self.qubits)

    @property
    def physical_channel_map(self) -> dict:
        return self._physical_channel_map

    def qubit_with_index(self, index: int | QubitId) -> Qubit:
        return self.qubits[index]

    def index_of_qubit(self, qubit: Qubit) -> QubitId:
        """Returns the index of the given qubit in the hardware model."""
        return self._qubits_to_qubit_ids.get(qubit, None)

    def pulse_channel_with_id(self, id_: str):
        return self._ids_to_pulse_channels.get(id_, None)

    def physical_channel_with_id(self, id_: str):
        return self._ids_to_physical_channels.get(id_, None)

    def physical_channel_for_pulse_channel_id(self, id_: str):
        return self._pulse_channel_ids_to_physical_channel.get(id_, None)

    def device_for_pulse_channel_id(self, id_: str):
        return self._pulse_channel_ids_to_device.get(id_, None)

    def device_for_physical_channel_id(self, id_: str) -> Qubit | Resonator | None:
        if id_ in self._physical_channel_ids_to_resonator:
            return self._physical_channel_ids_to_resonator[id_]
        if id_ in self._physical_channel_ids_to_qubit:
            return self._physical_channel_ids_to_qubit[id_]
        log.warning(f"No device found for physical channel id {id_}.")

    def qubit_for_physical_channel_id(self, id_: str) -> Qubit | None:
        """Returns the qubit associated with the given physical channel id."""
        if id_ in self._physical_channel_ids_to_qubit:
            return self._physical_channel_ids_to_qubit[id_]
        log.warning(f"No qubit found for physical channel id {id_}.")

    def qubit_for_resonator(self, resonator: Resonator) -> Qubit | None:
        """Returns the qubit associated with the given resonator."""
        if resonator in self._resonators_to_qubits:
            return self._resonators_to_qubits[resonator]
        log.warning(f"No qubit found for resonator {resonator}.")

    @property
    def quantum_devices(self) -> list[Qubit, Resonator]:
        """
        Returns all quantum (an)harmonic oscillator devices
        in this hardware model as a list.
        """
        qubits = list(self.qubits.values())
        resonators = [qubit.resonator for qubit in qubits]
        return qubits + resonators

    def qubit_quality(self, physical_qubit_index: int):
        linear_mitigation = getattr(
            self.error_mitigation.readout_mitigation, "linear", None
        )
        if linear_mitigation is None:
            return 1.0
        readout_quality = linear_mitigation.get(physical_qubit_index, None)
        if readout_quality is not None:
            # TODO: COMPILER-706 linear readout mitigation currently a 2x2 matrix,
            #  we may want to change this to be a dictionary like in the old hardware model.
            return (readout_quality[0, 0] + readout_quality[1, 1]) / 2
        else:
            return 0


PydLogicalHardwareModel = LogicalHardwareModel
PydPhysicalHardwareModel = PhysicalHardwareModel
Model = PhysicalHardwareModel | QuantumHardwareModel
