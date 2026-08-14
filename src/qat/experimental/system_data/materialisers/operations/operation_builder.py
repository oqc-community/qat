# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Abstract operation-builder contract for canonical per-qubit operation sets."""

from __future__ import annotations

from abc import ABC, abstractmethod

from qat.experimental.system_data.canonical.schema import OperationData


class AbstractOperationBuilder(ABC):
    """Contract for constructing qubit-owned canonical operation sets.

    Subclasses implement the operation-construction methods and inherit the
    template ``build_single_qubit_operations`` and ``build`` assembly logic.

    :param qubit_id: Identifier of the qubit that will own these operations.
    :param coupled_qubit_ids: Identifiers of qubits this qubit drives.
    :param control_qubit_ids: Identifiers of qubits that drive this qubit.
    :param has_x_pi: Whether a calibrated X(π) pulse is available.
    """

    def __init__(
        self,
        qubit_id: str,
        coupled_qubit_ids: tuple[str, ...] = (),
        control_qubit_ids: tuple[str, ...] = (),
        has_x_pi: bool = True,
    ) -> None:
        if not isinstance(qubit_id, str) or not qubit_id:
            raise ValueError("qubit_id must be a non-empty string.")

        self.qubit_id = qubit_id
        self.coupled_qubit_ids = coupled_qubit_ids
        self.control_qubit_ids = control_qubit_ids
        self.has_x_pi = has_x_pi

    @staticmethod
    def _merge_operations(
        base: tuple[OperationData, ...],
        extra: tuple[OperationData, ...],
    ) -> tuple[OperationData, ...]:
        """Merge ``extra`` into ``base`` using last-wins ID deduplication."""
        by_id: dict[str, OperationData] = {op.id: op for op in base}
        for op in extra:
            by_id[op.id] = op
        return tuple(by_id.values())

    @abstractmethod
    def make_z_operation(self) -> OperationData: ...  # pragma: no cover

    @abstractmethod
    def make_x_gate(self) -> OperationData: ...  # pragma: no cover

    @abstractmethod
    def make_y_gate(self) -> OperationData: ...  # pragma: no cover

    @abstractmethod
    def make_u_gate(self) -> OperationData: ...  # pragma: no cover

    @abstractmethod
    def make_had_operation(self) -> OperationData: ...  # pragma: no cover

    @abstractmethod
    def make_sx_operation(self) -> OperationData: ...  # pragma: no cover

    @abstractmethod
    def make_sxdg_operation(self) -> OperationData: ...  # pragma: no cover

    @abstractmethod
    def make_s_operation(self) -> OperationData: ...  # pragma: no cover

    @abstractmethod
    def make_sdg_operation(self) -> OperationData: ...  # pragma: no cover

    @abstractmethod
    def make_t_operation(self) -> OperationData: ...  # pragma: no cover

    @abstractmethod
    def make_tdg_operation(self) -> OperationData: ...  # pragma: no cover

    @abstractmethod
    def make_rx_gate(self) -> OperationData: ...  # pragma: no cover

    @abstractmethod
    def make_ry_gate(self) -> OperationData: ...  # pragma: no cover

    @abstractmethod
    def make_rz_gate(self) -> OperationData: ...  # pragma: no cover

    @abstractmethod
    def make_u1_gate(self) -> OperationData: ...  # pragma: no cover

    @abstractmethod
    def make_u2_gate(self) -> OperationData: ...  # pragma: no cover

    @abstractmethod
    def make_id_gate(self) -> OperationData: ...  # pragma: no cover

    @abstractmethod
    def make_delay_operation(self) -> OperationData: ...  # pragma: no cover

    @abstractmethod
    def make_measure_operation(self) -> OperationData: ...  # pragma: no cover

    @abstractmethod
    def make_initiate_operation(self) -> OperationData: ...  # pragma: no cover

    @abstractmethod
    def make_reset_operation(self) -> OperationData: ...  # pragma: no cover

    @abstractmethod
    def make_two_qubit_operations(
        self,
    ) -> tuple[OperationData, ...]: ...  # pragma: no cover

    def make_private_single_qubit_operations(self) -> tuple[OperationData, ...]:
        """Return implementation-specific private/support single-qubit operations.

        The default implementation returns no private operations.
        """
        return ()

    def build_single_qubit_operations(
        self,
        extra_operations: tuple[OperationData, ...] = (),
    ) -> tuple[OperationData, ...]:
        """Return the single-qubit operation set for this builder's qubit."""
        defaults = (
            *self.make_private_single_qubit_operations(),
            self.make_rz_gate(),
            self.make_rx_gate(),
            self.make_ry_gate(),
            self.make_u_gate(),
            self.make_x_gate(),
            self.make_y_gate(),
            self.make_z_operation(),
            self.make_had_operation(),
            self.make_sx_operation(),
            self.make_sxdg_operation(),
            self.make_s_operation(),
            self.make_sdg_operation(),
            self.make_t_operation(),
            self.make_tdg_operation(),
            self.make_u1_gate(),
            self.make_u2_gate(),
            self.make_id_gate(),
            self.make_delay_operation(),
            self.make_measure_operation(),
            self.make_initiate_operation(),
            self.make_reset_operation(),
        )
        if not extra_operations:
            return defaults
        return self._merge_operations(defaults, extra_operations)

    def build(
        self,
        extra_operations: tuple[OperationData, ...] = (),
    ) -> tuple[OperationData, ...]:
        """Return the full operation set including topology-derived multi-qubit gates."""
        operations: list[OperationData] = list(self.build_single_qubit_operations())
        operations.extend(self.make_two_qubit_operations())
        if not extra_operations:
            return tuple(operations)
        return self._merge_operations(tuple(operations), extra_operations)
