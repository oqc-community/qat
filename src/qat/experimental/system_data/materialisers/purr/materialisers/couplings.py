# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Coupling builders for PuRR canonical materialisation."""

from typing import Any

from qat.experimental.system_data.canonical.schema import (
    QubitCouplingData,
    TwoQubitGateFidelityData,
)
from qat.experimental.system_data.materialisers.errors import (
    MaterialisationConsistencyError,
    MaterialisationIntegrityError,
)


def _build_qubit_id_by_index(quantum_devices: dict[str, Any]) -> dict[int, str]:
    """Build qubit-id lookup by integer index from quantum-device payloads."""

    return {
        payload.get("index"): payload.get("id")
        for payload in quantum_devices.values()
        if isinstance(payload, dict)
        and isinstance(payload.get("index"), int)
        and isinstance(payload.get("id"), str)
    }


def _build_coupling_entry(
    *,
    entry: Any,
    qubit_id_by_index: dict[int, str],
) -> QubitCouplingData:
    """Build one canonical coupling entry from a PuRR coupling record.

    :raises MaterialisationIntegrityError: If the coupling record shape is malformed.
    :raises MaterialisationConsistencyError: If direction indices do not map to known
        qubits.
    """

    if not isinstance(entry, dict):
        raise MaterialisationIntegrityError(
            "Coupling entry must be a dictionary.",
            source_type="purr",
            path="$.qubit_direction_couplings[*]",
            details={"value": entry},
        )

    direction = entry.get("direction")
    if (
        not isinstance(direction, list)
        or len(direction) != 2
        or not all(isinstance(index, int) for index in direction)
    ):
        raise MaterialisationIntegrityError(
            "Coupling direction must be a 2-element integer list.",
            source_type="purr",
            path="$.qubit_direction_couplings[*].direction",
            details={"value": direction},
        )

    source_qubit_id = qubit_id_by_index.get(direction[0])
    target_qubit_id = qubit_id_by_index.get(direction[1])
    if not isinstance(source_qubit_id, str) or not isinstance(target_qubit_id, str):
        raise MaterialisationConsistencyError(
            "Coupling direction references unknown qubit index.",
            source_type="purr",
            path="$.qubit_direction_couplings[*].direction",
            details={"direction": direction},
        )

    quality = entry.get("quality")
    fidelity = float(quality) if isinstance(quality, int | float) else 0.0

    return QubitCouplingData(
        source_qubit_id=source_qubit_id,
        target_qubit_id=target_qubit_id,
        gate_fidelities=(TwoQubitGateFidelityData(gate="ZX", fidelity=fidelity),),
    )


def _build_couplings(
    *,
    qubit_direction_couplings: list[Any],
    quantum_devices: dict[str, Any],
) -> tuple[QubitCouplingData, ...]:
    """Build canonical directional coupling records from PuRR coupling metadata.

    Coupling entries are parsed strictly: malformed records are surfaced as
    materialiser errors rather than being silently skipped.
    """

    qubit_id_by_index = _build_qubit_id_by_index(quantum_devices)

    couplings: list[QubitCouplingData] = []
    for entry in qubit_direction_couplings:
        coupling = _build_coupling_entry(
            entry=entry,
            qubit_id_by_index=qubit_id_by_index,
        )
        couplings.append(coupling)

    return tuple(couplings)
