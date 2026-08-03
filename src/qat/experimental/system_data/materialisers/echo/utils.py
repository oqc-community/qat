# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Factory helpers for building default echo QPU canonical system data.

These functions produce a fully populated
:class:`~qat.experimental.system_data.canonical.schema.CanonicalSystemData` without
requiring any external source payload or parse step.

Typical usage
-------------

Build a 4-qubit ring topology with all defaults::

    from qat.experimental.system_data.materialisers.echo.utils import (
        build_default_echo_topology,
    )

    builder = build_default_echo_topology(qubit_count=4)
    canonical = builder.build()

Override frequencies and connectivity::

    builder = build_default_echo_topology(
        qubit_count=3,
        connectivity=[(0, 1), (1, 2)],
        calibration_id="lab-echo-001",
        drive_frequency_hz=5_200_000_000,
        readout_frequency_hz=7_500_000_000,
    )
    canonical = builder.build()

All parameters have useful defaults that reproduce a standard transmon-style echo QPU.
A :class:`~qat.experimental.system_data.materialisers.builder.CanonicalSystemDataBuilder`
is returned — call :meth:`~CanonicalSystemDataBuilder.build` to produce the frozen
:class:`~qat.experimental.system_data.canonical.schema.CanonicalSystemData`.  No boundary
parse step is required.

Default hardware parameters
---------------------------

- Drive frequency: 5.5 GHz (``drive_frequency_hz``)
- Readout frequency: 8.5 GHz (``readout_frequency_hz``)
- Sample time: 1 000 ps / 1 ns (``sample_time_ps``)
- Acquire limit: unlimited / ``-1`` (``acquire_limit``)
- Acquire modes: ``integrator``, ``scope`` (``acquire_modes``)
- Default acquire mode: ``integrator`` (``default_acquire_mode``)
- Reset methods: ``passive`` (``reset_methods``)
- Default reset method: ``passive`` (``default_reset_method``)
"""

from __future__ import annotations

from qat.experimental.system_data.canonical.schema import ModeData
from qat.experimental.system_data.materialisers.builder import CanonicalSystemDataBuilder


def generate_echo_ring_connectivity(qubit_count: int) -> list[tuple[int, int]]:
    """Return undirected ring connectivity edges for ``qubit_count`` qubits.

    :param qubit_count: Number of qubits in the lattice.
    :returns: List of undirected ``(source_index, target_index)`` pairs.  Returns an
        empty list for ``qubit_count <= 1`` and a single pair for ``qubit_count == 2``.
    :raises ValueError: If ``qubit_count`` is negative.
    """
    if qubit_count < 0:
        raise ValueError(f"qubit_count must be non-negative, got {qubit_count!r}.")
    if qubit_count <= 1:
        return []
    if qubit_count == 2:
        return [(0, 1)]
    return [(i, (i + 1) % qubit_count) for i in range(qubit_count)]


def build_default_echo_topology(
    qubit_count: int = 4,
    connectivity: list[tuple[int, int]] | None = None,
    calibration_id: str = "",
    drive_frequency_hz: int = 5_500_000_000,
    readout_frequency_hz: int = 8_500_000_000,
    sample_time_ps: int = 1_000,
    acquire_limit: int = -1,
    acquire_modes: list[str] | None = None,
    default_acquire_mode: str = "integrator",
    reset_methods: list[str] | None = None,
    default_reset_method: str = "passive",
) -> CanonicalSystemDataBuilder:
    """Build a default echo QPU canonical system data instance.

    Produces a fully populated
    :class:`~qat.experimental.system_data.canonical.schema.CanonicalSystemData`
    with a transmon-inspired signal path for each qubit and a configurable coupling
    topology.  The result is returned directly — no boundary parse step is required.

    Per-qubit signal path
    ~~~~~~~~~~~~~~~~~~~~~
    For each qubit at index ``i``:

    - **Oscillators**: ``lo_drive_q{i}`` (``drive_frequency_hz``),
      ``lo_readout_q{i}`` (``readout_frequency_hz``).
    - **Ports**: ``port_drive_q{i}`` (``sample_time_ps``),
      ``port_readout_q{i}`` (``sample_time_ps``, acquire-capable).
    - **Channels**: ``ch_drive_q{i}``, ``ch_measure_q{i}``, ``ch_acquire_q{i}``.
    - **Modes**: ``drive_q{i}``, ``measure_q{i}``, ``acquire_q{i}``.

    Coupling topology
    ~~~~~~~~~~~~~~~~~
    Each undirected edge in ``connectivity`` produces two directed
    :class:`~qat.experimental.system_data.canonical.schema.QubitCouplingData` entries
    (both directions).  If ``connectivity`` is ``None``, a ring topology is generated
    via :func:`generate_echo_ring_connectivity`.

    :param qubit_count: Number of qubits to include in the topology.
    :param connectivity: Undirected coupling edges as ``(source_index, target_index)``
        pairs.  ``None`` generates a ring topology automatically.
    :param calibration_id: Calibration identifier embedded in the returned record.
    :param drive_frequency_hz: Drive channel frequency in Hz.  Defaults to 5.5 GHz.
    :param readout_frequency_hz: Readout channel frequency in Hz.  Defaults to 8.5 GHz.
    :param sample_time_ps: Port sample period in picoseconds.  Defaults to 1 000 ps (1 ns).
    :param acquire_limit: Maximum acquisitions per execution batch; ``-1`` is unlimited.
    :param acquire_modes: Supported acquisition mode type strings.  Defaults to
        ``["integrator", "scope"]``.
    :param default_acquire_mode: Default acquisition mode type.  Defaults to
        ``"integrator"``.
    :param reset_methods: Supported reset strategy type strings.  Defaults to
        ``["passive"]``.
    :param default_reset_method: Default reset strategy type.  Defaults to ``"passive"``.
    :returns: :class:`~qat.experimental.system_data.materialisers.builder.CanonicalSystemDataBuilder`
        populated with the echo QPU topology.  Call :meth:`~CanonicalSystemDataBuilder.build`
        to produce the frozen :class:`~qat.experimental.system_data.canonical.schema.CanonicalSystemData`.
    :raises ValueError: If ``qubit_count`` is negative, or if any connectivity index is
        out of range for ``qubit_count``.
    """
    if qubit_count < 0:
        raise ValueError(f"qubit_count must be non-negative, got {qubit_count!r}.")

    if connectivity is None:
        connectivity = generate_echo_ring_connectivity(qubit_count)

    _validate_connectivity(connectivity, qubit_count)

    if acquire_modes is None:
        acquire_modes = ["integrator", "scope"]
    if reset_methods is None:
        reset_methods = ["passive"]

    builder = CanonicalSystemDataBuilder()
    builder.with_calibration_id(calibration_id)
    builder.with_acquire_limit(acquire_limit)
    for mode in acquire_modes:
        builder.with_acquire_mode(mode)
    builder.with_default_acquire_mode(default_acquire_mode)
    for method in reset_methods:
        builder.with_reset_method(method)
    builder.with_default_reset_method(default_reset_method)

    for i in range(qubit_count):
        _add_qubit_signal_path(
            builder,
            i,
            drive_frequency_hz=drive_frequency_hz,
            readout_frequency_hz=readout_frequency_hz,
            sample_time_ps=sample_time_ps,
        )

    for source_idx, target_idx in connectivity:
        _add_bidirectional_coupling(builder, source_idx, target_idx)

    return builder


def _validate_connectivity(connectivity: list[tuple[int, int]], qubit_count: int) -> None:
    """Raise ``ValueError`` if any connectivity edge references an out-of-range index."""

    for source_idx, target_idx in connectivity:
        for idx in (source_idx, target_idx):
            if not (0 <= idx < qubit_count):
                raise ValueError(
                    f"Connectivity index {idx!r} is out of range for "
                    f"qubit_count={qubit_count!r}."
                )


def _add_qubit_signal_path(
    builder: CanonicalSystemDataBuilder,
    index: int,
    *,
    drive_frequency_hz: int,
    readout_frequency_hz: int,
    sample_time_ps: int,
) -> None:
    """Append oscillators, ports, channels, and a qubit with modes for qubit ``index``.

    Mutates ``builder`` in place.
    """
    drive_lo_id = f"lo_drive_q{index}"
    readout_lo_id = f"lo_readout_q{index}"
    drive_port_id = f"port_drive_q{index}"
    readout_port_id = f"port_readout_q{index}"
    drive_ch_id = f"ch_drive_q{index}"
    measure_ch_id = f"ch_measure_q{index}"
    acquire_ch_id = f"ch_acquire_q{index}"

    builder.with_oscillator(drive_lo_id, drive_frequency_hz)
    builder.with_oscillator(readout_lo_id, readout_frequency_hz)

    builder.with_port(drive_port_id, sample_time_ps)
    builder.with_port(readout_port_id, sample_time_ps, acquire_allowed=True)

    builder.with_channel(
        drive_ch_id,
        drive_port_id,
        drive_frequency_hz,
        oscillator_reference=drive_lo_id,
    )
    builder.with_channel(
        measure_ch_id,
        readout_port_id,
        readout_frequency_hz,
        oscillator_reference=readout_lo_id,
    )
    builder.with_channel(
        acquire_ch_id,
        readout_port_id,
        readout_frequency_hz,
        oscillator_reference=readout_lo_id,
    )

    builder.with_qubit(
        f"q{index}",
        index,
        modes=(
            ModeData(id=f"drive_q{index}", channel_id=drive_ch_id),
            ModeData(id=f"measure_q{index}", channel_id=measure_ch_id),
            ModeData(id=f"acquire_q{index}", channel_id=acquire_ch_id),
        ),
    )


def _add_bidirectional_coupling(
    builder: CanonicalSystemDataBuilder, source_idx: int, target_idx: int
) -> None:
    """Append two directed couplings (both directions) for an undirected edge."""

    builder.with_coupling(f"q{source_idx}", f"q{target_idx}")
    builder.with_coupling(f"q{target_idx}", f"q{source_idx}")
