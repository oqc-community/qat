# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Unit tests for the echo QPU topology factory."""

import pytest

from qat.experimental.system_data.materialisers import boundary
from qat.experimental.system_data.materialisers.echo.utils import (
    build_default_echo_topology,
    generate_echo_ring_connectivity,
)


def test_ring_connectivity_zero_qubits():
    assert generate_echo_ring_connectivity(0) == []


def test_ring_connectivity_single_qubit():
    assert generate_echo_ring_connectivity(1) == []


def test_ring_connectivity_two_qubits():
    assert generate_echo_ring_connectivity(2) == [(0, 1)]


def test_ring_connectivity_three_qubits():
    assert generate_echo_ring_connectivity(3) == [(0, 1), (1, 2), (2, 0)]


def test_ring_connectivity_four_qubits():
    assert generate_echo_ring_connectivity(4) == [(0, 1), (1, 2), (2, 3), (3, 0)]


def test_ring_connectivity_negative_raises():
    with pytest.raises(ValueError, match="non-negative"):
        generate_echo_ring_connectivity(-1)


@pytest.mark.parametrize("qubit_count", [1, 2, 4, 8])
def test_topology_qubit_count(qubit_count):
    canonical = build_default_echo_topology(qubit_count=qubit_count).build()
    assert len(canonical.qubits) == qubit_count


@pytest.mark.parametrize("qubit_count", [1, 4])
def test_topology_oscillator_count(qubit_count):
    """Two oscillators (drive + readout) per qubit."""
    canonical = build_default_echo_topology(qubit_count=qubit_count).build()
    assert len(canonical.oscillators) == qubit_count * 2


@pytest.mark.parametrize("qubit_count", [1, 4])
def test_topology_port_count(qubit_count):
    """Two ports (drive + readout) per qubit."""
    canonical = build_default_echo_topology(qubit_count=qubit_count).build()
    assert len(canonical.ports) == qubit_count * 2


@pytest.mark.parametrize("qubit_count", [1, 4])
def test_topology_channel_count(qubit_count):
    """Three channels (drive + measure + acquire) per qubit."""
    canonical = build_default_echo_topology(qubit_count=qubit_count).build()
    assert len(canonical.channels) == qubit_count * 3


@pytest.mark.parametrize("qubit_count", [1, 4])
def test_topology_modes_per_qubit(qubit_count):
    """Three modes (drive + measure + acquire) per qubit."""
    canonical = build_default_echo_topology(qubit_count=qubit_count).build()
    for qubit in canonical.qubits:
        assert len(qubit.modes) == 3


def test_topology_qubit_ids():
    canonical = build_default_echo_topology(qubit_count=3).build()
    qubit_ids = {q.id for q in canonical.qubits}
    assert qubit_ids == {"q0", "q1", "q2"}


def test_topology_qubit_indices():
    canonical = build_default_echo_topology(qubit_count=3).build()
    qubit_indices = sorted(q.index for q in canonical.qubits)
    assert qubit_indices == [0, 1, 2]


def test_topology_qubit_mode_ids():
    canonical = build_default_echo_topology(qubit_count=2).build()
    for qubit in canonical.qubits:
        i = qubit.index
        mode_ids = {m.id for m in qubit.modes}
        assert mode_ids == {f"drive_q{i}", f"measure_q{i}", f"acquire_q{i}"}


def test_topology_readout_ports_acquire_allowed():
    canonical = build_default_echo_topology(qubit_count=2).build()
    readout_ports = [p for p in canonical.ports if p.acquire_allowed]
    assert len(readout_ports) == 2  # one per qubit


def test_topology_drive_ports_no_acquire():
    canonical = build_default_echo_topology(qubit_count=2).build()
    drive_ports = [p for p in canonical.ports if not p.acquire_allowed]
    assert len(drive_ports) == 2  # one per qubit


def test_topology_ring_couplings_four_qubits():
    """Ring of 4: 4 undirected edges → 8 directed coupling entries."""
    canonical = build_default_echo_topology(qubit_count=4).build()
    assert len(canonical.couplings) == 8


def test_topology_ring_couplings_two_qubits():
    """Ring of 2: 1 undirected edge → 2 directed coupling entries."""
    canonical = build_default_echo_topology(qubit_count=2).build()
    assert len(canonical.couplings) == 2


def test_topology_single_qubit_no_couplings():
    canonical = build_default_echo_topology(qubit_count=1).build()
    assert len(canonical.couplings) == 0


def test_topology_custom_connectivity():
    """Custom connectivity [(0,1),(0,2)] → 4 directed entries."""
    canonical = build_default_echo_topology(
        qubit_count=3, connectivity=[(0, 1), (0, 2)]
    ).build()
    assert len(canonical.couplings) == 4


def test_topology_custom_connectivity_qubit_ids_in_couplings():
    canonical = build_default_echo_topology(qubit_count=3, connectivity=[(0, 1)]).build()
    coupling_pairs = {(c.source_qubit_id, c.target_qubit_id) for c in canonical.couplings}
    assert coupling_pairs == {("q0", "q1"), ("q1", "q0")}


def test_topology_out_of_range_connectivity_raises():
    with pytest.raises(ValueError, match="out of range"):
        build_default_echo_topology(qubit_count=2, connectivity=[(0, 5)])


def test_topology_negative_qubit_count_raises():
    with pytest.raises(ValueError, match="non-negative"):
        build_default_echo_topology(qubit_count=-1)


def test_topology_acquire_mode_types():
    canonical = build_default_echo_topology().build()
    mode_types = {m.type for m in canonical.acquire_modes}
    assert mode_types == {"integrator", "scope"}


def test_topology_default_acquire_mode():
    canonical = build_default_echo_topology().build()
    assert canonical.default_acquire_mode == "integrator"


def test_topology_reset_method():
    canonical = build_default_echo_topology().build()
    assert len(canonical.reset_methods) == 1
    assert canonical.reset_methods[0].type == "passive"


def test_topology_default_reset_method():
    canonical = build_default_echo_topology().build()
    assert canonical.default_reset_method == "passive"


def test_topology_calibration_id_default():
    canonical = build_default_echo_topology().build()
    assert canonical.calibration_id == ""


def test_topology_calibration_id_custom():
    canonical = build_default_echo_topology(calibration_id="echo-test-42").build()
    assert canonical.calibration_id == "echo-test-42"


def test_topology_boundary_materialise_round_trip():
    """build_default_echo_topology via the full boundary pathway reconstructs an equal
    instance."""
    from qat.experimental.system_data.canonical.schema import CanonicalSystemData

    builder = build_default_echo_topology(qubit_count=2, calibration_id="echo-boundary")
    expected = builder.build()
    result = boundary.materialise(source_payload=builder.build_payload())

    assert isinstance(result, CanonicalSystemData)
    assert result == expected


def test_topology_drive_channel_references_drive_port_and_oscillator():
    canonical = build_default_echo_topology(qubit_count=1).build()
    port_ids = {p.id for p in canonical.ports}
    oscillator_ids = {o.id for o in canonical.oscillators}

    drive_channel = next(c for c in canonical.channels if c.id == "ch_drive_q0")
    assert drive_channel.port_id in port_ids
    assert drive_channel.oscillator_reference in oscillator_ids


def test_topology_acquire_channel_references_readout_port():
    canonical = build_default_echo_topology(qubit_count=1).build()
    readout_port_id = "port_readout_q0"
    acquire_channel = next(c for c in canonical.channels if c.id == "ch_acquire_q0")
    assert acquire_channel.port_id == readout_port_id


def test_topology_custom_drive_frequency():
    canonical = build_default_echo_topology(
        qubit_count=1, drive_frequency_hz=4_000_000_000
    ).build()
    drive_osc = next(o for o in canonical.oscillators if o.id == "lo_drive_q0")
    drive_ch = next(c for c in canonical.channels if c.id == "ch_drive_q0")
    assert drive_osc.frequency == 4_000_000_000
    assert drive_ch.frequency == 4_000_000_000


def test_topology_custom_readout_frequency():
    canonical = build_default_echo_topology(
        qubit_count=1, readout_frequency_hz=7_000_000_000
    ).build()
    readout_osc = next(o for o in canonical.oscillators if o.id == "lo_readout_q0")
    measure_ch = next(c for c in canonical.channels if c.id == "ch_measure_q0")
    assert readout_osc.frequency == 7_000_000_000
    assert measure_ch.frequency == 7_000_000_000


def test_topology_custom_sample_time():
    canonical = build_default_echo_topology(qubit_count=1, sample_time_ps=500).build()
    for port in canonical.ports:
        assert port.sample_time == 500


def test_topology_custom_acquire_limit():
    canonical = build_default_echo_topology(acquire_limit=100).build()
    assert canonical.acquire_limit == 100


def test_topology_default_acquire_limit():
    canonical = build_default_echo_topology().build()
    assert canonical.acquire_limit == -1


def test_topology_custom_acquire_modes():
    canonical = build_default_echo_topology(
        acquire_modes=["scope"], default_acquire_mode="scope"
    ).build()
    mode_types = [m.type for m in canonical.acquire_modes]
    assert mode_types == ["scope"]
    assert canonical.default_acquire_mode == "scope"


def test_topology_custom_reset_methods():
    canonical = build_default_echo_topology(
        reset_methods=["active", "passive"], default_reset_method="active"
    ).build()
    reset_types = [r.type for r in canonical.reset_methods]
    assert reset_types == ["active", "passive"]
    assert canonical.default_reset_method == "active"
