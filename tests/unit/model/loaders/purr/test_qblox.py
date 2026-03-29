# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import pytest

from tests.unit.backend.qblox.utils import create_parameters

test_parameters = create_parameters(["model", "dummy_config", "qubit_count", "allocation"])


@pytest.mark.parametrize(
    "qblox_model,dummy_config,qubit_count,allocation",
    test_parameters,
    indirect=["qblox_model"],
)
def test_module_allocation(qblox_model, dummy_config, qubit_count, allocation):
    """During hw model construction, modules are allocated for qubits. This test
    parametrizes the hw model's construction with different dummy configuration
    scenarios where QCM-RF, QRM-RF, or QRC is chosen first for module allocation.
    See the helper QbloxSlotAllocator.
    """

    assert len(qblox_model.qubits) == qubit_count
    for index in range(qubit_count):
        qubit = qblox_model.get_qubit(index)
        control_physical_channel = qubit.get_drive_channel().physical_channel
        readout_physical_channel = qubit.get_measure_channel().physical_channel
        alloc = allocation[index]
        assert control_physical_channel.slot_idx == alloc.control_slot
        assert readout_physical_channel.slot_idx == alloc.readout_slot
        assert (
            dummy_config[alloc.control_slot].value.split(" ")[1]
            in control_physical_channel.full_id()
        )
        assert (
            dummy_config[alloc.readout_slot].value.split(" ")[1]
            in readout_physical_channel.full_id()
        )
