# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025-2026 Oxford Quantum Circuits Ltd

import pytest

from qat.experimental.system_data.materialisers.purr.decoder import (
    decode_jsonpickle_payload,
)

from tests.unit.backend.qblox.utils import create_parameters

test_parameters = create_parameters(["model", "dummy_config", "qubit_count", "allocation"])


def test_pydantic_qblox_configuration_decodes_to_source_fields():
    """Keep stable Qblox consumers compatible with Pydantic-v2 calibration state."""

    decoded = decode_jsonpickle_payload(
        {
            "py/object": "qat.backend.qblox.config.specification.QbloxConfig",
            "py/state": {
                "__dict__": {
                    "slot_idx": None,
                    "module": {
                        "py/object": "qat.backend.qblox.config.specification.ModuleConfig",
                        "py/state": {
                            "__dict__": {"lo": {"out0_in0_en": True}},
                            "__pydantic_extra__": None,
                            "__pydantic_fields_set__": {"py/set": ["lo"]},
                            "__pydantic_private__": None,
                        },
                    },
                    "sequencers": {},
                },
                "__pydantic_extra__": None,
                "__pydantic_fields_set__": {"py/set": ["module", "sequencers"]},
                "__pydantic_private__": None,
            },
        }
    )

    assert decoded == {
        "slot_idx": None,
        "module": {"lo": {"out0_in0_en": True}},
        "sequencers": {},
    }


@pytest.mark.parametrize(
    "qblox_model,dummy_config,qubit_count,allocation",
    test_parameters,
    indirect=["qblox_model"],
)
def test_module_allocation(qblox_model, dummy_config, qubit_count, allocation):
    """During hw model construction, modules are allocated for qubits.

    This test parametrizes the hw model's construction with different dummy configuration
    scenarios where QCM-RF, QRM-RF, or QRC is chosen first for module allocation. See the
    helper QbloxSlotAllocator.
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
