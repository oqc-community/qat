# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from importlib.metadata import version
from json import loads

import pytest

from qat.executables import Executable
from qat.experimental.backend.qblox.payload import QbloxPackage, QbloxProgram
from qat.experimental.dialect.q1 import StopOp
from qat.experimental.dialect.q1_sequence.ir.attrs import (
    ConnectionAttr,
    ModuleConfigAttr,
    OutputConfigAttr,
    SequencerConfigAttr,
)
from qat.experimental.dialect.q1_sequence.ir.ops import SequenceOp
from qat.experimental.system_data.hardware.qblox.models import (
    DirectionKind,
    QbloxModuleKind,
)


def _sequence() -> SequenceOp:
    return SequenceOp(
        "q0.readout",
        [StopOp()],
        port_id="readout",
        instrument_id="cluster0",
        slot_idx=2,
        seq_idx=0,
        sequencer_config=SequencerConfigAttr(
            port_id="readout",
            connections=[ConnectionAttr(DirectionKind.output, [0])],
            output_path_connections=[],
            acquisition_path_connections=[],
            disabled_outputs=[],
            disabled_acquisition_paths=[],
        ),
        module_config=ModuleConfigAttr(
            2,
            "cluster0",
            QbloxModuleKind.qcm,
            outputs=[OutputConfigAttr(0)],
        ),
    )


def _program() -> QbloxProgram:
    package = QbloxPackage.from_sequence(_sequence())
    return QbloxProgram(
        packages={package.pulse_channel_id: package},
        metadata={"arguments": [1, 2]},
    )


def _package_fields(package: QbloxPackage) -> dict[str, object]:
    return {
        "pulse_channel_id": package.pulse_channel_id,
        "physical_channel_id": package.physical_channel_id,
        "instrument_id": package.instrument_id,
        "seq_idx": package.seq_idx,
        "seq_config": package.seq_config,
        "slot_idx": package.slot_idx,
        "mod_config": package.mod_config,
        "sequence": package.sequence,
    }


def test_program_serializes_q1_sequence_objects():
    program = _program()

    serialized = loads(program.to_json())

    assert serialized == program.to_dict()
    package = serialized["packages"]["q0.readout"]
    assert package["sequence"] == {
        "program": "stop\n",
        "waveforms": {},
        "weights": {},
        "acquisitions": {},
    }
    assert package["seq_config"]["connections"] == [{"direction": "out", "port_ids": [0]}]
    assert package["mod_config"]["kind"] == "qcm"
    assert program.to_json() == program.to_json()


def test_program_discovers_runtime_compatibility_versions():
    program = QbloxProgram()

    assert program.driver_version == version("qblox-instruments")
    assert program.fw_version == "2.0.0"


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        pytest.param(
            {"pulse_channel_id": ""},
            "at least 1 character",
            id="empty-pulse-channel",
        ),
        pytest.param(
            {"physical_channel_id": ""},
            "at least 1 character",
            id="empty-physical-channel",
        ),
        pytest.param(
            {"instrument_id": ""},
            "at least 1 character",
            id="empty-instrument",
        ),
        pytest.param(
            {"slot_idx": 0},
            "greater than or equal to 1",
            id="zero-slot",
        ),
        pytest.param(
            {"seq_idx": -1},
            "greater than or equal to 0",
            id="negative-sequencer",
        ),
    ],
)
def test_package_rejects_invalid_identity(values, expected):
    fields = _package_fields(QbloxPackage.from_sequence(_sequence()))
    fields.update(values)

    with pytest.raises(ValueError, match=expected):
        QbloxPackage(**fields)


def test_package_requires_fully_configured_sequence():
    with pytest.raises(ValueError, match="fully configured and allocated"):
        QbloxPackage.from_sequence(SequenceOp("q0.drive", [StopOp()]))


@pytest.mark.parametrize("field", ["seq_config", "mod_config", "sequence"])
def test_package_rejects_non_mapping_emitted_data(field):
    fields = _package_fields(QbloxPackage.from_sequence(_sequence()))
    fields[field] = []

    with pytest.raises(TypeError, match="must be a mapping"):
        QbloxPackage(**fields)


def test_package_rejects_identity_that_drifts_from_emitted_configuration():
    package = QbloxPackage.from_sequence(_sequence())
    fields = _package_fields(package)
    fields["instrument_id"] = "cluster1"

    with pytest.raises(ValueError, match="does not match its module configuration"):
        QbloxPackage(**fields)


def test_package_rejects_boolean_module_slot_identity():
    package = QbloxPackage.from_sequence(_sequence())
    fields = _package_fields(package)
    fields["mod_config"] = {**package.mod_config, "slot_idx": True}

    with pytest.raises(ValueError, match="slot_idx does not match"):
        QbloxPackage(**fields)


def test_package_rejects_physical_channel_that_drifts_from_sequencer_config():
    package = QbloxPackage.from_sequence(_sequence())
    fields = _package_fields(package)
    fields["physical_channel_id"] = "different"

    with pytest.raises(ValueError, match="does not match its sequencer configuration"):
        QbloxPackage(**fields)


def test_program_revalidates_mutated_payload_before_serialization():
    program = _program()
    package = program.packages["q0.readout"]
    package.mod_config["slot_idx"] = 3

    with pytest.raises(ValueError, match="does not match its module configuration"):
        program.to_json()
    with pytest.raises(ValueError, match="does not match its module configuration"):
        Executable[QbloxProgram](programs=[program]).serialize()


def test_program_round_trips_through_executable_serialization():
    executable = Executable[QbloxProgram](programs=[_program()])

    restored = Executable[QbloxProgram].deserialize(executable.serialize())

    assert isinstance(restored.programs[0], QbloxProgram)
    assert restored.programs[0].to_dict() == _program().to_dict()


def test_program_rejects_unknown_fields_during_rehydration():
    data = _program().to_dict()
    data["pakages"] = data.pop("packages")

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        QbloxProgram.model_validate(data)


def test_program_rejects_incorrect_object_type():
    data = _program().to_dict()
    data["object_type"] = "example.OtherProgram"

    with pytest.raises(ValueError, match="object_type"):
        QbloxProgram.model_validate(data)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        (
            "driver_version",
            "0.0.0",
            "must match the installed qblox-instruments version",
        ),
        ("fw_version", "0.0.0", "must be 2.0.0"),
    ],
)
def test_program_rejects_incompatible_versions(field, value, expected):
    data = _program().to_dict()
    data[field] = value

    with pytest.raises(ValueError, match=expected):
        QbloxProgram.model_validate(data)


def test_program_rejects_inconsistent_package_mapping_identity():
    program = _program()
    package = next(iter(program.packages.values()))

    with pytest.raises(ValueError, match="package key"):
        QbloxProgram(packages={"different": package})


def test_program_rejects_non_mapping_packages():
    with pytest.raises(TypeError, match="packages must be a mapping"):
        QbloxProgram(packages=[])


def test_program_rejects_empty_package_mapping_key():
    package = QbloxPackage.from_sequence(_sequence())

    with pytest.raises(ValueError, match="mapping keys must be non-empty"):
        QbloxProgram(packages={"": package})


def test_program_rejects_non_string_package_mapping_keys_before_coercion():
    first = QbloxPackage.from_sequence(_sequence())
    second_sequence = _sequence()
    second_sequence.attributes["channel_id"] = type(second_sequence.channel_id)(
        "q1.readout"
    )
    second_sequence.properties["seq_idx"] = type(second_sequence.seq_idx)(1)
    second = QbloxPackage.from_sequence(second_sequence)

    with pytest.raises(TypeError, match="mapping keys must be strings"):
        QbloxProgram(packages={b"q0.readout": first, "q0.readout": second})


def test_program_rejects_duplicate_physical_allocations():
    first = QbloxPackage.from_sequence(_sequence())
    second = first.model_copy(update={"pulse_channel_id": "q1.readout"})

    with pytest.raises(ValueError, match="duplicate physical allocation"):
        QbloxProgram(
            packages={
                first.pulse_channel_id: first,
                second.pulse_channel_id: second,
            }
        )


def test_program_rejects_conflicting_shared_module_configuration():
    first = QbloxPackage.from_sequence(_sequence())
    second_sequence = _sequence()
    second_sequence.attributes["channel_id"] = type(second_sequence.channel_id)(
        "q1.readout"
    )
    second_sequence.properties["seq_idx"] = type(second_sequence.seq_idx)(1)
    second = QbloxPackage.from_sequence(second_sequence)
    second.mod_config["kind"] = "qrm"

    with pytest.raises(ValueError, match="conflicting module configurations"):
        QbloxProgram(
            packages={
                first.pulse_channel_id: first,
                second.pulse_channel_id: second,
            }
        )


def test_program_compares_shared_module_configuration_types_strictly():
    first = QbloxPackage.from_sequence(_sequence())
    second_sequence = _sequence()
    second_sequence.attributes["channel_id"] = type(second_sequence.channel_id)(
        "q1.readout"
    )
    second_sequence.properties["seq_idx"] = type(second_sequence.seq_idx)(1)
    second = QbloxPackage.from_sequence(second_sequence)
    first.mod_config["flag"] = True
    second.mod_config["flag"] = 1

    with pytest.raises(ValueError, match="conflicting module configurations"):
        QbloxProgram(
            packages={
                first.pulse_channel_id: first,
                second.pulse_channel_id: second,
            }
        )


@pytest.mark.parametrize(
    ("metadata", "error", "expected"),
    [
        pytest.param([], TypeError, "must be a mapping", id="not-a-mapping"),
        pytest.param(
            {"unsupported": object()},
            TypeError,
            "Unsupported Qblox payload",
            id="unsupported-value",
        ),
        pytest.param(
            {"value": float("nan")},
            ValueError,
            "must be finite",
            id="non-finite-value",
        ),
    ],
)
def test_program_rejects_invalid_metadata(metadata, error, expected):
    with pytest.raises(error, match=expected):
        QbloxProgram(metadata=metadata)


@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param({1: "integer", "1": "string"}, id="root-collision"),
        pytest.param({"nested": {1: "integer", "1": "string"}}, id="nested-collision"),
    ],
)
def test_program_rejects_non_string_json_mapping_keys(metadata):
    with pytest.raises(TypeError, match="payload mapping keys must be strings, got int"):
        QbloxProgram(metadata=metadata)


def test_program_normalizes_tuple_metadata_to_json_arrays():
    program = QbloxProgram(metadata={"nested": {"tuple": (1, 2)}})

    assert program.metadata == {"nested": {"tuple": [1, 2]}}
