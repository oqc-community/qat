# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import pytest
from pydantic import ValidationError

from qat.executables import AcquireData, Executable
from qat.ir.measure import AcquireMode

from tests.unit.utils.executables import MockProgram, MockProgram2


class TestExecutable:
    @pytest.mark.parametrize("program", [MockProgram, MockProgram2])
    def test_serialize_deserialize_roundtrip_returns_correct_type(self, program):
        dummy_program = program(shapes={"a": (10,), "b": (5, 20)})
        acquire_data = {
            "a": AcquireData(
                mode=AcquireMode.INTEGRATOR, shape=(10,), physical_channel="ch1"
            ),
            "b": AcquireData(mode=AcquireMode.RAW, shape=(5, 20), physical_channel="ch2"),
        }
        returns = {"a", "b"}
        calibration_id = "id_254"
        executable = Executable(
            programs=[dummy_program],
            acquires=acquire_data,
            assigns=[],
            returns=returns,
            calibration_id=calibration_id,
        )

        serialized = executable.serialize()
        deserialized = Executable.deserialize(serialized)

        assert isinstance(deserialized, Executable)
        assert len(deserialized.programs) == 1
        assert isinstance(deserialized.programs[0], program)
        assert deserialized.programs == [dummy_program]
        assert deserialized.acquires == acquire_data
        assert deserialized.assigns == []
        assert deserialized.returns == returns
        assert deserialized.calibration_id == calibration_id

    def test_serialize_deserialize_roundtrip_with_multiple_programs(self):
        dummy_programs = [
            MockProgram(shapes={"a": (10,)}),
            MockProgram(shapes={"b": (5, 20)}),
            MockProgram(shapes={"c": (2, 3, 4)}),
        ]
        acquire_data = {
            "a": AcquireData(
                mode=AcquireMode.INTEGRATOR, shape=(10,), physical_channel="ch1"
            ),
            "b": AcquireData(mode=AcquireMode.RAW, shape=(5, 20), physical_channel="ch2"),
            "c": AcquireData(
                mode=AcquireMode.SCOPE, shape=(2, 3, 4), physical_channel="ch3"
            ),
        }
        returns = {"a", "b", "c"}
        calibration_id = "id_254"
        executable = Executable(
            programs=dummy_programs,
            acquires=acquire_data,
            assigns=[],
            returns=returns,
            calibration_id=calibration_id,
        )

        serialized = executable.serialize()
        deserialized = Executable.deserialize(serialized)

        assert isinstance(deserialized, Executable)
        assert isinstance(deserialized.programs, list)
        assert len(deserialized.programs) == 3
        assert isinstance(deserialized.programs[0], MockProgram)
        assert isinstance(deserialized.programs[1], MockProgram)
        assert deserialized.programs == dummy_programs
        assert deserialized.acquires == acquire_data
        assert deserialized.assigns == []
        assert deserialized.returns == returns
        assert deserialized.calibration_id == calibration_id

    def test_different_program_types_raises_error(self):
        dummy_programs = [
            MockProgram(shapes={"a": (10,)}),
            MockProgram2(shapes={"b": (5, 20)}),
        ]

        with pytest.raises(
            ValidationError, match="All programs in the executable must be of the same type"
        ):
            Executable(
                programs=dummy_programs,
                acquires={},
                assigns=[],
                returns=set(),
                calibration_id="",
            )

    def test_single_program_is_saved_as_list(self):
        dummy_program = MockProgram(shapes={"a": (10,)})

        executable = Executable(
            programs=dummy_program,
            acquires={},
            assigns=[],
            returns=set(),
            calibration_id="",
        )

        assert executable.programs == [dummy_program]
