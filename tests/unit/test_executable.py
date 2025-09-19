# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import pytest

from qat.executables import (
    BaseExecutable,
    BatchedExecutable,
    ChannelExecutable,
    Executable,
    ParameterizedExecutable,
)


class MockExecutableA(Executable):
    super_cool_instruction: int

    @property
    def acquires(self) -> list:
        return []


class MockExecutableB(ChannelExecutable):
    super_cool_instruction: int

    @property
    def acquires(self) -> list:
        return []


class MockExecutableC(BaseExecutable):
    super_cool_instruction: int = 5


class TestExecutableSerialization:
    @pytest.mark.parametrize(
        "executable, deserialize_cls",
        [
            (MockExecutableA(super_cool_instruction=42), Executable),
            (MockExecutableA(super_cool_instruction=42), BaseExecutable),
            (
                MockExecutableB(super_cool_instruction=42, channel_data={"ch1": {}}),
                ChannelExecutable,
            ),
            (
                MockExecutableB(super_cool_instruction=42, channel_data={"ch1": {}}),
                BaseExecutable,
            ),
            (
                MockExecutableB(super_cool_instruction=42, channel_data={"ch1": {}}),
                Executable,
            ),
            (MockExecutableC(super_cool_instruction=42), BaseExecutable),
        ],
    )
    def test_serialize_deserialize_roundtrip_works(self, executable, deserialize_cls):
        serialized = executable.serialize()
        deserialized = deserialize_cls.deserialize(serialized)
        assert deserialized == executable
        assert isinstance(deserialized, type(executable))

    @pytest.mark.parametrize(
        "executable, deserialize_cls",
        [
            (MockExecutableA(super_cool_instruction=42), ChannelExecutable),
            (MockExecutableC(super_cool_instruction=42), Executable),
        ],
    )
    def test_serialize_deserialize_raises_value_error(self, executable, deserialize_cls):
        serialized = executable.serialize()
        with pytest.raises(TypeError):
            deserialize_cls.deserialize(serialized)


class TestBatchedExecutable:
    @pytest.mark.parametrize(
        "executable, deserialize_cls",
        [
            (MockExecutableA(super_cool_instruction=42), BatchedExecutable),
            (MockExecutableA(super_cool_instruction=42), BaseExecutable),
            (
                MockExecutableB(super_cool_instruction=42, channel_data={"ch1": {}}),
                BatchedExecutable,
            ),
            (
                MockExecutableB(super_cool_instruction=42, channel_data={"ch1": {}}),
                BaseExecutable,
            ),
            (MockExecutableC(super_cool_instruction=42), BatchedExecutable),
            (MockExecutableC(super_cool_instruction=42), BaseExecutable),
        ],
    )
    def test_batched_executable_serialization_roundtrip_works(
        self, executable, deserialize_cls
    ):
        batched = BatchedExecutable(
            executables=[
                ParameterizedExecutable(executable=executable, parameters={"param1": 1}),
                ParameterizedExecutable(executable=executable, parameters={"param2": 2}),
            ]
        )
        serialized = batched.serialize()
        deserialized = deserialize_cls.deserialize(serialized)
        assert deserialized == batched
        assert isinstance(deserialized, BatchedExecutable)
        for i in range(2):
            assert isinstance(deserialized.executables[i].executable, type(executable))
            assert deserialized.executables[i].parameters == {f"param{i + 1}": i + 1}
            assert deserialized.executables[i].executable.super_cool_instruction == 42

    def test_batched_executable_raises_value_error_on_mixed_types(self):
        with pytest.raises(
            ValueError,
            match="All executables in a BatchedExecutable must be of the same type.",
        ):
            BatchedExecutable(
                executables=[
                    ParameterizedExecutable(
                        executable=MockExecutableA(super_cool_instruction=42),
                        parameters={"param1": 1},
                    ),
                    ParameterizedExecutable(
                        executable=MockExecutableC(super_cool_instruction=42),
                        parameters={"param2": 2},
                    ),
                ]
            )

    def test_serialization_of_parameters(self):
        parameters = {
            "param1": [1.0, 2.0, 3.0],
            "param2": [1 + 1j, 2 + 2j, 3 + 3j],
        }

        executables = [
            ParameterizedExecutable(
                executable=MockExecutableA(super_cool_instruction=42),
                parameters={"param1": val1, "param2": val2},
            )
            for val1 in parameters["param1"]
            for val2 in parameters["param2"]
        ]

        batched = BatchedExecutable(shape=(3, 3), executables=executables)
        serialized = batched.serialize()
        deserialized = BatchedExecutable.deserialize(serialized)
        assert isinstance(deserialized, BatchedExecutable)
        assert deserialized.shape == (3, 3)
        for i, exe in enumerate(deserialized.executables):
            assert executables[i] == exe
