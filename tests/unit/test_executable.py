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
                ParameterizedExecutable(executable=executable, params={"param1": 1}),
                ParameterizedExecutable(executable=executable, params={"param2": 2}),
            ]
        )
        serialized = batched.serialize()
        deserialized = deserialize_cls.deserialize(serialized)
        assert deserialized == batched
        assert isinstance(deserialized, BatchedExecutable)
        for i in range(2):
            assert isinstance(deserialized.executables[i].executable, type(executable))
            assert deserialized.executables[i].params == {f"param{i + 1}": i + 1}
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
                        params={"param1": 1},
                    ),
                    ParameterizedExecutable(
                        executable=MockExecutableC(super_cool_instruction=42),
                        params={"param2": 2},
                    ),
                ]
            )
