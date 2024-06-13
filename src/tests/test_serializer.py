# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd

import math
from dataclasses import asdict, dataclass, is_dataclass

import pytest
from numpy import isclose

from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler.builders import InstructionBuilder, SerialiserBackend
from qat.purr.compiler.instructions import PhaseShift
from qat.purr.compiler.runtime import get_builder
from qat.purr.utils import serializer
from qat.purr.utils.serializer import json_dumps, json_loads


@dataclass
class TestDataClass:
    field1: int
    field2: str
    field3: dict
    field4: str = "Default"


class TestClass:
    def __init__(self, name: str):
        self.field1 = name
        self.field2 = True


class TestDerivedClass(TestClass):
    def __init__(self, value: int):
        super().__init__("DerivedClass")
        self.field3 = value


@dataclass
class TestDataClassWithCustomClass:
    field1: int
    field2: str
    field3: TestClass


class TestCustomJSONEncoder:
    def test_serialize_dict(self):
        test_obj = {"key1": "data1", "key2": "data2"}
        serialized_obj = json_dumps(test_obj)
        loaded_obj = json_loads(serialized_obj)
        assert test_obj == loaded_obj

    def test_serialize_int(self):
        test_obj = 34
        serialized_obj = json_dumps(test_obj)
        loaded_obj = json_loads(serialized_obj)
        assert test_obj == loaded_obj

    def test_serialize_list(self):
        test_obj = [1, 2, 3]
        serialized_obj = json_dumps(test_obj)
        loaded_obj = json_loads(serialized_obj)
        assert test_obj == loaded_obj

    def test_serialize_boolean(self):
        test_obj = True
        serialized_obj = json_dumps(test_obj)
        loaded_obj = json_loads(serialized_obj)
        assert test_obj == loaded_obj

    def test_serialize_none(self):
        test_obj = None
        serialized_obj = json_dumps(test_obj)
        loaded_obj = json_loads(serialized_obj)
        assert loaded_obj is None

    def test_serialize_dataclass(self):
        test_obj = TestDataClass(23, "Hello World!", {"key1": "data1", "key2": "data2"})
        serialized_obj = json_dumps(test_obj)
        loaded_obj: TestDataClass = json_loads(serialized_obj)
        assert test_obj.field1 == loaded_obj.field1
        assert test_obj.field2 == loaded_obj.field2
        assert test_obj.field3 == loaded_obj.field3
        assert test_obj.field4 == loaded_obj.field4
        assert test_obj.__class__ == loaded_obj.__class__

    def test_serialize_basic_custom_class(self):
        test_obj = TestClass("TestParameter")
        serialized_obj = json_dumps(test_obj)
        loaded_obj = json_loads(serialized_obj)
        assert test_obj.field1 == loaded_obj.field1
        assert test_obj.field2 == loaded_obj.field2
        assert test_obj.__class__ == loaded_obj.__class__

    def test_serialize_inherited_custom_class(self):
        test_obj = TestDerivedClass(34)
        serialized_obj = json_dumps(test_obj)
        loaded_obj = json_loads(serialized_obj)
        assert test_obj.field1, loaded_obj.field1
        assert test_obj.field2, loaded_obj.field2
        assert test_obj.field3, loaded_obj.field3
        assert test_obj.__class__, loaded_obj.__class__

    def test_serialize_dataclass_with_custom_field(self):
        test_obj = TestDataClassWithCustomClass(
            23, "Hello World!", TestClass("TestParameter")
        )
        serialized_obj = json_dumps(test_obj)
        loaded_obj = json_loads(serialized_obj)
        assert test_obj.field1 == loaded_obj.field1
        assert test_obj.field2 == loaded_obj.field2
        assert test_obj.field3.field1 == loaded_obj.field3.field1
        assert test_obj.field3.field2 == loaded_obj.field3.field2
        assert test_obj.__class__ == loaded_obj.__class__
        assert test_obj.field3.__class__ == loaded_obj.field3.__class__

    def test_serialize_complex(self):
        test_obj = complex(1, 3)
        serialized_obj = json_dumps(test_obj)
        loaded_obj = json_loads(serialized_obj)
        assert test_obj == loaded_obj


class TestGetType:
    def test_get_type_int(self):
        assert int == serializer._get_type(str(type(34)))

    def test_get_type_complex(self):
        assert complex == serializer._get_type(str(type(complex(3, 4))))

    def test_get_type_custom_class(self):
        assert TestClass == serializer._get_type(str(type(TestClass("something"))))

    class TestNestedClass:
        def __init__(self, name: str):
            self.field1 = name

    def test_get_type_nested_custom_class(self):
        assert TestGetType.TestNestedClass == serializer._get_type(
            str(type(TestGetType.TestNestedClass("something")))
        )


class TestDeserialize:
    def test_deserialize_dict(self):
        test_obj = {"key1": "data1", "key2": "data2"}
        serialized_obj = json_dumps(test_obj)
        loaded_obj = json_loads(serialized_obj)
        assert test_obj == loaded_obj

    def test_deserialize_int(self):
        test_obj = 34
        serialized_obj = json_dumps(test_obj)
        loaded_obj = json_loads(serialized_obj)
        assert test_obj == loaded_obj

    def test_deserialize_list(self):
        test_obj = [1, 2, 3]
        serialized_obj = json_dumps(test_obj)
        loaded_obj = json_loads(serialized_obj)
        assert test_obj == loaded_obj

    def test_deserialize_boolean(self):
        test_obj = True
        serialized_obj = json_dumps(test_obj)
        loaded_obj = json_loads(serialized_obj)
        assert test_obj == loaded_obj

    def test_deserialize_none(self):
        test_obj = None
        serialized_obj = json_dumps(test_obj)
        loaded_obj = json_loads(serialized_obj)
        assert loaded_obj is None

    def test_deserialize_dataclass(self):
        test_obj = TestDataClass(23, "Hello World!", {"key1": "data1", "key2": "data2"})
        formatted_json = {
            "$type": str(type(test_obj)),
            "$dataclass": True,
            "$data": asdict(test_obj),
        }
        serialized_obj = json_dumps(formatted_json)
        loaded_obj = json_loads(serialized_obj)
        assert type(loaded_obj) == TestDataClass
        assert is_dataclass(loaded_obj)
        assert asdict(loaded_obj) == asdict(test_obj)

    def test_deserialize_basic_custom_class(self):
        test_obj = TestClass("TestParameter")
        formatted_json = {"$type": str(type(test_obj)), "$data": test_obj.__dict__}
        serialized_obj = json_dumps(formatted_json)
        loaded_obj = json_loads(serialized_obj)
        assert type(loaded_obj) == TestClass
        assert loaded_obj.__dict__ == test_obj.__dict__

    def test_deserialize_inherited_custom_class(self):
        test_obj = TestDerivedClass(34)
        formatted_json = {"$type": str(type(test_obj)), "$data": test_obj.__dict__}
        serialized_obj = json_dumps(formatted_json)
        loaded_obj = json_loads(serialized_obj)
        assert type(loaded_obj) == TestDerivedClass
        assert loaded_obj.__dict__ == test_obj.__dict__

    def test_deserialize_dataclass_with_custom_field(self):
        custom_field = TestClass("TestParameter")
        test_obj = TestDataClassWithCustomClass(23, "Hello World!", custom_field)
        formatted_json_data = asdict(test_obj)
        formatted_json_data.pop("field3")
        formatted_json_data["field3"] = {
            "$type": str(type(custom_field)),
            "$data": custom_field.__dict__,
        }
        formatted_json = {
            "$type": str(type(test_obj)),
            "$dataclass": True,
            "$data": formatted_json_data,
        }
        serialized_obj = json_dumps(formatted_json)
        loaded_obj = json_loads(serialized_obj)
        assert type(loaded_obj) == TestDataClassWithCustomClass
        assert is_dataclass(loaded_obj)
        assert loaded_obj.field1 == test_obj.field1
        assert loaded_obj.field2 == test_obj.field2
        assert type(loaded_obj.field3) == TestClass
        assert loaded_obj.field3.__dict__ == custom_field.__dict__

    def test_deserialize_complex(self):
        test_obj = complex(1, 3)
        formatted_json = {"$type": str(type(test_obj)), "$data": str(test_obj)}
        serialized_obj = json_dumps(formatted_json)
        loaded_obj = json_loads(serialized_obj)
        assert type(loaded_obj) == complex
        assert loaded_obj == test_obj


@pytest.mark.parametrize("backend", [SerialiserBackend.json, SerialiserBackend.ujson])
class TestBuilderSerialiser:
    """
    Note here, that typically doing a direct float comparison is not meaningful in a test.
    Here however, we need to ensure that the integrity of the floats is exactly preserved across serialisation.
    """

    floating_point_num = math.sqrt(math.pi) / 2.14324354

    @pytest.fixture
    def builder(self):
        hardware = get_default_echo_hardware(4)

        hardware.get_qubit(0).get_drive_channel().frequency = self.floating_point_num

        builder = get_builder(hardware)
        builder.X(hardware.get_qubit(0))
        builder.U(
            hardware.get_qubit(0),
            theta=self.floating_point_num,
            phi=self.floating_point_num,
            lamb=self.floating_point_num,
        )
        return builder

    def test_floating_point_numbers_hardware(self, backend, builder):
        """
        Testing that different serialisation backends respect floating point numbers in hardware
        """
        string = builder.serialize(backend=backend)

        new_builder = InstructionBuilder.deserialize(string, backend=backend)

        assert (
            new_builder.model.get_qubit(0).get_drive_channel().frequency
            == self.floating_point_num
        )

    def test_floating_point_numbers_instructions(self, backend, builder):
        """
        Testing that different serialisation backends respect floating point numbers in hardware

        """
        string = builder.serialize(backend=backend)

        new_builder = InstructionBuilder.deserialize(string, backend=backend)

        for instruction in new_builder.instructions:
            if isinstance(instruction, PhaseShift):
                phase = instruction.phase
                if isclose(phase, self.floating_point_num):
                    # There are phase instructions that aren't just the pure rotation, easiest way to find them
                    assert instruction.phase == self.floating_point_num

    def test_object_reconstruction(self, backend, builder):
        string = builder.serialize(backend=backend)
        new_builder = InstructionBuilder.deserialize(string, backend=backend)
        model = new_builder.model

        qubit_route_1 = model.get_qubit(0)
        print(model.physical_channels)
        qubit_route_2 = model.get_physical_channel("CH1").related_qubit

        qubit_2 = model.get_qubit(1)

        assert qubit_route_1.id == qubit_route_2.id
        assert qubit_2.id != qubit_route_1.id
