# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
from dataclasses import asdict, dataclass, is_dataclass

from qat_config import serializer
from qat_config.serializer import json_dumps, json_loads


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
            "$type": str(type(test_obj)), "$dataclass": True, "$data": asdict(test_obj)
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
            "$type": str(type(custom_field)), "$data": custom_field.__dict__
        }
        formatted_json = {
            "$type": str(type(test_obj)), "$dataclass": True, "$data": formatted_json_data
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
