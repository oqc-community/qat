# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
import json
import re
import sys
from dataclasses import asdict, is_dataclass
from enum import Enum
from importlib import import_module
from json import JSONDecoder, JSONEncoder
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scc.compiler.hardware_models import QuantumHardwareModel


# Set of common methods so we don't have to add/remove the custom serializer.
def json_dumps(*args, serializable_types=None, **kwargs):
    kwargs.setdefault("cls", CustomJSONEncoder)
    return json.dumps(*args, serializable_types=serializable_types, **kwargs)


def json_loads(
    *args, serializable_types=None, model: "QuantumHardwareModel" = None, **kwargs
):
    kwargs.setdefault("cls", CustomJsonDecoder)
    return json.loads(*args, serializable_types=serializable_types, model=model, **kwargs)


def json_dump(*args, serializable_types=None, **kwargs):
    kwargs.setdefault("cls", CustomJSONEncoder)
    return json.dump(*args, serializable_types=serializable_types, **kwargs)


def json_load(
    *args, serializable_types=None, model: "QuantumHardwareModel" = None, **kwargs
):
    kwargs.setdefault("cls", CustomJsonDecoder)
    return json.load(*args, serializable_types=serializable_types, model=model, **kwargs)


class CustomJsonDecoder(JSONDecoder):
    def __init__(self, *args, serializable_types=None, model=None, **kwargs):
        self.serializable_types = serializable_types
        self.model: "QuantumHardwareModel" = model
        super().__init__(object_hook=self.default, *args, **kwargs)

    def default(self, obj):
        if not isinstance(obj, dict):
            return obj

        # Components are objects directly related to hardware and you never want to serialize them, so we re-link
        # upon deserialization.
        component_id = obj.get("$component_id")
        if component_id is not None:
            if self.model is None:
                raise ValueError(
                    "Attempted to deserialize object that requires re-linking to a "
                    "hardware model and we have no hardware model."
                )

            return self.model.get_device(component_id)

        obj_type = obj.get("$type")
        if obj_type is None:
            return obj

        if obj_type == "<class 'tuple'>":
            return tuple(obj["$data"])

        if self.serializable_types is not None:
            old_paths = ["scc.compiler.config"]
            for old_path in old_paths:
                if old_path in obj_type:
                    obj_type = obj_type.replace(old_path, "qat.purr.compiler.config")
                    break

            typ = self.serializable_types.get(obj_type)
            if typ is None:
                raise ValueError(f"Invalid type attempted to be serialized: {obj_type}.")
        else:
            typ = _get_type(obj_type)

        if issubclass(typ, Enum):
            return typ(obj["$value"])

        if "$data" in obj:
            data = obj["$data"]
            if obj.get("$dataclass", False):
                fields = self.default(data)
                return typ(**fields)
            elif isinstance(data, dict):
                new_obj = object.__new__(typ)
                new_obj.__dict__ = {key: self.default(value) for key, value in data.items()}
                return new_obj
            elif isinstance(data, str):
                return typ(data)
            else:
                return data

        return obj


class CustomJSONEncoder(JSONEncoder):
    """
    It is a customised JSON encoder, which allows the serialization of the more complex
    objects.

    There are four major cases, based on the provided object to be serialized:

    - if the type of the object is supported by the default :class:`JSONEncoder`, than
      the default method is used.
    - if the class of the object is a :class:`dataclass`, then the serialization will
      contain the name of the type, ``dataclass`` flag in order to help at the
      deserialization, and the dictionary of the fields.
    - if the object is none from the above, then the type name is saved, and the
      interior data using ``__dict__``.
    - if an exception is encountered from any cases from above (e.g. ``__dict__`` is not
      available in case of complex numbers), then the type name is saved, and the data
      is the string representation of the object.
    """

    def __init__(self, *args, serializable_types=None, **kwargs):
        self.serializable_types = serializable_types
        super().__init__(*args, **kwargs)

    def default(self, obj):
        typ = type(obj)
        if issubclass(typ, Enum):
            typ_str = f"<enum '{obj.__module__}.{typ.__name__}'>"
        else:
            typ_str = str(typ)

        if self.serializable_types is not None:
            if (
                typ_str not in self.serializable_types
                and type(obj).__module__ != "builtins"
            ):
                raise ValueError(f"Invalid type attempted to be serialized: {(type(obj))}.")

        try:
            from qat.purr.compiler.devices import QuantumComponent
            from qat.purr.compiler.instructions import Acquire

            # TODO: Acquire is a special wrapper component, not an actual component. Have a few too many special-cases
            #   for it now, think about reverting its special status.
            if isinstance(obj, QuantumComponent) and not isinstance(obj, Acquire):
                return {"$component_id": obj.full_id()}
            elif is_dataclass(obj):
                return {
                    "$type": typ_str,
                    "$dataclass": True,
                    "$data": self.default(asdict(obj)),
                }
            elif isinstance(obj, Enum):
                return {"$type": typ_str, "$value": obj.value}
            elif isinstance(obj, complex):
                return {"$type": typ_str, "$data": str(obj)}
            elif isinstance(obj, tuple):
                return {
                    "$type": typ_str,
                    "$data": tuple(self.default(val) for val in obj),
                }

            if hasattr(obj, "__dict__"):
                return {
                    "$type": typ_str,
                    "$data": {
                        key: self.default(value) for key, value in obj.__dict__.items()
                    },
                }
        except (TypeError, AttributeError):
            pass

        return obj


_type_name_matcher = r"<(?:class|enum) '((?:\w+\.)*)(\w+)'>"


def _get_type(s: str):
    match = re.match(_type_name_matcher, s)
    if match:
        namespace = match.group(1)
        if namespace != "":
            expanded_namespace = namespace[:-1].split(".")
            module_name = expanded_namespace.pop(0)
            imported_module = import_module(module_name)
            for item in expanded_namespace:
                module_name = f"{module_name}.{item}"
                try:
                    imported_module = getattr(imported_module, item)
                except AttributeError:
                    import_module(module_name)
                    imported_module = getattr(imported_module, item)
        else:
            imported_module = sys.modules["builtins"]
        try:
            return getattr(imported_module, match.group(2))
        except AttributeError:
            raise AttributeError(
                f"Class {s} not found in built-in modules or namespace '{namespace}'"
            )
    return None
