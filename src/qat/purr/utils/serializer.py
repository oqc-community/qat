# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023-2025 Oxford Quantum Circuits Ltd
import json
from typing import TYPE_CHECKING

import compiler_config.serialiser as legacy_serialiser
import numpy as np

from qat.purr.compiler.devices import QuantumComponent
from qat.purr.compiler.instructions import Acquire

if TYPE_CHECKING:
    from qat.purr.compiler.hardware_models import QuantumHardwareModel


# Set of common methods so we don't have to add/remove the custom serializer.
def json_dumps(*args, serializable_types=None, **kwargs):
    kwargs.setdefault("cls", CustomQatJsonEncoder)
    return json.dumps(*args, serializable_types=serializable_types, **kwargs)


def json_loads(
    *args, serializable_types=None, model: "QuantumHardwareModel" = None, **kwargs
):
    kwargs.setdefault("cls", CustomQatJsonDecoder)
    return json.loads(*args, serializable_types=serializable_types, model=model, **kwargs)


def json_dump(*args, serializable_types=None, **kwargs):
    kwargs.setdefault("cls", CustomQatJsonEncoder)
    return json.dump(*args, serializable_types=serializable_types, **kwargs)


def json_load(
    *args, serializable_types=None, model: "QuantumHardwareModel" = None, **kwargs
):
    kwargs.setdefault("cls", CustomQatJsonDecoder)
    return json.load(*args, serializable_types=serializable_types, model=model, **kwargs)


class CustomQatJsonDecoder(legacy_serialiser.CustomJsonDecoder):
    def __init__(self, *args, model=None, **kwargs):
        self.model: "QuantumHardwareModel" = model
        super().__init__(*args, **kwargs)
        self.object_hook = self.default

    def default(self, obj):
        if not isinstance(obj, dict):
            return obj
        elif obj.get("type", False) == "numpyarray":
            if "value" in obj:
                array = np.frombuffer(
                    bytearray.fromhex(obj["value"]), dtype=np.dtype(obj["dtype"])
                )
                return array.reshape(obj["shape"])
            return np.array(obj["list"])

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

        return super().default(obj)


class CustomQatJsonEncoder(legacy_serialiser.CustomJSONEncoder):
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

    def default(self, obj):
        # TODO: Acquire is a special wrapper component, not an actual component. Have a few too many special-cases
        #   for it now, think about reverting its special status.
        if isinstance(obj, QuantumComponent) and not isinstance(obj, Acquire):
            return {"$component_id": obj.full_id()}
        elif isinstance(obj, np.ndarray):
            return {
                "type": "numpyarray",
                "dtype": obj.dtype.name,
                "shape": obj.shape,
                "value": obj.tobytes().hex(),
            }
        return super().default(obj)
