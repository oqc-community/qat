# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Compiler-owned decoder for jsonpickle-like PuRR payloads.

The PuRR calibration fixtures are serialised object graphs rather than plain transport JSON.
This module decodes the supported jsonpickle markers into ordinary Python data while
preserving reference relationships needed by the boundary adapter.
"""

import base64
import math
import struct
import zlib
from typing import Any

from qat.experimental.system_data.materialisers.errors import SourceValidationError

_JSON_PRIMITIVE_TYPES = (str, int, float, bool, type(None))
_KNOWN_MARKER_KEYS = {
    "py/state",
    "py/object",
    "py/tuple",
    "py/set",
    "py/newargs",
    "py/reduce",
    "py/id",
    "py/obj_ref_id",
    "py/type",
    "value",
    "dtype",
}

_ALLOWED_REDUCE_TARGET_TYPES = {
    "qat.ir.instruction_basetypes.AcquireMode",
    "qat.purr.compiler.devices.ChannelType",
    "qat.purr.compiler.devices.PulseShapeType",
    "qat.purr.compiler.instructions.AcquireMode",
    "qblox_instruments.types.ClusterType",
}
_ALLOWED_REDUCE_TARGET_SUFFIXES = set()

_CUSTOM_PULSE_OBJECT_TYPE = "qat.purr.compiler.instructions.CustomPulse"


class _DeferredReference:
    """Placeholder used when a ``py/id`` reference appears before its declaration."""

    def __init__(self, ref_id: int, path: str):
        self.ref_id = ref_id
        self.path = path


class PurrJsonpickleDecoder:
    """Stateful decoder for PuRR jsonpickle-like payloads.

    The decoder owns per-run state such as object reference tables and deferred lookups,
    allowing the public decode API to stay pure and stateless from the caller's point of
    view.
    """

    def __init__(
        self,
        *,
        extra_reduce_target_types: set[str] | None = None,
        extra_reduce_target_suffixes: set[str] | None = None,
    ) -> None:
        self._references: dict[int, Any] = {}
        self._deferred_references: list[_DeferredReference] = []
        self._allowed_reduce_target_types = set(_ALLOWED_REDUCE_TARGET_TYPES)
        self._allowed_reduce_target_suffixes = set(_ALLOWED_REDUCE_TARGET_SUFFIXES)

        if extra_reduce_target_types:
            self._allowed_reduce_target_types.update(extra_reduce_target_types)
        if extra_reduce_target_suffixes:
            self._allowed_reduce_target_suffixes.update(extra_reduce_target_suffixes)

    def decode(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Decode one raw PuRR payload into plain Python data.

        The returned structure may still contain shared references, but it no longer
        contains jsonpickle control markers.
        """

        decoded_payload = self._decode_node(payload, path="$")
        self._resolve_deferred_references(decoded_payload)
        return self._normalise_root(decoded_payload)

    def _raise_decode_error(
        self, message: str, *, path: str, details: dict[str, Any]
    ) -> None:
        """Raise a structured decode-stage validation error."""

        raise SourceValidationError(
            message,
            source_type="purr",
            path=path,
            details=details,
        )

    def _decode_numpy_scalar(
        self, node: dict[str, Any], *, path: str
    ) -> float | int | complex:
        """Decode wrapped numpy scalar payloads to plain Python numeric values.

        Some PuRR fixtures wrap numeric values directly, while others wrap an inner
        ``builtins.complex`` payload. Both variants are supported here.
        """

        if "value" not in node:
            self._raise_decode_error(
                "Malformed numpy scalar wrapper: missing value.",
                path=path,
                details={"keys": tuple(sorted(node.keys()))},
            )

        value = node["value"]
        decoded_value = (
            self._decode_node(value, path=f"{path}.value")
            if isinstance(value, dict)
            else value
        )
        if not isinstance(decoded_value, int | float | complex):
            self._raise_decode_error(
                "Unsupported numpy scalar value type.",
                path=path,
                details={"value_type": type(decoded_value).__name__},
            )

        return decoded_value

    def _coerce_numeric_sequence(self, value: Any, *, path: str) -> list[Any]:
        """Coerce decoded sequence payloads to a list of numeric/complex values."""

        if isinstance(value, tuple):
            value = list(value)

        if not isinstance(value, list):
            self._raise_decode_error(
                "Unsupported numpy array payload shape.",
                path=path,
                details={"value_type": type(value).__name__},
            )

        for idx, item in enumerate(value):
            if not isinstance(item, int | float | complex):
                self._raise_decode_error(
                    "Unsupported numpy array element type.",
                    path=f"{path}[{idx}]",
                    details={"value_type": type(item).__name__},
                )

        return value

    def _decode_numpy_array(self, node: dict[str, Any], *, path: str) -> list[Any]:
        """Decode wrapped numpy-array payloads into plain Python sample lists."""

        if "value" in node:
            raw_value = node["value"]
            decoded_value = self._decode_node(raw_value, path=f"{path}.value")
            return self._coerce_numeric_sequence(decoded_value, path=f"{path}.value")

        if "values" in node:
            raw_values = node["values"]
            if isinstance(raw_values, str):
                return self._decode_numpy_array_blob(node, path=path)
            decoded_values = self._decode_node(raw_values, path=f"{path}.values")
            return self._coerce_numeric_sequence(decoded_values, path=f"{path}.values")

        state_payload = node.get("py/state")
        if isinstance(state_payload, dict):
            decoded_state = self._decode_node(state_payload, path=f"{path}.py/state")
            if isinstance(decoded_state, dict):
                for key in ("value", "values", "samples", "data"):
                    if key not in decoded_state:
                        continue
                    decoded_values = decoded_state[key]
                    return self._coerce_numeric_sequence(
                        decoded_values,
                        path=f"{path}.py/state.{key}",
                    )

            self._raise_decode_error(
                "Unsupported numpy array state payload shape.",
                path=path,
                details={"state_type": type(decoded_state).__name__},
            )

        self._raise_decode_error(
            "Malformed numpy array wrapper.",
            path=path,
            details={"keys": tuple(sorted(node.keys()))},
        )

    def _decode_numpy_array_blob(self, node: dict[str, Any], *, path: str) -> list[Any]:
        """Decode base64/zlib-encoded numpy-array values payloads."""

        encoded_values = node.get("values")
        dtype = node.get("dtype")
        shape = node.get("shape")
        byteorder = node.get("byteorder", "<")

        if not isinstance(encoded_values, str):
            self._raise_decode_error(
                "Malformed numpy array values payload.",
                path=f"{path}.values",
                details={"value_type": type(encoded_values).__name__},
            )

        if not isinstance(dtype, str):
            self._raise_decode_error(
                "Malformed numpy array dtype payload.",
                path=f"{path}.dtype",
                details={"value_type": type(dtype).__name__},
            )

        if not isinstance(shape, list) or not all(
            isinstance(dim, int) and dim >= 0 for dim in shape
        ):
            self._raise_decode_error(
                "Malformed numpy array shape payload.",
                path=f"{path}.shape",
                details={"value": shape},
            )

        if not isinstance(byteorder, str) or byteorder not in {"<", ">", "="}:
            self._raise_decode_error(
                "Malformed numpy array byteorder payload.",
                path=f"{path}.byteorder",
                details={"value": byteorder},
            )

        try:
            packed = base64.b64decode(encoded_values)
        except Exception as exc:
            self._raise_decode_error(
                "Malformed numpy array base64 payload.",
                path=f"{path}.values",
                details={"error_type": type(exc).__name__},
            )

        try:
            payload_bytes = zlib.decompress(packed)
        except zlib.error:
            # Some fixtures may store uncompressed byte payloads.
            payload_bytes = packed

        total_elems = math.prod(shape)

        if dtype == "complex128":
            expected_bytes = total_elems * 16
            if len(payload_bytes) != expected_bytes:
                self._raise_decode_error(
                    "Unexpected numpy array byte length for complex128.",
                    path=f"{path}.values",
                    details={
                        "expected_bytes": expected_bytes,
                        "actual_bytes": len(payload_bytes),
                    },
                )

            unpacked = struct.unpack(
                f"{byteorder}{2 * total_elems}d",
                payload_bytes,
            )
            return [
                complex(unpacked[idx], unpacked[idx + 1])
                for idx in range(0, len(unpacked), 2)
            ]

        if dtype == "float64":
            expected_bytes = total_elems * 8
            if len(payload_bytes) != expected_bytes:
                self._raise_decode_error(
                    "Unexpected numpy array byte length for float64.",
                    path=f"{path}.values",
                    details={
                        "expected_bytes": expected_bytes,
                        "actual_bytes": len(payload_bytes),
                    },
                )

            return list(struct.unpack(f"{byteorder}{total_elems}d", payload_bytes))

        self._raise_decode_error(
            "Unsupported numpy array dtype.",
            path=f"{path}.dtype",
            details={"dtype": dtype},
        )

    def _is_numpy_array_wrapper(self, node: dict[str, Any]) -> bool:
        """Return True when a numpy wrapper likely encodes an array-like payload."""

        object_type = node.get("py/object")
        if not isinstance(object_type, str):
            return False

        if object_type.endswith("ndarray") or object_type.endswith(".array"):
            return True

        return "values" in node or "py/state" in node

    def _decode_custom_pulse(self, node: dict[str, Any], *, path: str) -> dict[str, Any]:
        """Decode CustomPulse payloads to a stable plain dictionary representation."""

        decoded_state = {
            key: self._decode_node(value, path=f"{path}.{key}")
            for key, value in node.items()
            if not key.startswith("py/")
        }

        decoded_custom_pulse: dict[str, Any] = {
            "object_type": _CUSTOM_PULSE_OBJECT_TYPE,
        }
        for key in (
            "samples",
            "ignore_channel_scale",
            "quantum_targets",
            "channel",
        ):
            if key in decoded_state:
                decoded_custom_pulse[key] = decoded_state[key]

        samples = decoded_custom_pulse.get("samples")
        if isinstance(samples, tuple):
            samples = list(samples)
            decoded_custom_pulse["samples"] = samples

        if samples is not None and not isinstance(samples, list):
            self._raise_decode_error(
                "Unsupported CustomPulse samples payload shape.",
                path=f"{path}.samples",
                details={"samples_type": type(samples).__name__},
            )

        return decoded_custom_pulse

    def _decode_reduce(self, node: dict[str, Any], *, path: str) -> Any:
        """Decode the allowlisted subset of jsonpickle ``py/reduce`` payloads.

        Reducers are intentionally restricted to the enum-like targets currently observed in
        supported PuRR fixtures so unsupported object construction fails early and visibly.
        """

        reduce_payload = node.get("py/reduce")
        if not isinstance(reduce_payload, list) or len(reduce_payload) != 2:
            self._raise_decode_error(
                "Malformed py/reduce payload.",
                path=path,
                details={"reduce_type": type(reduce_payload).__name__},
            )

        target_node, args_node = reduce_payload
        target_type = target_node.get("py/type") if isinstance(target_node, dict) else None
        if not isinstance(args_node, dict) or not isinstance(
            args_node.get("py/tuple"), list
        ):
            self._raise_decode_error(
                "Malformed py/reduce args payload.",
                path=path,
                details={"target": target_node},
            )

        reduce_args = args_node["py/tuple"]
        if len(reduce_args) != 1:
            self._raise_decode_error(
                "Unsupported py/reduce arity.",
                path=path,
                details={"target": target_node, "arity": len(reduce_args)},
            )

        # Keep a narrow, explicit allowlist of known reducer targets observed in
        # supported PuRR payloads. For external integrations, the extension tooling can be
        # used to define additional allowed targets.
        if self._is_allowed_reduce_target(target_type):
            return reduce_args[0]

        self._raise_decode_error(
            "Unsupported py/reduce target.",
            path=path,
            details={"target": target_node},
        )

    def _is_allowed_reduce_target(self, target_type: Any) -> bool:
        """Return True when a ``py/reduce`` target type is allowlisted."""

        if not isinstance(target_type, str):
            return False

        if target_type in self._allowed_reduce_target_types:
            return True

        target_suffix = target_type.rsplit(".", maxsplit=1)[-1]
        return target_suffix in self._allowed_reduce_target_suffixes

    def _decode_builtin_complex(self, node: dict[str, Any], *, path: str) -> complex:
        """Decode ``builtins.complex`` payloads encoded with ``py/newargs``."""

        newargs = node.get("py/newargs")
        if not isinstance(newargs, dict):
            self._raise_decode_error(
                "Malformed builtins.complex wrapper: missing py/newargs.",
                path=path,
                details={"keys": tuple(sorted(node.keys()))},
            )

        tuple_payload = newargs.get("py/tuple")
        if (
            not isinstance(tuple_payload, list)
            or len(tuple_payload) != 2
            or not isinstance(tuple_payload[0], int | float)
            or not isinstance(tuple_payload[1], int | float)
        ):
            self._raise_decode_error(
                "Malformed builtins.complex py/newargs payload.",
                path=path,
                details={"newargs": newargs},
            )

        return complex(tuple_payload[0], tuple_payload[1])

    def _decode_mapping(self, node: dict[str, Any], *, path: str) -> Any:
        """Decode mapping nodes while handling supported jsonpickle control keys.

        This method is where reference objects, reducer payloads, wrapper objects, and
        ordinary mappings are distinguished.
        """

        if "py/id" in node and len(node) == 1:
            ref_id = node["py/id"]
            if not isinstance(ref_id, int):
                self._raise_decode_error(
                    "Malformed py/id reference.",
                    path=path,
                    details={"ref_type": type(ref_id).__name__},
                )

            if ref_id not in self._references:
                deferred = _DeferredReference(ref_id=ref_id, path=path)
                self._deferred_references.append(deferred)
                return deferred

            return self._references[ref_id]

        if node.get("py/object") == _CUSTOM_PULSE_OBJECT_TYPE:
            decoded = self._decode_custom_pulse(node, path=path)
            self._register_obj_ref(node=node, decoded=decoded, path=path)
            return decoded

        if "py/state" in node:
            decoded = self._decode_node(node["py/state"], path=f"{path}.py/state")
            self._register_obj_ref(node=node, decoded=decoded, path=path)
            return decoded

        if "py/tuple" in node:
            tuple_payload = node["py/tuple"]
            if not isinstance(tuple_payload, list):
                self._raise_decode_error(
                    "Malformed py/tuple payload.",
                    path=path,
                    details={"tuple_type": type(tuple_payload).__name__},
                )
            decoded = [
                self._decode_node(item, path=f"{path}.py/tuple[{idx}]")
                for idx, item in enumerate(tuple_payload)
            ]
            self._register_obj_ref(node=node, decoded=decoded, path=path)
            return decoded

        if "py/set" in node:
            set_payload = node["py/set"]
            if not isinstance(set_payload, list):
                self._raise_decode_error(
                    "Malformed py/set payload.",
                    path=path,
                    details={"set_type": type(set_payload).__name__},
                )
            decoded = [
                self._decode_node(item, path=f"{path}.py/set[{idx}]")
                for idx, item in enumerate(set_payload)
            ]
            self._register_obj_ref(node=node, decoded=decoded, path=path)
            return decoded

        if "py/reduce" in node:
            decoded = self._decode_reduce(node, path=path)
            self._register_obj_ref(node=node, decoded=decoded, path=path)
            return decoded

        if node.get("py/object") == "builtins.complex" and "py/newargs" in node:
            decoded = self._decode_builtin_complex(node, path=path)
            self._register_obj_ref(node=node, decoded=decoded, path=path)
            return decoded

        if node.get("py/object", "").startswith("numpy."):
            if self._is_numpy_array_wrapper(node):
                decoded = self._decode_numpy_array(node, path=path)
            elif "value" in node:
                decoded = self._decode_numpy_scalar(node, path=path)
            else:
                decoded = None

            if decoded is not None:
                self._register_obj_ref(node=node, decoded=decoded, path=path)
                return decoded

        if node and all(key.startswith("py/") for key in node):
            unknown_keys = sorted(
                key for key in node.keys() if key not in _KNOWN_MARKER_KEYS
            )
            self._raise_decode_error(
                "Unsupported jsonpickle marker object.",
                path=path,
                details={
                    "keys": tuple(sorted(node.keys())),
                    "unknown_keys": tuple(unknown_keys),
                },
            )

        decoded: dict[str, Any] = {}
        self._register_obj_ref(node=node, decoded=decoded, path=path)
        for key, value in node.items():
            decoded[key] = self._decode_node(value, path=f"{path}.{key}")
        return decoded

    def _decode_node(self, node: Any, *, path: str) -> Any:
        """Recursively decode one node from the PuRR payload graph."""

        if isinstance(node, _JSON_PRIMITIVE_TYPES):
            return node

        if isinstance(node, list):
            return [
                self._decode_node(item, path=f"{path}[{idx}]")
                for idx, item in enumerate(node)
            ]

        if isinstance(node, dict):
            return self._decode_mapping(node, path=path)

        self._raise_decode_error(
            "Unsupported node type in payload.",
            path=path,
            details={"node_type": type(node).__name__},
        )

    def _normalise_root(self, decoded_payload: Any) -> dict[str, Any]:
        """Ensure the decoded payload root has the mapping shape expected downstream."""

        if not isinstance(decoded_payload, dict):
            self._raise_decode_error(
                "Decoded payload root must be a mapping.",
                path="$",
                details={"root_type": type(decoded_payload).__name__},
            )
        return decoded_payload

    def _register_obj_ref(self, *, node: dict[str, Any], decoded: Any, path: str) -> None:
        """Register a decoded object for later ``py/id`` reference resolution."""

        obj_ref_id = node.get("py/obj_ref_id")
        if obj_ref_id is None:
            return

        if not isinstance(obj_ref_id, int):
            self._raise_decode_error(
                "Malformed py/obj_ref_id reference.",
                path=path,
                details={"ref_type": type(obj_ref_id).__name__},
            )

        self._references[obj_ref_id] = decoded

    def _resolve_deferred_references(self, decoded_root: Any) -> None:
        """Resolve forward ``py/id`` references after the first decode traversal."""

        if not self._deferred_references:
            return

        visited_containers: set[int] = set()

        def resolve_node(node: Any) -> Any:
            if isinstance(node, _DeferredReference):
                if node.ref_id not in self._references:
                    self._raise_decode_error(
                        "Unresolved py/id reference.",
                        path=node.path,
                        details={"ref_id": node.ref_id},
                    )
                return self._references[node.ref_id]

            if isinstance(node, list):
                node_id = id(node)
                if node_id in visited_containers:
                    return node
                visited_containers.add(node_id)
                for idx, item in enumerate(node):
                    node[idx] = resolve_node(item)
                return node

            if isinstance(node, dict):
                node_id = id(node)
                if node_id in visited_containers:
                    return node
                visited_containers.add(node_id)
                for key, value in list(node.items()):
                    node[key] = resolve_node(value)
                return node

            return node

        resolve_node(decoded_root)


def decode_jsonpickle_payload(
    payload: dict[str, Any],
    *,
    extra_reduce_target_types: set[str] | None = None,
    extra_reduce_target_suffixes: set[str] | None = None,
) -> dict[str, Any]:
    """Decode a jsonpickle-like payload into plain Python data.

    :param payload: Raw parsed JSON payload.
    :param extra_reduce_target_types: Optional extra fully-qualified ``py/reduce``
        targets to allow at runtime.
    :param extra_reduce_target_suffixes: Optional extra terminal type-name suffixes
        to allow at runtime.
    :returns: Decoded payload with jsonpickle markers removed.
    """
    decoder = PurrJsonpickleDecoder(
        extra_reduce_target_types=extra_reduce_target_types,
        extra_reduce_target_suffixes=extra_reduce_target_suffixes,
    )
    return decoder.decode(payload)
