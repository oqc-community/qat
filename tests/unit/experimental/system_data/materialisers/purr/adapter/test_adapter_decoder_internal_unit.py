# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest

from qat.experimental.system_data.materialisers.errors import SourceValidationError
from qat.experimental.system_data.materialisers.purr import adapter
from qat.experimental.system_data.materialisers.purr.decoder import (
    PurrJsonpickleDecoder,
    _DeferredReference,
    decode_jsonpickle_payload,
)


def _minimal_top_level_payload():
    return {
        "quantum_devices": {},
        "pulse_channels": {},
        "physical_channels": {},
        "basebands": {},
    }


def test_adapter_profile_detection_and_normalisation_helpers():
    assert adapter._detect_source_profile({"py/state": {}}) == "jsonpickle"
    assert adapter._detect_source_profile({"x": 1}) == "plain"

    payload = _minimal_top_level_payload()
    adapter._normalise_default_acquire_mode(payload)
    assert payload.get("default_acquire_mode") is None

    payload = _minimal_top_level_payload()
    payload["default_acquire_mode"] = ["integrator"]
    adapter._normalise_default_acquire_mode(payload)
    assert payload["default_acquire_mode"] == "integrator"

    with pytest.raises(SourceValidationError, match="Unsupported default_acquire_mode"):
        adapter._normalise_default_acquire_mode(
            {
                **_minimal_top_level_payload(),
                "default_acquire_mode": [1],
            }
        )


def test_adapter_normalise_top_level_and_errors():
    normalised = adapter._normalise_top_level(_minimal_top_level_payload())
    assert normalised["qubit_direction_couplings"] == []

    with pytest.raises(SourceValidationError, match="missing required top-level keys"):
        adapter._normalise_top_level({"quantum_devices": {}})


def test_adapter_reference_and_projection_helpers_cover_cyclic_and_repeated_nodes():
    assert adapter._build_reference_stub({"id": "x"}) == {"id": "x"}
    assert adapter._build_reference_stub({"instrument_id": "x"}) == {"instrument_id": "x"}
    assert adapter._build_reference_stub({"x": 1}) == {"_adapter_reference": "mapping"}
    assert adapter._build_reference_stub([1, 2]) == []
    assert adapter._build_reference_stub(3) == 3

    assert adapter._resolve_next_root_key(
        path=("quantum_devices", "Q0"), root_key=None
    ) == (
        "quantum_devices",
        "Q0",
    )
    assert adapter._resolve_next_root_key(path=("x",), root_key=("a", "b")) == ("a", "b")

    seen = set()
    by_root = {}
    active = adapter._get_active_seen_nodes(
        root_key=None,
        seen_nodes=seen,
        seen_nodes_by_root=by_root,
    )
    assert active is seen

    active_root = adapter._get_active_seen_nodes(
        root_key=("quantum_devices", "Q0"),
        seen_nodes=seen,
        seen_nodes_by_root=by_root,
    )
    assert isinstance(active_root, set)

    assert adapter._is_repeated_or_cyclic(
        node_id=1, stack=frozenset({1}), active_seen=set()
    )
    assert adapter._is_repeated_or_cyclic(node_id=1, stack=frozenset(), active_seen={1})
    assert not adapter._is_repeated_or_cyclic(
        node_id=1, stack=frozenset(), active_seen=set()
    )

    cyc = {}
    cyc["self"] = cyc
    projected = adapter._project_acyclic_payload(node=cyc)
    assert projected["self"] == {"_adapter_reference": "mapping"}


def test_adapt_purr_payload_covers_input_validation_and_source_hint(monkeypatch):
    with pytest.raises(SourceValidationError, match="root must be a dictionary"):
        adapter.adapt_purr_payload("bad")

    with pytest.raises(SourceValidationError, match="payload is empty"):
        adapter.adapt_purr_payload({})

    plain = adapter.adapt_purr_payload(_minimal_top_level_payload())
    assert plain["qubit_direction_couplings"] == []

    def _fake_decode(_payload, **_kwargs):
        return _minimal_top_level_payload()

    monkeypatch.setattr(adapter, "decode_jsonpickle_payload", _fake_decode)
    decoded = adapter.adapt_purr_payload({"py/object": "fake.Source", "py/state": {}})
    assert decoded["_adapter_source_hint"] == "fake.Source"


def test_adapt_purr_payload_converts_custom_pulse_weights_to_samples_list():
    payload = {
        "py/state": {
            "quantum_devices": {
                "Q0": {
                    "measure_acquire": {
                        "weights": {
                            "py/object": "qat.purr.compiler.instructions.CustomPulse",
                            "samples": {
                                "py/object": "numpy.ndarray",
                                "value": [
                                    {"py/object": "numpy.float64", "value": 1.25},
                                    {
                                        "py/object": "builtins.complex",
                                        "py/newargs": {"py/tuple": [0.5, -0.25]},
                                    },
                                ],
                            },
                        }
                    }
                }
            },
            "pulse_channels": {},
            "physical_channels": {},
            "basebands": {},
        }
    }

    decoded = adapter.adapt_purr_payload(payload)

    weights = decoded["quantum_devices"]["Q0"]["measure_acquire"]["weights"]
    assert weights["object_type"] == "qat.purr.compiler.instructions.CustomPulse"
    assert weights["samples"] == [1.25, complex(0.5, -0.25)]


def test_decoder_error_paths_for_additional_marker_shapes():
    decoded_set = decode_jsonpickle_payload({"py/state": {"s": {"py/set": [1, 2, 3]}}})
    assert decoded_set["s"] == [1, 2, 3]

    with pytest.raises(SourceValidationError, match="Malformed py/tuple payload"):
        decode_jsonpickle_payload({"py/tuple": "bad"})

    with pytest.raises(SourceValidationError, match="Malformed py/set payload"):
        decode_jsonpickle_payload({"py/set": "bad"})

    with pytest.raises(SourceValidationError, match="Malformed py/reduce args payload"):
        decode_jsonpickle_payload({"py/reduce": [{"py/type": "x"}, {"py/tuple": "bad"}]})

    with pytest.raises(SourceValidationError, match="Unsupported py/reduce arity"):
        decode_jsonpickle_payload(
            {
                "py/reduce": [
                    {"py/type": "qat.ir.instruction_basetypes.AcquireMode"},
                    {"py/tuple": ["a", "b"]},
                ]
            }
        )

    with pytest.raises(SourceValidationError, match="Malformed builtins.complex wrapper"):
        decode_jsonpickle_payload({"py/object": "builtins.complex", "py/newargs": []})

    with pytest.raises(SourceValidationError, match="Unsupported numpy scalar value type"):
        decode_jsonpickle_payload(
            {"py/object": "numpy.float64", "value": {"py/tuple": [1]}}
        )

    with pytest.raises(SourceValidationError, match="Malformed py/obj_ref_id reference"):
        decode_jsonpickle_payload({"py/state": {}, "py/obj_ref_id": "bad"})


def test_decoder_deferred_resolution_handles_cycles_and_non_container_nodes():
    decoder = PurrJsonpickleDecoder()
    decoder._references[1] = {"value": 10}
    decoder._deferred_references.append(_DeferredReference(ref_id=1, path="$.x"))

    cyclic_list: list[object] = []
    cyclic_list.append(cyclic_list)
    cyclic_list.append(_DeferredReference(ref_id=1, path="$.x"))

    cyclic_dict: dict[str, object] = {"self": None}
    cyclic_dict["self"] = cyclic_dict
    cyclic_dict["ref"] = _DeferredReference(ref_id=1, path="$.x")

    root = {"list": cyclic_list, "dict": cyclic_dict, "raw": 7}
    decoder._resolve_deferred_references(root)

    assert root["list"][1] == {"value": 10}
    assert root["dict"]["ref"] == {"value": 10}
    assert root["raw"] == 7


def test_project_acyclic_payload_collapses_repeated_sequence_nodes():
    repeated = [1, 2]
    payload = {"items": [repeated, repeated]}

    projected = adapter._project_acyclic_payload(node=payload)

    assert projected["items"][0] == [1, 2]
    assert projected["items"][1] == []


def test_decoder_internal_decode_helpers_cover_remaining_branches():
    decoder = PurrJsonpickleDecoder()

    with pytest.raises(SourceValidationError, match="missing value"):
        decoder._decode_numpy_scalar({"py/object": "numpy.float64"}, path="$")

    decoded_list = decoder._decode_node([1, {"k": 2}], path="$.list")
    assert decoded_list == [1, {"k": 2}]
