# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from types import SimpleNamespace

import pytest

from qat.experimental.system_data.materialisers.errors import (
    SourceConsistencyError,
    SourceValidationError,
    UnsupportedSourceVersionError,
)
from qat.experimental.system_data.materialisers.purr import materialise as purr_materialise
from qat.experimental.system_data.materialisers.purr.ingress.v0_1_0 import PurrIngressV010
from qat.experimental.system_data.materialisers.purr.materialisers.capabilities import (
    _build_acquire_modes,
    _build_reset_methods,
)
from qat.experimental.system_data.materialisers.purr.validators.common import (
    _collect_qubit_index_by_id,
    _collect_qubit_indices,
    _iter_device_pulse_views,
    _iter_indexed_quantum_devices,
    _iter_parsed_coupling_directions,
)
from qat.experimental.system_data.materialisers.purr.validators.signal_paths import (
    _validate_basebands,
    _validate_physical_channels,
    _validate_pulse_channel_references,
    _validate_top_level_collections,
)
from qat.model.target_data import TargetData


def _base_payload() -> dict:
    return {
        "calibration_id": "cal",
        "supported_acquire_modes": ["integrator"],
        "default_acquire_mode": "integrator",
        "quantum_devices": {
            "Q0": {
                "id": "Q0",
                "index": 0,
                "pulse_channels": {
                    "drive": {
                        "pulse_channel": {
                            "id": "Q0.drive",
                            "physical_channel": {"id": "p_q0"},
                            "frequency": 5e9,
                        }
                    }
                },
            },
            "R0": {
                "id": "R0",
                "pulse_channels": {
                    "measure": {
                        "pulse_channel": {
                            "id": "R0.measure",
                            "physical_channel": {"id": "p_r0"},
                            "frequency": 6e9,
                        }
                    }
                },
            },
        },
        "pulse_channels": {"global": {"physical_channel": "p_q0"}},
        "physical_channels": {
            "p_q0": {"id": "p_q0", "sample_time": 1e-9, "acquire_allowed": False},
            "p_r0": {"id": "p_r0", "sample_time": 1e-9, "acquire_allowed": True},
            "raw": "bad",
        },
        "basebands": {
            "bb0": {"frequency": 5e9},
            "bad": "bad",
        },
        "qubit_direction_couplings": [{"direction": [0, 1]}],
    }


def test_capability_helpers_cover_append_and_default_resolution_paths():
    modes, default_mode = _build_acquire_modes(["integrator"], "scope")
    assert [mode.type for mode in modes] == ["integrator", "scope"]
    assert default_mode == "scope"

    methods, resolved_default = _build_reset_methods(["active", "passive"], "bad", 1e-6)
    assert {method.type for method in methods} == {"active", "passive"}
    for method in methods:
        assert method.operation_name is not None
    assert resolved_default == "passive"


def test_common_validators_helpers_cover_sparse_and_mixed_shapes():
    dto = PurrIngressV010.model_validate(
        _base_payload()
        | {
            "physical_channels": {"p": {"sample_time": 1e-9}},
            "basebands": {"bb": {"frequency": 5e9}},
        }
    )

    views = list(_iter_device_pulse_views(dto))
    assert len(views) >= 2

    indexed = list(_iter_indexed_quantum_devices(dto))
    assert len(indexed) == 1

    assert _collect_qubit_indices(dto) == {0}
    assert _collect_qubit_index_by_id(dto) == {"Q0": 0}

    assert list(_iter_parsed_coupling_directions(dto.qubit_direction_couplings)) == [(0, 1)]


def test_common_validator_helpers_skip_invalid_entry_shapes():
    payload = _base_payload() | {
        "quantum_devices": {
            "Q0": {"id": "Q0", "index": 0, "pulse_channels": {"raw": "bad"}},
            "Q1": {"id": "Q1", "pulse_channels": {}},
        },
        "qubit_direction_couplings": [{"direction": [0, 1]}],
        "physical_channels": {"p0": {"sample_time": 1e-9}},
        "basebands": {"bb": {"frequency": 5e9}},
    }
    dto = PurrIngressV010.model_validate(payload)

    views = list(_iter_device_pulse_views(dto))
    assert views == []

    indexed = list(_iter_indexed_quantum_devices(dto))
    assert [entry[0] for entry in indexed] == ["Q0"]

    assert list(_iter_parsed_coupling_directions(dto.qubit_direction_couplings)) == [(0, 1)]
    assert list(
        _iter_parsed_coupling_directions(["bad", {"direction": [0]}, {"direction": [0, 1]}])
    ) == [(0, 1)]

    fake_dto = SimpleNamespace(
        quantum_devices={
            "raw": "bad",
            "q0": {"pulse_channels": "bad"},
            "q1": {"pulse_channels": {"raw": "bad"}},
        }
    )
    assert list(_iter_device_pulse_views(fake_dto)) == []


def test_signal_path_validators_cover_top_level_and_entry_shape_errors():
    with pytest.raises(SourceConsistencyError, match="contains no quantum devices"):
        _validate_top_level_collections(
            quantum_devices={},
            physical_channels={"p": {}},
            basebands={"bb": {}},
        )

    with pytest.raises(SourceConsistencyError, match="contains no physical channels"):
        _validate_top_level_collections(
            quantum_devices={"Q0": {}},
            physical_channels={},
            basebands={"bb": {}},
        )

    with pytest.raises(SourceConsistencyError, match="contains no basebands"):
        _validate_top_level_collections(
            quantum_devices={"Q0": {}},
            physical_channels={"p": {}},
            basebands={},
        )

    with pytest.raises(SourceValidationError, match="entry must be a mapping"):
        _validate_physical_channels({"raw": "bad"})

    with pytest.raises(SourceValidationError, match="entry must be a mapping"):
        _validate_basebands({"raw": "bad"})

    with pytest.raises(SourceValidationError, match="entry must be a mapping"):
        _validate_pulse_channel_references(
            pulse_channels={"raw": "bad"},
            physical_channel_ids=frozenset({"p0"}),
        )


def test_materialise_helper_functions_and_version_error_paths(monkeypatch):
    assert purr_materialise._is_qubit_device_payload({"index": 0})
    assert purr_materialise._is_qubit_device_payload({"measure_device": {}})
    assert not purr_materialise._is_qubit_device_payload({"id": "R0"})

    methods = purr_materialise._detect_supported_reset_methods(
        {
            "quantum_devices": {
                "raw": "bad",
                "Q0": {
                    "index": 0,
                    "pulse_channels": {"reset": {}},
                    "ddrop_reset": {},
                },
                "Q1": {"index": 1},
            }
        }
    )
    assert "ddrop" in methods
    assert "active" not in methods

    injected_preexisting_passive = purr_materialise._inject_supported_reset_methods(
        {
            "quantum_devices": {"Q0": {"index": 0, "pulse_channels": {"reset": {}}}},
            "supported_reset_methods": ["passive", 7],
            "default_reset_method": "passive",
            "passive_reset_time": 1e-6,
        }
    )
    assert injected_preexisting_passive["supported_reset_methods"].count("passive") == 1
    assert injected_preexisting_passive["default_reset_method"] == "passive"

    injected_existing_passive_without_detection = (
        purr_materialise._inject_supported_reset_methods(
            {
                "quantum_devices": {},
                "supported_reset_methods": ["passive"],
                "default_reset_method": "passive",
                "passive_reset_time": 1e-6,
            }
        )
    )
    assert injected_existing_passive_without_detection["supported_reset_methods"] == [
        "passive"
    ]

    injected = purr_materialise._inject_supported_reset_methods(
        {
            "quantum_devices": {"Q0": {"index": 0, "pulse_channels": {"reset": {}}}},
            "supported_reset_methods": ["active"],
            "default_reset_method": "invalid",
            "passive_reset_time": 1e-6,
        }
    )
    assert "passive" in injected["supported_reset_methods"]
    assert injected["default_reset_method"] in injected["supported_reset_methods"]

    target_data = TargetData()
    enriched = purr_materialise._inject_target_data_fields(
        {
            "physical_channels": {
                "q": {"acquire_allowed": False},
                "r": {"acquire_allowed": True},
                "raw": "bad",
            }
        },
        target_data,
    )
    assert isinstance(enriched["physical_channels"]["q"], dict)
    assert isinstance(enriched["physical_channels"]["r"], dict)
    assert enriched["physical_channels"]["raw"] == "bad"

    enriched_no_channels = purr_materialise._inject_target_data_fields({}, target_data)
    assert "physical_channels" not in enriched_no_channels

    shaped = purr_materialise._inject_native_waveform_shapes(
        {"physical_channels": {"p0": {}, "raw": "bad"}},
        ["square"],
    )
    assert shaped["physical_channels"]["p0"]["native_waveform_shapes"] == ("square",)
    assert shaped["physical_channels"]["raw"] == "bad"

    with pytest.raises(UnsupportedSourceVersionError):
        purr_materialise.PurrMaterialiserV010().materialise(
            adapted_payload={},
            source_version="9.9.9",
        )

    with pytest.raises(SourceValidationError, match="ingress DTO validation failed"):
        purr_materialise.PurrMaterialiserV010().materialise(
            adapted_payload={},
            source_version="9.9.9",
            strict_version_check=False,
        )

    with pytest.raises(SourceValidationError, match="ingress DTO validation failed"):
        purr_materialise.PurrMaterialiserV010().materialise(
            adapted_payload={"x": 1},
            source_version="0.1.0",
        )


def test_split_combined_readout_channels_splits_top_level_macq():
    node = {
        "pulse_channels": {
            "macq": {"pulse_channel": {"id": "R0.macq"}},
            "drive": {"pulse_channel": {"id": "Q0.drive"}},
        }
    }

    result = purr_materialise._split_combined_readout_channels(node)

    pulse_channels = result["pulse_channels"]
    assert "macq" not in pulse_channels
    assert pulse_channels["measure"] == {"pulse_channel": {"id": "R0.macq"}}
    assert pulse_channels["acquire"] == {"pulse_channel": {"id": "R0.macq"}}
    # Unrelated channels are preserved untouched.
    assert pulse_channels["drive"] == {"pulse_channel": {"id": "Q0.drive"}}


def test_split_combined_readout_channels_recurses_into_nested_structures():
    node = {
        "quantum_devices": {
            "Q0": {
                "id": "Q0",
                "measure_device": {
                    "pulse_channels": {"macq": {"pulse_channel": {"id": "R0.macq"}}}
                },
            }
        },
        "resonators": [{"pulse_channels": {"macq": {"pulse_channel": {"id": "R1.macq"}}}}],
    }

    result = purr_materialise._split_combined_readout_channels(node)

    inline_channels = result["quantum_devices"]["Q0"]["measure_device"]["pulse_channels"]
    assert set(inline_channels) == {"measure", "acquire"}

    list_channels = result["resonators"][0]["pulse_channels"]
    assert set(list_channels) == {"measure", "acquire"}


def test_split_combined_readout_channels_leaves_untouched_and_does_not_mutate():
    node = {
        "pulse_channels": {"measure": {"id": "m"}, "acquire": {"id": "a"}},
        "scalar": 7,
        "list": [1, "two", 3.0],
    }
    original = {
        "pulse_channels": {"measure": {"id": "m"}, "acquire": {"id": "a"}},
        "scalar": 7,
        "list": [1, "two", 3.0],
    }

    result = purr_materialise._split_combined_readout_channels(node)

    # Resonators without a combined ``macq`` channel are left untouched.
    assert result["pulse_channels"] == {"measure": {"id": "m"}, "acquire": {"id": "a"}}
    assert result["scalar"] == 7
    assert result["list"] == [1, "two", 3.0]
    # Non-dict/list scalars pass straight through.
    assert purr_materialise._split_combined_readout_channels("bad") == "bad"
    assert purr_materialise._split_combined_readout_channels(5) == 5
    # The input node is not mutated.
    assert node == original


def test_split_combined_readout_channels_preserves_existing_measure_and_acquire():
    node = {
        "pulse_channels": {
            "macq": {"id": "macq"},
            "measure": {"id": "existing-measure"},
        }
    }

    result = purr_materialise._split_combined_readout_channels(node)

    pulse_channels = result["pulse_channels"]
    assert "macq" not in pulse_channels
    # Existing role channels win over the split (``setdefault`` semantics).
    assert pulse_channels["measure"] == {"id": "existing-measure"}
    assert pulse_channels["acquire"] == {"id": "macq"}


def test_prepare_ingress_splits_combined_readout_channels():
    payload = _base_payload() | {
        "quantum_devices": {
            "Q0": {
                "id": "Q0",
                "index": 0,
                "pulse_channels": {
                    "drive": {
                        "pulse_channel": {
                            "id": "Q0.drive",
                            "physical_channel": {"id": "p_q0"},
                            "frequency": 5e9,
                        }
                    }
                },
            },
            "R0": {
                "id": "R0",
                "pulse_channels": {
                    "macq": {
                        "pulse_channel": {
                            "id": "R0.macq",
                            "physical_channel": {"id": "p_r0"},
                            "frequency": 6e9,
                        }
                    }
                },
            },
        },
        "physical_channels": {
            "p_q0": {"id": "p_q0", "sample_time": 1e-9, "acquire_allowed": False},
            "p_r0": {"id": "p_r0", "sample_time": 1e-9, "acquire_allowed": True},
        },
        "basebands": {"bb0": {"frequency": 5e9}},
    }
    dto = PurrIngressV010.model_validate(payload)

    enriched = purr_materialise.PurrMaterialiserV010().prepare_ingress(
        adapted_payload=payload,
        source_ingress_dto=dto,
    )

    resonator_channels = enriched["quantum_devices"]["R0"]["pulse_channels"]
    assert "macq" not in resonator_channels
    assert set(resonator_channels) == {"measure", "acquire"}
    # The drive channel on the qubit is unaffected by the split.
    assert set(enriched["quantum_devices"]["Q0"]["pulse_channels"]) == {"drive"}


def test_materialiser_preserves_explicit_empty_configuration_lists():
    materialiser = purr_materialise.PurrMaterialiserV010(
        supported_acquire_modes=[],
        native_waveform_shapes=[],
    )

    assert materialiser._supported_acquire_modes == []
    assert materialiser._native_waveform_shapes == []


def test_materialise_enrichment_validation_error_path(monkeypatch):
    valid_adapted = _base_payload() | {
        "physical_channels": {
            "p_q0": {"sample_time": 1e-9, "acquire_allowed": False},
            "p_r0": {"sample_time": 1e-9, "acquire_allowed": True},
        },
        "basebands": {"bb0": {"frequency": 5e9}},
    }

    monkeypatch.setattr(
        purr_materialise,
        "validate_purr_ingress_graph",
        lambda _dto: None,
    )

    def _invalid_inject(_payload, _target_data):
        return {"broken": "payload"}

    monkeypatch.setattr(purr_materialise, "_inject_target_data_fields", _invalid_inject)

    with pytest.raises(SourceValidationError, match="could not be prepared"):
        purr_materialise.PurrMaterialiserV010().materialise(
            adapted_payload=valid_adapted,
            source_version="0.1.0",
        )


def test_materialise_default_args_and_supported_reset_branch_permutations(monkeypatch):
    injected = purr_materialise._inject_supported_reset_methods(
        {
            "quantum_devices": {},
            "supported_reset_methods": ["active", "passive"],
            "default_reset_method": "active",
            "passive_reset_time": -1.0,
        }
    )
    assert injected["default_reset_method"] == "active"

    captured = {}

    valid_adapted = _base_payload() | {
        "physical_channels": {
            "p_q0": {"id": "p_q0", "sample_time": 1e-9, "acquire_allowed": False},
            "p_r0": {"id": "p_r0", "sample_time": 1e-9, "acquire_allowed": True},
        },
        "basebands": {"bb0": {"frequency": 5e9}},
    }

    monkeypatch.setattr(purr_materialise, "validate_purr_ingress_graph", lambda _dto: None)
    configured_acquire_modes = ["integrator"]

    def _capture_top_level(self, *, dto, source_version):
        dto.supported_acquire_modes.append("scope")
        captured["source_version"] = source_version
        captured["supported_acquire_modes"] = dto.supported_acquire_modes
        captured["physical_channels"] = dto.physical_channels
        captured["operation_builder_type"] = self._operation_builder_type
        captured["extra_operations"] = self._extra_operations
        return object()

    monkeypatch.setattr(
        purr_materialise.PurrMaterialiserV010, "assemble", _capture_top_level
    )

    result = purr_materialise.PurrMaterialiserV010(
        supported_acquire_modes=configured_acquire_modes,
        native_waveform_shapes=["gaussian"],
    ).materialise(
        adapted_payload=valid_adapted,
        source_version="0.1.0",
    )
    assert result is not None
    assert captured["source_version"] == "0.1.0"
    assert captured["operation_builder_type"] is purr_materialise.DefaultOperationBuilder
    assert captured["extra_operations"] == ()
    assert configured_acquire_modes == ["integrator"]


def test_materialiser_subclass_builder_hooks_are_used(monkeypatch):
    valid_adapted = _base_payload() | {
        "physical_channels": {
            "p_q0": {"id": "p_q0", "sample_time": 1e-9, "acquire_allowed": False},
            "p_r0": {"id": "p_r0", "sample_time": 1e-9, "acquire_allowed": True},
        },
        "basebands": {"bb0": {"frequency": 5e9}},
    }
    calls = []

    class _HookedMaterialiser(purr_materialise.PurrMaterialiserV010):
        def build_qubits(self, **kwargs):
            calls.append("qubits")
            return super().build_qubits(**kwargs)

        def build_operation_builder(self, **kwargs):
            calls.append("operation_builder")
            return super().build_operation_builder(**kwargs)

        def build_channels(self, **kwargs):
            calls.append("channels")
            return super().build_channels(**kwargs)

        def build_couplings(self, **kwargs):
            calls.append("couplings")
            return super().build_couplings(**kwargs)

    monkeypatch.setattr(purr_materialise, "validate_purr_ingress_graph", lambda _dto: None)
    # Use valid_adapted which has quantum_devices and physical_channels but no couplings.
    valid_adapted["qubit_direction_couplings"] = []

    result = _HookedMaterialiser().materialise(
        adapted_payload=valid_adapted,
        source_version="0.1.0",
    )

    assert result is not None
    assert calls == ["channels", "qubits", "operation_builder", "couplings"]
