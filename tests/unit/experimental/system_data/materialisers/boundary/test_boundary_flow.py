# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import json

import pytest

import qat.experimental.system_data.materialisers.boundary as boundary_api
import qat.experimental.system_data.materialisers.plugins as plugin_api
from qat.experimental.system_data.canonical.schema import CanonicalSystemData
from qat.experimental.system_data.materialisers.boundary import materialise
from qat.experimental.system_data.materialisers.errors import (
    SourceIntegrityError,
    SourceValidationError,
    UnsupportedSourceError,
    UnsupportedSourceVersionError,
)
from qat.experimental.system_data.materialisers.types import SourceType
from qat.model.target_data import TargetData


class _OptionalAdditionalData(plugin_api.SourceAdditionalDataModel):
    target_data: TargetData | None = None


class _RequiredAdditionalData(plugin_api.SourceAdditionalDataModel):
    required_value: str


class _PluginWithModel:
    source_type = SourceType.PURR

    def __init__(
        self,
        *,
        source_version: str,
        additional_data_model,
        on_verify=None,
        on_materialise=None,
    ):
        self.source_version = source_version
        self.additional_data_model = additional_data_model
        self._on_verify = on_verify or (lambda _payload: None)
        self._on_materialise = on_materialise

    def resolve_type_and_version(self, source_payload):
        _ = source_payload
        return self.source_type, self.source_version

    def verify_integrity(self, source_payload):
        self._on_verify(source_payload)

    def materialise(self, *, source_payload, source_version, additional_data):
        if self._on_materialise is None:
            return "ok"
        return self._on_materialise(
            source_payload=source_payload,
            source_version=source_version,
            additional_data=additional_data,
        )


def _set_plugins(monkeypatch, *plugins):
    monkeypatch.setattr(
        plugin_api,
        "_PLUGIN_REGISTRY",
        {(plugin.source_type, plugin.source_version): plugin for plugin in plugins},
    )


def _target_data() -> TargetData:
    return TargetData()


def _make_valid_source_payload(
    *,
    default_acquire_mode: str | list[str] | None = "integrator",
) -> dict:
    """Build a minimal synthetic PuRR-like payload for full flow tests."""

    return {
        "calibration_id": "synthetic-cal",
        "supported_acquire_modes": ["integrator", "raw"],
        "default_acquire_mode": default_acquire_mode,
        "repeat_limit": 100,
        "quantum_devices": {
            "qubit_0": {
                "id": "qubit_0",
                "index": 0,
                "measure_device": {"id": "res_0"},
                "pulse_channels": {
                    "drive": {
                        "pulse_channel": {
                            "id": "ch_drive_0",
                            "physical_channel": {"id": "port_q0"},
                            "frequency": 5.0e9,
                            "scale": 1.0,
                        }
                    }
                },
            },
            "res_0": {
                "id": "res_0",
                "pulse_channels": {
                    "measure": {
                        "pulse_channel": {
                            "id": "ch_measure_0",
                            "physical_channel": {"id": "port_r0"},
                            "frequency": 6.0e9,
                        }
                    },
                    "acquire": {
                        "pulse_channel": {
                            "id": "ch_acquire_0",
                            "physical_channel": {"id": "port_r0"},
                            "frequency": 6.0e9,
                        }
                    },
                },
            },
        },
        "pulse_channels": {},
        "physical_channels": {
            "port_q0": {
                "id": "port_q0",
                "sample_time": 1.0e-9,
                "acquire_allowed": False,
                "baseband": {"id": "bb_q"},
            },
            "port_r0": {
                "id": "port_r0",
                "sample_time": 1.0e-9,
                "acquire_allowed": True,
                "baseband": {"id": "bb_r"},
            },
        },
        "basebands": {
            "bb_q": {"id": "bb_q", "frequency": 5.0e9},
            "bb_r": {"id": "bb_r", "frequency": 6.0e9},
        },
        "qubit_direction_couplings": [],
    }


def test_materialise_rejects_legacy_target_data_argument():
    with pytest.raises(TypeError, match="unexpected keyword argument 'target_data'"):
        materialise(
            source_payload={"x": 1},
            target_data=TargetData(),
        )


def test_materialise_rejects_missing_required_additional_data(monkeypatch):
    _set_plugins(
        monkeypatch,
        _PluginWithModel(
            source_version="0.1.0",
            additional_data_model=_RequiredAdditionalData,
        ),
    )

    with pytest.raises(SourceValidationError, match="validation failed"):
        materialise(
            source_payload={"x": 1},
            source_additional_data={},
        )


def test_materialise_rejects_unexpected_additional_data_keys(monkeypatch):
    _set_plugins(
        monkeypatch,
        _PluginWithModel(
            source_version="0.1.0",
            additional_data_model=_OptionalAdditionalData,
        ),
    )

    with pytest.raises(SourceValidationError, match="validation failed"):
        materialise(
            source_payload={"x": 1},
            source_additional_data={
                "target_data": TargetData(),
                "extra": 1,
            },
        )


def test_materialise_rejects_invalid_additional_data_types(monkeypatch):
    _set_plugins(
        monkeypatch,
        _PluginWithModel(
            source_version="0.1.0",
            additional_data_model=_OptionalAdditionalData,
        ),
    )

    with pytest.raises(SourceValidationError, match="validation failed"):
        materialise(
            source_payload={"x": 1},
            source_additional_data={"target_data": "bad"},
        )


def test_materialise_allows_missing_optional_source_additional_data(monkeypatch):
    _set_plugins(
        monkeypatch,
        _PluginWithModel(
            source_version="0.1.0",
            additional_data_model=_OptionalAdditionalData,
        ),
    )

    result = materialise(
        source_payload={"x": 1},
        source_additional_data=None,
    )

    assert result == "ok"


def test_materialise_rejects_non_mapping_source_additional_data(monkeypatch):
    _set_plugins(
        monkeypatch,
        _PluginWithModel(
            source_version="0.1.0",
            additional_data_model=_OptionalAdditionalData,
        ),
    )

    with pytest.raises(SourceValidationError, match="must be a mapping"):
        materialise(
            source_payload={"x": 1},
            source_additional_data="bad",
        )


def test_materialise_wraps_unexpected_verify_integrity_failures(monkeypatch):
    def _boom(_payload):
        raise RuntimeError("boom")

    _set_plugins(
        monkeypatch,
        _PluginWithModel(
            source_version="0.1.0",
            additional_data_model=_OptionalAdditionalData,
            on_verify=_boom,
        ),
    )

    with pytest.raises(SourceIntegrityError, match="plugin_integrity_execution"):
        materialise(
            source_payload={"x": 1},
            source_additional_data={"target_data": _target_data()},
        )


def test_materialise_propagates_unsupported_source_version_errors(monkeypatch):
    def _raise_version_error(_payload):
        raise UnsupportedSourceVersionError.for_version(
            source_type="purr",
            source_version="9.9.9",
            supported_versions=["0.1.0"],
        )

    _set_plugins(
        monkeypatch,
        _PluginWithModel(
            source_version="0.1.0",
            additional_data_model=_OptionalAdditionalData,
            on_verify=_raise_version_error,
        ),
    )

    with pytest.raises(UnsupportedSourceVersionError, match="Unsupported source version"):
        materialise(
            source_payload={"x": 1},
            source_additional_data={"target_data": _target_data()},
        )


def test_materialise_raises_unsupported_version_when_plugin_lookup_misses(monkeypatch):
    monkeypatch.setattr(
        boundary_api,
        "resolve_source_descriptor",
        lambda _payload: (SourceType.PURR, "9.9.9"),
    )
    monkeypatch.setattr(
        boundary_api,
        "get_registered_source_versions",
        lambda _source: ["0.1.0"],
    )
    monkeypatch.setattr(
        boundary_api,
        "get_materialiser_plugin",
        lambda **_kwargs: None,
    )

    with pytest.raises(UnsupportedSourceVersionError, match="Unsupported source version"):
        materialise(source_payload={"x": 1}, source_additional_data={})


def test_materialise_flow_good_input():
    """End-to-end boundary flow should materialise canonical data for valid input."""

    result = materialise(
        source_payload=_make_valid_source_payload(),
        source_additional_data={"target_data": _target_data()},
    )

    assert isinstance(result, CanonicalSystemData)
    assert result.calibration_id is not None
    assert len(result.qubits) > 0
    assert len(result.channels) > 0


def test_materialise_end_to_end_from_qblox_calibration_fixture_minimal(testpath):
    fixture_path = testpath / "files" / "calibrations" / "qblox_calibration.json"
    with fixture_path.open(encoding="utf-8") as file_handle:
        source_payload = json.load(file_handle)

    result = materialise(
        source_payload=source_payload,
        source_additional_data={},
    )

    assert isinstance(result, CanonicalSystemData)
    assert result.calibration_id == ""
    assert result.acquire_limit == 10000


@pytest.mark.parametrize(
    "kwargs, expected_error",
    [
        (
            {
                "source_payload": {
                    "metadata": {
                        "source_type": "unsupported_source",
                        "source_version": "0.1.0",
                    }
                },
                "source_additional_data": {"target_data": TargetData()},
            },
            UnsupportedSourceError,
        ),
        (
            {
                "source_payload": {
                    "metadata": {
                        "source_type": "purr",
                        "source_version": "9.9.9",
                    }
                },
                "source_additional_data": {"target_data": TargetData()},
            },
            UnsupportedSourceVersionError,
        ),
        (
            {
                "source_payload": {
                    "metadata": {
                        "source_type": "purr",
                        "source_version": "0.1.0",
                    },
                    "quantum_devices": {},
                },
                "source_additional_data": {"target_data": TargetData()},
            },
            SourceValidationError,
        ),
    ],
)
def test_materialise_flow_bad_inputs_raise_structured_errors(kwargs, expected_error):
    """Bad boundary inputs should fail early with structured materialisation errors."""

    with pytest.raises(expected_error):
        materialise(**kwargs)


def test_materialise_flow_integrity_hook_failure_is_structured(monkeypatch):
    """Unexpected verifier failures should be wrapped as SourceIntegrityError."""

    plugin = _PluginWithModel(
        source_version="0.1.0",
        additional_data_model=_OptionalAdditionalData,
        on_verify=lambda _payload: (_ for _ in ()).throw(RuntimeError("integrity boom")),
    )
    _set_plugins(monkeypatch, plugin)

    with pytest.raises(SourceIntegrityError) as exc_info:
        materialise(
            source_payload={"x": 1},
            source_additional_data={"target_data": _target_data()},
        )

    error_payload = exc_info.value.to_dict()
    assert error_payload["code"] == "source_integrity_error"
    assert error_payload["details"]["check"] == "plugin_integrity_execution"


def test_materialise_flow_materialisation_error_from_verifier_is_passthrough(monkeypatch):
    """MaterialisationError subclasses from verifier should not be rewrapped."""

    def _failing_verify(_payload):
        raise SourceValidationError("invalid signature", source_type="purr", path="$")

    _set_plugins(
        monkeypatch,
        _PluginWithModel(
            source_version="0.1.0",
            additional_data_model=_OptionalAdditionalData,
            on_verify=_failing_verify,
        ),
    )

    with pytest.raises(SourceValidationError, match="invalid signature"):
        materialise(
            source_payload={"x": 1},
            source_additional_data={"target_data": _target_data()},
        )


def test_materialise_flow_missing_source_registry_entry_raises_unsupported_source(
    monkeypatch,
):
    """Missing source entry in plugin registry should raise UnsupportedSourceError."""

    _set_plugins(monkeypatch)

    with pytest.raises(UnsupportedSourceError):
        materialise(
            source_payload={"x": 1},
            source_additional_data={"target_data": _target_data()},
        )
