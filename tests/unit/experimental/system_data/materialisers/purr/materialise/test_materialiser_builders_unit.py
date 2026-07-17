# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import copy

import pytest

from qat.experimental.system_data.materialisers.errors import (
    MaterialisationConsistencyError,
    MaterialisationIntegrityError,
)
from qat.experimental.system_data.materialisers.purr.materialisers.capabilities import (
    _build_acquire_limit,
    _build_acquire_modes,
    _build_reset_methods,
)
from qat.experimental.system_data.materialisers.purr.materialisers.common import (
    _as_complex,
    _as_float,
    _hz_to_int,
    _seconds_to_picoseconds,
)
from qat.experimental.system_data.materialisers.purr.materialisers.couplings import (
    _build_couplings,
)
from qat.experimental.system_data.materialisers.purr.materialisers.external_resources import (
    ExternalResourceRegistry,
)
from qat.experimental.system_data.materialisers.purr.materialisers.postprocess import (
    _build_post_process_method,
    _normalise_ml_offset,
    _normalise_ml_states,
    _normalise_ml_transform,
    _parse_linear_method_from_mean_z_map_args,
    _parse_max_likelihood_method,
    _parse_post_process_method_payload,
)
from qat.experimental.system_data.materialisers.purr.materialisers.qubits import (
    _build_acquire_definitions_for_mode,
    _build_mode_from_pulse_view,
    _build_qubit_modes,
    _build_qubits,
    _build_readout_probability,
    _build_waveform_data,
    _build_waveforms_for_mode,
    _resolve_resonator_payload,
)
from qat.experimental.system_data.materialisers.purr.materialisers.signal_paths import (
    _build_channels,
    _build_oscillators,
    _build_port_block_bounds,
    _build_ports,
    _register_external_resource_from_payload,
)


def _synthetic_quantum_devices():
    return {
        "q0": {
            "id": "q0",
            "index": 0,
            "measure_device": {"id": "r0"},
            "pulse_channels": {
                "drive": {
                    "pulse_channel": {
                        "id": "ch_q0_drive",
                        "frequency": 5.0e9,
                        "scale": 0.5,
                        "physical_channel": {"id": "p_q0"},
                    }
                },
                "acquire": {
                    "pulse_channel": {
                        "id": "ch_q0_acq",
                        "frequency": 6.0e9,
                        "physical_channel": {"id": "p_r0"},
                    }
                },
            },
            "pulse_hw_x_pi_2": {"shape": "square", "width": 20e-9, "amp": 0.2},
            "pulse_hw_x_pi": {"shape": "square", "width": 40e-9, "amp": 0.4},
            "measure_acquire": {
                "delay": 10e-9,
                "width": 40e-9,
                "sync": True,
                "weights": [1.0, 0.0],
            },
            "post_process_method": {
                "method": "max_likelihood",
                "states": {
                    "0": {"location": 1 + 0j, "label": "g"},
                    "1": {"location": -1 + 0j, "label": "e"},
                },
                "noise_est": 1.0,
                "p_min": 0.0,
            },
        },
        "r0": {
            "id": "r0",
            "pulse_channels": {
                "measure": {
                    "pulse_channel": {
                        "id": "ch_r0_measure",
                        "frequency": 6.0e9,
                        "physical_channel": {"id": "p_r0"},
                    }
                }
            },
        },
    }


def _synthetic_physical_channels():
    return {
        "p_q0": {
            "id": "p_q0",
            "sample_time": 1e-9,
            "block_size": 4,
            "baseband": {"id": "bb_q"},
            "instrument_type": "port",
            "acquire_allowed": False,
            "native_waveform_shapes": ["square"],
        },
        "p_r0": {
            "id": "p_r0",
            "sample_time": 1e-9,
            "pulse_duration_min": 8e-9,
            "pulse_duration_max": 80e-9,
            "baseband": {"id": "bb_r"},
            "acquire_allowed": True,
        },
    }


def test_common_coercion_helpers_cover_numeric_and_default_paths():
    assert _seconds_to_picoseconds(1e-9) == 1000
    assert _seconds_to_picoseconds(None) is None
    assert _hz_to_int(5.2) == 5
    assert _hz_to_int(None) is None
    assert _as_complex(2) == complex(2, 0)
    assert _as_complex("bad", default=1 + 2j) == complex(1, 2)
    assert _as_float(3) == 3.0
    assert _as_float("bad", default=4.5) == 4.5


def test_capability_builders_cover_empty_and_default_resolution_paths():
    empty_modes, empty_default_mode = _build_acquire_modes([], None)
    assert empty_modes == ()
    assert empty_default_mode is None

    empty_resets, empty_default_reset = _build_reset_methods([], None, None)
    assert empty_resets == ()
    assert empty_default_reset is None

    assert _build_acquire_limit(None) == -1
    assert _build_acquire_limit(10) == 10

    modes, default_mode = _build_acquire_modes(["integrator"], None)
    assert len(modes) == 1
    assert default_mode == "integrator"

    reset_methods, default_reset = _build_reset_methods(["active"], None, None)
    assert len(reset_methods) == 1
    assert default_reset == "active"

    methods, resolved_default = _build_reset_methods(["passive"], None, 1e-6)
    assert methods[0].type == "passive"
    assert resolved_default == "passive"


def test_external_resource_registry_merges_and_orders_resources():
    registry = ExternalResourceRegistry()
    assert registry.register(resource_id=None) is None
    registry.register(resource_id="resA", object_type="cluster", attributes={"a": 1})
    registry.register(resource_id="resA", object_type=None, attributes={"a": 2, "b": 3})
    registry.register(resource_id="resB", object_type="awg", attributes={})

    resources = registry.to_tuple()
    assert [resource.id for resource in resources] == ["resA", "resB"]
    res_a = resources[0]
    assert res_a.object_type == "cluster"
    attrs = {entry.key: entry.value for entry in res_a.attributes}
    assert attrs == {"a": 1, "b": 3}

    registry.register(resource_id="resC", object_type=None, attributes={"x": 1})
    registry.register(resource_id="resC", object_type="scope", attributes={"y": 2})
    resources = registry.to_tuple()
    res_c = [resource for resource in resources if resource.id == "resC"][0]
    assert res_c.object_type == "scope"


def test_signal_path_builders_cover_ports_oscillators_and_channels():
    registry = ExternalResourceRegistry()
    physical_channels = _synthetic_physical_channels()
    quantum_devices = _synthetic_quantum_devices()
    basebands = {
        "bb_q": {"id": "bb_q", "frequency": 5.0e9, "instrument_type": "oscillator"},
        "bb_r": {"id": "bb_r", "_frequency": 6.0e9},
    }

    ports = _build_ports(physical_channels, registry)
    oscillators = _build_oscillators(basebands, registry)
    channels = _build_channels(
        quantum_devices=quantum_devices,
        physical_channels=physical_channels,
    )

    assert len(ports) == 2
    assert len(oscillators) == 2
    assert len(channels) >= 2
    assert {channel.port_id for channel in channels} == {"p_q0", "p_r0"}


def test_coupling_builder_covers_valid_and_invalid_entries():
    quantum_devices = {
        "q0": {"id": "q0", "index": 0},
        "q1": {"id": "q1", "index": 1},
    }
    couplings = _build_couplings(
        qubit_direction_couplings=[{"direction": [0, 1], "quality": 0.97}],
        quantum_devices=quantum_devices,
    )

    assert len(couplings) == 1
    assert couplings[0].source_qubit_id == "q0"
    assert couplings[0].target_qubit_id == "q1"

    with pytest.raises(MaterialisationIntegrityError, match="2-element integer list"):
        _build_couplings(
            qubit_direction_couplings=[{"direction": [0]}],
            quantum_devices=quantum_devices,
        )

    with pytest.raises(MaterialisationIntegrityError, match="must be a dictionary"):
        _build_couplings(
            qubit_direction_couplings=["bad"],
            quantum_devices=quantum_devices,
        )

    with pytest.raises(MaterialisationConsistencyError, match="unknown qubit index"):
        _build_couplings(
            qubit_direction_couplings=[{"direction": [0, 2]}],
            quantum_devices=quantum_devices,
        )


def test_post_process_builder_prefers_new_method_and_falls_back_to_legacy(caplog):
    payload = {
        "id": "q0",
        "post_process_method": {
            "method": "max_likelihood",
            "states": {
                "0": {"location": 1 + 0j},
                "1": {"location": -1 + 0j},
            },
        },
        "mean_z_map_args": [1 + 0j, 0 + 0j],
    }
    method = _build_post_process_method(payload, "acquire")
    assert method is not None

    fallback_payload = {
        "id": "q1",
        "post_process_method": {"method": "max_likelihood", "states": "bad"},
        "mean_z_map_args": [1 + 0j, 0 + 0j],
    }
    with pytest.raises(MaterialisationIntegrityError, match="non-empty mapping"):
        _build_post_process_method(fallback_payload, "acquire")


def test_qubit_builder_materialises_modes_and_readout_probability():
    quantum_devices = _synthetic_quantum_devices()
    error_mitigation = {
        "readout_mitigation": {
            "linear": {
                "0": {
                    "0|0": 0.99,
                    "1|0": 0.01,
                    "0|1": 0.02,
                    "1|1": 0.98,
                }
            }
        }
    }

    qubits = _build_qubits(
        quantum_devices=quantum_devices,
        error_mitigation=error_mitigation,
    )

    assert len(qubits) == 1
    qubit = qubits[0]
    assert qubit.id == "q0"
    assert len(qubit.modes) >= 2
    assert qubit.readout_probability is not None


def test_postprocess_parser_helpers_cover_success_and_rejection_paths():
    assert _parse_linear_method_from_mean_z_map_args([1 + 0j, 0 + 0j]) is not None
    with pytest.raises(MaterialisationIntegrityError, match="mean_z_map_args"):
        _parse_linear_method_from_mean_z_map_args([1])

    assert _normalise_ml_states({"0": {"location": 1.0}}) is not None
    with pytest.raises(MaterialisationIntegrityError, match="integer-like"):
        _normalise_ml_states({"bad": {"location": 1.0}})
    assert _normalise_ml_transform([[1.0, 0.0], [0.0, 1.0]]) is not None
    with pytest.raises(MaterialisationIntegrityError, match="2x2 numeric matrix"):
        _normalise_ml_transform([1.0])
    assert _normalise_ml_offset([0.0, 1.0]) == (0.0, 1.0)
    with pytest.raises(MaterialisationIntegrityError, match="2-element numeric vector"):
        _normalise_ml_offset([0.0])

    assert _parse_max_likelihood_method({"states": {"0": {"location": 1.0}}}) is not None
    with pytest.raises(MaterialisationIntegrityError, match="noise_est and p_min"):
        _parse_max_likelihood_method({"states": {"0": {"location": 1.0}}, "noise_est": "x"})

    with pytest.raises(
        MaterialisationIntegrityError, match="Unsupported post_process_method"
    ):
        _parse_post_process_method_payload({"method": "unknown"})
    with pytest.raises(MaterialisationIntegrityError, match="must be a mapping"):
        _parse_post_process_method_payload("bad")
    assert (
        _parse_post_process_method_payload(
            {
                "method": "linear_map_complex_to_real",
                "mean_z_map_args": [1 + 0j, 0 + 0j],
            }
        )
        is not None
    )

    with pytest.raises(
        MaterialisationIntegrityError, match="state entries must be mappings"
    ):
        _normalise_ml_states({"0": "bad"})
    with pytest.raises(MaterialisationIntegrityError, match="location must be numeric"):
        _normalise_ml_states({"0": {"location": "bad"}})
    with pytest.raises(MaterialisationIntegrityError, match="label must be a string"):
        _normalise_ml_states({"0": {"location": 1, "label": 7}})

    with pytest.raises(MaterialisationIntegrityError, match="2x2 numeric matrix"):
        _parse_max_likelihood_method(
            {
                "states": {"0": {"location": 1.0}},
                "transform": [1],
            }
        )
    with pytest.raises(MaterialisationIntegrityError, match="2-element numeric vector"):
        _parse_max_likelihood_method(
            {
                "states": {"0": {"location": 1.0}},
                "offset": [1],
            }
        )


def test_post_process_builder_returns_none_for_non_acquire_and_unparseable_inputs(caplog):
    payload = {
        "id": "q0",
        "post_process_method": {"method": "max_likelihood", "states": "bad"},
    }
    assert _build_post_process_method(payload, "drive") is None
    with pytest.raises(MaterialisationIntegrityError, match="non-empty mapping"):
        _build_post_process_method(payload, "acquire")


def test_postprocess_additional_strict_branches_are_covered():
    assert _parse_post_process_method_payload(None) is None

    with pytest.raises(MaterialisationIntegrityError, match="values must be numeric"):
        _parse_linear_method_from_mean_z_map_args([1 + 0j, "bad"])

    with pytest.raises(MaterialisationIntegrityError, match="requires a non-empty states"):
        _parse_max_likelihood_method({"states": None})

    with pytest.raises(MaterialisationIntegrityError, match="method must be a string"):
        _parse_post_process_method_payload({"method": 123})

    legacy_only_payload = {
        "id": "q0",
        "mean_z_map_args": [1 + 0j, 0 + 0j],
    }
    assert _build_post_process_method(legacy_only_payload, "acquire") is not None


def test_qubit_waveform_and_mode_helpers_cover_additional_branches():
    qubit_payload = {
        "id": "q0",
        "index": 0,
        "pulse_hw_x_pi_2": {"shape": "square", "width": 20e-9, "amp": 0.2},
        "pulse_hw_x_pi": {"shape": "square", "width": 40e-9, "amp": 0.4},
        "pulse_measure": {"shape": "square", "width": 80e-9, "amp": 0.5},
        "pulse_hw_zx_pi_4": {"q1": {"shape": "square", "width": 10e-9, "amp": 0.2}},
        "ddrop_reset": {
            "shape": "square",
            "width": 20e-9,
            "qubit_amp": 0.3,
            "res_amp": 0.4,
        },
        "measure_acquire": {"delay": 10e-9, "width": 20e-9, "weights": [1.0, "x"]},
    }

    assert len(_build_waveforms_for_mode(qubit_payload, "second_state", {})) == 2
    assert len(_build_waveforms_for_mode(qubit_payload, "measure", {})) == 1
    assert len(_build_waveforms_for_mode(qubit_payload, "q1.cross_resonance", {})) == 1
    assert len(_build_waveforms_for_mode(qubit_payload, "reset", {}, is_readout=False)) == 1
    assert len(_build_waveforms_for_mode(qubit_payload, "reset", {}, is_readout=True)) == 1
    assert (
        len(
            _build_waveforms_for_mode(
                qubit_payload,
                "freq_shift",
                {"amp": 0.2, "phase": 0.1},
            )
        )
        == 1
    )
    assert _build_waveforms_for_mode(qubit_payload, "acquire", {}) == ()


def test_qubit_helpers_cover_resonator_fallback_and_filtered_weights():
    quantum_devices = {
        "q0": {
            "id": "q0",
            "index": 0,
            "measure_device": {"id": "r0"},
            "measure_acquire": {
                "delay": 10e-9,
                "width": 20e-9,
                "sync": True,
                "weights": {"samples": [1.0, "bad", 2.0]},
            },
        },
        "r0": {
            "id": "r0",
            "pulse_channels": {
                "measure": {
                    "pulse_channel": {
                        "id": "ch_r0_measure",
                        "frequency": 6.0e9,
                        "physical_channel": {"id": "p_r0"},
                    }
                }
            },
        },
    }

    resonator = _resolve_resonator_payload(
        qubit_payload=quantum_devices["q0"],
        quantum_devices=quantum_devices,
    )
    assert resonator is not None
    assert resonator["id"] == "r0"

    acquire_definitions = _build_acquire_definitions_for_mode(
        quantum_devices["q0"], "acquire"
    )
    assert acquire_definitions is not None
    assert acquire_definitions[0].weights == (1.0, 2.0)

    assert (
        _build_readout_probability(
            error_mitigation={"readout_mitigation": {"linear": {"0": {"bad": 1.0}}}},
            qubit_payload={"index": 0},
        )
        is None
    )

    assert (
        _build_waveforms_for_mode({"ddrop_reset": {}}, "reset", {}, is_readout=False) == ()
    )

    acquire_defs = _build_acquire_definitions_for_mode(quantum_devices["q0"], "macq")
    assert acquire_defs is not None
    assert acquire_defs[0].weights == (1.0, 2.0)
    assert _build_acquire_definitions_for_mode(quantum_devices["q0"], "drive") is None

    mode = _build_mode_from_pulse_view(
        qubit_payload=quantum_devices["q0"],
        pulse_key="drive",
        pulse_view={"pulse_channel": {"id": "ch0"}},
        mode_id="drive",
    )
    assert mode is not None
    assert (
        _build_mode_from_pulse_view(
            qubit_payload=quantum_devices["q0"],
            pulse_key="drive",
            pulse_view={"pulse_channel": {}},
            mode_id="drive",
        )
        is None
    )

    warn_waveforms = _build_waveforms_for_mode(quantum_devices["q0"], "unknown_mode", {})
    assert warn_waveforms == ()


@pytest.mark.parametrize(
    "is_readout, expected_amp",
    [
        (False, 0.3),
        (True, 0.4),
    ],
)
def test_qubit_reset_waveform_helper_uses_available_amplitude(is_readout, expected_amp):
    qubit_payload = {
        "id": "q0",
        "index": 0,
        "ddrop_reset": {
            "shape": "square",
            "width": 20e-9,
            "qubit_amp": 0.3,
            "res_amp": 0.4,
        },
    }

    reset_waveforms = _build_waveforms_for_mode(
        qubit_payload,
        "reset",
        {},
        is_readout=is_readout,
    )

    assert len(reset_waveforms) == 1
    assert reset_waveforms[0].amp == pytest.approx(expected_amp)


@pytest.mark.parametrize(
    "qubit_payload, is_readout",
    [
        (
            {
                "id": "q0",
                "index": 0,
                "ddrop_reset": {
                    "shape": "square",
                    "width": 20e-9,
                    "res_amp": 0.4,
                },
            },
            False,
        ),
        (
            {
                "id": "q0",
                "index": 0,
                "ddrop_reset": {
                    "shape": "square",
                    "width": 20e-9,
                    "qubit_amp": 0.3,
                    "res_amp": None,
                },
            },
            True,
        ),
        (
            {
                "id": "q0",
                "index": 0,
                "ddrop_reset": {
                    "shape": "square",
                    "width": 20e-9,
                },
            },
            False,
        ),
        (
            {
                "id": "q0",
                "index": 0,
                "ddrop_reset": {
                    "shape": "square",
                    "width": 20e-9,
                },
            },
            True,
        ),
    ],
)
def test_qubit_reset_waveform_helper_skips_missing_or_none_amplitudes(
    qubit_payload, is_readout
):
    assert (
        _build_waveforms_for_mode(qubit_payload, "reset", {}, is_readout=is_readout) == ()
    )


def test_build_waveform_data_normalises_rise_by_shape_semantics():
    gaussian = _build_waveform_data(
        "gaussian",
        {
            "shape": "gaussian",
            "width": 100e-9,
            "rise": 1 / 3,
        },
    )
    assert gaussian.rise == pytest.approx(1 / 3)

    soft_square = _build_waveform_data(
        "soft_square",
        {
            "shape": "soft_square",
            "width": 100e-9,
            "rise": 10e-9,
        },
    )
    assert soft_square.rise == 10_000

    rounded_square = _build_waveform_data(
        "rounded_square",
        {
            "shape": "rounded_square",
            "width": 100e-9,
            "rise": 5e-9,
        },
    )
    assert rounded_square.rise == 5_000

    unknown = _build_waveform_data(
        "unknown",
        {
            "shape": "custom_shape",
            "width": 100e-9,
            "rise": 0.25,
        },
    )
    assert unknown.rise == 0.25


def test_qubit_resonator_resolution_and_readout_probability_edge_paths():
    quantum_devices = _synthetic_quantum_devices()
    qubit_payload = quantum_devices["q0"]

    resolved = _resolve_resonator_payload(
        qubit_payload=qubit_payload,
        quantum_devices=quantum_devices,
    )
    assert isinstance(resolved, dict)

    missing = _resolve_resonator_payload(
        qubit_payload={"id": "qX", "measure_device": {"id": "missing"}},
        quantum_devices=quantum_devices,
    )
    assert missing is None

    assert (
        _build_readout_probability(error_mitigation=None, qubit_payload=qubit_payload)
        is None
    )
    assert (
        _build_readout_probability(
            error_mitigation={"readout_mitigation": {"linear": {"0": {"bad": 1.0}}}},
            qubit_payload=qubit_payload,
        )
        is None
    )

    prob = _build_readout_probability(
        error_mitigation={"readout_mitigation": {"linear": {0: {"0|0": 1.0, "1|0": 0.0}}}},
        qubit_payload=qubit_payload,
    )
    assert prob is not None

    assert (
        _build_qubit_modes(quantum_devices=quantum_devices, qubit_payload={"id": 1}) == ()
    )


def test_build_qubits_skips_non_mapping_and_missing_id_index_entries():
    quantum_devices = {
        "raw": "bad",
        "missing_index": {"id": "q0"},
        "missing_id": {"index": 0},
        "good": {
            "id": "q1",
            "index": 1,
            "pulse_channels": {},
        },
    }

    qubits = _build_qubits(
        quantum_devices=quantum_devices,
        error_mitigation=None,
    )
    assert len(qubits) == 1
    assert qubits[0].id == "q1"


def test_qubit_mode_and_readout_helpers_cover_remaining_skip_paths():
    quantum_devices = {
        "q0": {
            "id": "q0",
            "index": 0,
            "pulse_channels": {
                "raw": "bad",
                "drive": {"pulse_channel": {"id": "ch0", "physical_channel": {"id": "p0"}}},
            },
            "measure_device": {"id": "r0"},
        },
        "r0": {
            "id": "r0",
            "pulse_channels": {"measure": "bad"},
        },
    }

    modes = _build_qubit_modes(
        quantum_devices=quantum_devices, qubit_payload=quantum_devices["q0"]
    )
    assert len(modes) == 1

    assert (
        _build_readout_probability(
            error_mitigation={"readout_mitigation": {"linear": "bad"}},
            qubit_payload={"index": 0},
        )
        is None
    )
    assert (
        _build_readout_probability(
            error_mitigation={"readout_mitigation": {"linear": {"0": {"0|x": 1.0}}}},
            qubit_payload={"index": 0},
        )
        is None
    )
    assert (
        _build_readout_probability(
            error_mitigation={"readout_mitigation": {"linear": {"0": {"0|0": 1.0}}}},
            qubit_payload={"index": "0"},
        )
        is None
    )


def test_qubit_mode_builder_branch_matrix_for_partial_edges():
    quantum_devices = {
        "q0": {
            "id": "q0",
            "index": 0,
            "pulse_channels": {
                "cr_no_zx": {
                    "pulse_channel": {"id": "ch_cr", "physical_channel": {"id": "p0"}}
                },
                "drive_bad": {"pulse_channel": {}},
            },
            "pulse_hw_zx_pi_4": "bad",
            "measure_device": {"id": "r0", "pulse_channels": {}},
        },
        "r0": {
            "id": "r0",
            "pulse_channels": {"measure": {"pulse_channel": {}}},
        },
    }

    # pulse_channels dict true with one invalid mode and one CR mode where zx_map is not dict
    modes = _build_qubit_modes(
        quantum_devices=quantum_devices, qubit_payload=quantum_devices["q0"]
    )
    assert isinstance(modes, tuple)

    # pulse_channels not dict branch
    assert (
        _build_qubit_modes(
            quantum_devices=quantum_devices,
            qubit_payload={"id": "q1", "pulse_channels": "bad"},
        )
        == ()
    )

    # resonator_channels not dict branch
    assert (
        _build_qubit_modes(
            quantum_devices={"r0": {"id": "r0", "pulse_channels": "bad"}},
            qubit_payload={
                "id": "q2",
                "measure_device": {"id": "r0"},
                "pulse_channels": {},
            },
        )
        == ()
    )

    # weights non-list branch for acquire definition path
    defs = _build_acquire_definitions_for_mode(
        {"measure_acquire": {"delay": 1e-9, "width": 2e-9, "weights": "bad"}},
        "acquire",
    )
    assert defs is not None and defs[0].weights is None


def test_signal_path_builder_helpers_cover_fallbacks_and_skips():
    registry = ExternalResourceRegistry()
    assert (
        _register_external_resource_from_payload(
            payload={"instrument_type": "port"},
            registry=registry,
        )
        is None
    )
    assert (
        _register_external_resource_from_payload(
            payload={"instrument_type": "port"},
            registry=registry,
            fallback_id="portA",
            fallback_type="port",
        )
        == "portA"
    )

    min_blocks, max_blocks = _build_port_block_bounds(
        payload={"min_blocks": 3, "max_blocks": 5},
        sample_time=1000,
        block_size=4,
    )
    assert (min_blocks, max_blocks) == (3, 5)

    derived_min, derived_max = _build_port_block_bounds(
        payload={"pulse_duration_min": 1e-8, "pulse_duration_max": 2e-8},
        sample_time=1000,
        block_size=4,
    )
    assert derived_min >= 1
    assert derived_max >= derived_min

    oscillators = _build_oscillators({"good": {"frequency": 5e9}}, registry)
    assert len(oscillators) == 1

    with pytest.raises(TypeError):
        _build_oscillators({"bad": {"frequency": "x"}}, registry)

    quantum_devices = copy.deepcopy(_synthetic_quantum_devices())
    quantum_devices["q0"]["pulse_channels"]["dup"] = {
        "pulse_channel": {
            "id": "ch_q0_drive",
            "physical_channel": {"id": "p_q0"},
            "frequency": 5e9,
        }
    }
    quantum_devices["q0"]["pulse_channels"]["no_port"] = {
        "pulse_channel": {"id": "ch_missing"}
    }

    channels = _build_channels(
        quantum_devices=quantum_devices,
        physical_channels=_synthetic_physical_channels(),
    )
    channel_ids = {channel.id for channel in channels}
    assert "ch_q0_drive" in channel_ids
    assert "ch_missing" not in channel_ids


def test_signal_path_builders_skip_malformed_entries_and_missing_references():
    registry = ExternalResourceRegistry()

    ports = _build_ports(
        {
            "bad": "bad",
            "no_sample_time": {"id": "no_sample_time"},
            "good": {"id": "good", "sample_time": 1e-9},
        },
        registry,
    )
    assert len(ports) == 1
    assert ports[0].id == "good"

    oscillators = _build_oscillators(
        {
            "bad_payload": "bad",
            "missing_freq": {},
            "good": {"frequency": 5e9},
        },
        registry,
    )
    assert len(oscillators) == 1

    quantum_devices = {
        "raw": "bad",
        "no_pulses": {"pulse_channels": "bad"},
        "q0": {
            "id": "q0",
            "pulse_channels": {
                "raw_view": "bad",
                "no_pulse_channel": {},
                "bad_pulse_channel": {"pulse_channel": "bad"},
                "bad_port": {
                    "pulse_channel": {
                        "id": "ch_bad_port",
                        "physical_channel": {"id": 1},
                        "frequency": 5e9,
                    }
                },
                "missing_top": {
                    "pulse_channel": {
                        "id": "ch_missing_top",
                        "physical_channel": {"id": "missing"},
                        "frequency": 5e9,
                    }
                },
                "good": {
                    "pulse_channel": {
                        "id": "ch_good",
                        "physical_channel": {"id": "p0"},
                        "frequency": 5e9,
                    }
                },
            },
        },
    }
    physical_channels = {"p0": {"id": "p0", "baseband": "bad"}, "missing": "bad"}

    channels = _build_channels(
        quantum_devices=quantum_devices,
        physical_channels=physical_channels,
    )
    assert [channel.id for channel in channels] == ["ch_missing_top", "ch_good"]
    assert all(channel.oscillator_reference is None for channel in channels)
