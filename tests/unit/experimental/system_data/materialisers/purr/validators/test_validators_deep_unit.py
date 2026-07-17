# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import copy
import math
from types import SimpleNamespace

import pytest

from qat.experimental.system_data.materialisers.errors import (
    SourceConsistencyError,
    SourceValidationError,
)
from qat.experimental.system_data.materialisers.purr.ingress.v0_1_0 import PurrIngressV010
from qat.experimental.system_data.materialisers.purr.validators.couplings import (
    _iter_normalized_cr_crc_entries,
    _validate_coupling_direction_entry,
    _validate_cr_crc_auxiliary_targets,
    _validate_cr_crc_channel_mapping_keys,
    _validate_cr_crc_counterparts,
    _validate_cr_crc_matches_coupling_graph,
    _warn_missing_coupling_quality,
)
from qat.experimental.system_data.materialisers.purr.validators.postprocess import (
    _validate_linear_post_process_method_payload,
    _validate_max_likelihood_post_process_method_payload,
    _validate_max_likelihood_states,
    _validate_mean_z_map_args,
    _validate_post_process_method,
    _validate_readout_mitigation,
    _validate_readout_mitigation_qubit_map,
    _warn_extra_mitigation_entries,
)
from qat.experimental.system_data.materialisers.purr.validators.signal_paths import (
    _validate_baseband_payload,
    _validate_passive_reset_time,
    _validate_physical_channel_payload,
    _validate_pulse_channel_frequencies,
    _validate_pulse_channel_reference,
    _validate_pulse_channel_scales,
    _warn_sample_time_consistency,
)
from qat.experimental.system_data.materialisers.purr.validators.waveforms import (
    _validate_measure_acquire_payloads,
    _validate_waveform_numeric_fields,
    _validate_waveform_payloads,
)


def _make_dto() -> PurrIngressV010:
    payload = {
        "calibration_id": "cal",
        "supported_acquire_modes": ["integrator"],
        "default_acquire_mode": "integrator",
        "repeat_limit": 10,
        "quantum_devices": {
            "Q0": {
                "id": "Q0",
                "index": 0,
                "pulse_channels": {
                    "Q1.cross_resonance": {
                        "pulse_channel": {
                            "id": "Q0.Q1.cross_resonance",
                            "physical_channel": {"id": "p_q0"},
                            "frequency": 5e9,
                            "scale": 1.0,
                            "auxiliary_qubit": "Q1",
                        }
                    },
                    "Q1.cross_resonance_cancellation": {
                        "pulse_channel": {
                            "id": "Q0.Q1.cross_resonance_cancellation",
                            "physical_channel": {"id": "p_q0"},
                            "frequency": 5e9,
                        }
                    },
                    "drive": {
                        "pulse_channel": {
                            "id": "Q0.drive",
                            "physical_channel": {"id": "p_q0"},
                            "frequency": 5e9,
                            "scale": 1.0,
                        }
                    },
                },
                "pulse_hw_x_pi_2": {"width": 20e-9, "rise": 5e-9, "amp": 0.2},
                "pulse_hw_x_pi": {"width": 40e-9, "rise": 10e-9, "amp": 0.4},
                "pulse_measure": {"width": 100e-9, "amp": 0.8},
                "pulse_hw_zx_pi_4": {
                    "Q1": {"width": 20e-9, "amp": 0.2, "phase": 0.0, "drag": 0.0}
                },
                "measure_acquire": {
                    "delay": 10e-9,
                    "width": 40e-9,
                    "weights": [1.0, 0.0],
                    "sync": True,
                },
                "mean_z_map_args": [1 + 0j, 0 + 0j],
                "post_process_method": {
                    "method": "max_likelihood",
                    "states": {
                        "0": {"location": 1 + 0j, "label": "g"},
                        "1": {"location": -1 + 0j, "label": "e"},
                    },
                    "noise_est": 1.0,
                    "p_min": 0.0,
                    "transform": [[1.0, 0.0], [0.0, 1.0]],
                    "offset": [0.0, 0.0],
                },
            },
            "Q1": {
                "id": "Q1",
                "index": 1,
                "pulse_channels": {
                    "drive": {
                        "pulse_channel": {
                            "id": "Q1.drive",
                            "physical_channel": {"id": "p_q1"},
                            "frequency": 5.1e9,
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
                            "frequency": 6.0e9,
                        }
                    }
                },
            },
        },
        "pulse_channels": {
            "ch_global": {"physical_channel": "p_q0"},
        },
        "physical_channels": {
            "p_q0": {
                "id": "p_q0",
                "sample_time": 1e-9,
                "block_size": 4,
                "acquire_allowed": False,
                "pulse_channel_min_frequency": 4.9e9,
                "pulse_channel_max_frequency": 5.2e9,
            },
            "p_q1": {
                "id": "p_q1",
                "sample_time": 2e-9,
                "block_size": 4,
                "acquire_allowed": False,
            },
            "p_r0": {
                "id": "p_r0",
                "sample_time": 3e-9,
                "block_size": 4,
                "acquire_allowed": True,
            },
        },
        "basebands": {"bb0": {"frequency": 5e9}},
        "qubit_direction_couplings": [{"direction": [0, 1], "quality": 0.9}],
        "error_mitigation": {
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
        },
    }
    return PurrIngressV010.model_validate(payload)


def _mutated_dto(mutator):
    payload = _make_dto().model_dump(mode="python")
    mutator(payload)
    return PurrIngressV010.model_validate(payload)


@pytest.mark.parametrize(
    "entry, exc_type, msg",
    [
        ("bad", SourceValidationError, "must be a mapping"),
        ({"direction": [0]}, SourceValidationError, "two-element integer list"),
        ({"direction": [0, 3]}, SourceConsistencyError, "unknown qubit indices"),
    ],
)
def test_validate_coupling_direction_entry_errors(entry, exc_type, msg):
    with pytest.raises(exc_type, match=msg):
        _validate_coupling_direction_entry(
            index=0,
            entry=entry,
            qubit_indices={0, 1},
        )


@pytest.mark.parametrize(
    "mutator, msg",
    [
        (
            lambda p: p["quantum_devices"]["Q0"]["pulse_channels"].__setitem__(
                "bad.cross_resonance",
                p["quantum_devices"]["Q0"]["pulse_channels"].pop("Q1.cross_resonance"),
            ),
            "must match '<target>.cross_resonance",
        ),
        (
            lambda p: p["quantum_devices"]["Q0"]["pulse_channels"][
                "Q1.cross_resonance"
            ].__setitem__("pulse_channel", "bad"),
            "must contain a pulse_channel mapping",
        ),
        (
            lambda p: p["quantum_devices"]["Q0"]["pulse_channels"]["Q1.cross_resonance"][
                "pulse_channel"
            ].__setitem__("id", 123),
            "must be a string",
        ),
        (
            lambda p: p["quantum_devices"]["Q0"]["pulse_channels"]["Q1.cross_resonance"][
                "pulse_channel"
            ].__setitem__("id", "invalid"),
            "must match '<source>.<target>.cross_resonance",
        ),
    ],
)
def test_validate_cr_crc_channel_mapping_keys_validation_errors(mutator, msg):
    dto = _mutated_dto(mutator)
    with pytest.raises(SourceValidationError, match=msg):
        _validate_cr_crc_channel_mapping_keys(dto)


@pytest.mark.parametrize(
    "channel_id, msg",
    [
        ("Q2.Q1.cross_resonance", "source in id does not match"),
        ("Q0.Q2.cross_resonance", "does not match pulse_channel.id target"),
        ("Q0.Q1.cross_resonance_cancellation", "does not match pulse_channel.id suffix"),
    ],
)
def test_validate_cr_crc_channel_mapping_keys_consistency_errors(channel_id, msg):
    dto = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"]["pulse_channels"]["Q1.cross_resonance"][
            "pulse_channel"
        ].__setitem__("id", channel_id)
    )
    with pytest.raises(SourceConsistencyError, match=msg):
        _validate_cr_crc_channel_mapping_keys(dto)


def test_validate_cr_crc_counterparts_and_graph_and_auxiliary_target_errors():
    dto_missing_counterpart = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"]["pulse_channels"].pop(
            "Q1.cross_resonance_cancellation"
        )
    )
    with pytest.raises(SourceConsistencyError, match="missing counterpart"):
        _validate_cr_crc_counterparts(dto_missing_counterpart)

    dto_missing_graph_mapping = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"]["pulse_channels"].pop("Q1.cross_resonance")
    )
    with pytest.raises(
        SourceConsistencyError, match="missing CR/CRC pulse-channel mappings"
    ):
        _validate_cr_crc_matches_coupling_graph(dto_missing_graph_mapping)

    dto_bad_aux = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"]["pulse_channels"]["Q1.cross_resonance"][
            "pulse_channel"
        ].__setitem__("auxiliary_qubit", "Q2")
    )
    with pytest.raises(SourceConsistencyError, match="auxiliary target does not match"):
        _validate_cr_crc_auxiliary_targets(dto_bad_aux)


def test_warn_missing_coupling_quality_and_extra_mitigation_entries(caplog):
    _warn_missing_coupling_quality(
        [
            {"direction": [0, 1]},
            {"direction": [1, 2], "quality": 0.99},
        ]
    )
    assert any("missing quality fidelity" in rec.message for rec in caplog.records)

    dto = _make_dto()
    mitigation = copy.deepcopy(dto.error_mitigation)
    mitigation["readout_mitigation"]["linear"]["99"] = {
        "0|0": 1.0,
        "1|0": 0.0,
        "0|1": 0.0,
        "1|1": 1.0,
    }
    _warn_extra_mitigation_entries(mitigation, {0, 1})
    assert any(
        "entries exist for qubits not in device set" in rec.message
        for rec in caplog.records
    )


def test_validate_readout_mitigation_qubit_map_error_branches():
    with pytest.raises(SourceValidationError, match="must be a mapping"):
        _validate_readout_mitigation_qubit_map(
            qubit_id="0",
            qubit_map="bad",
            qubit_indices={0},
        )

    with pytest.raises(SourceValidationError, match="coercible to an integer"):
        _validate_readout_mitigation_qubit_map(
            qubit_id="bad",
            qubit_map={},
            qubit_indices={0},
        )

    with pytest.raises(SourceValidationError, match="must have 'measured|prepared' form"):
        _validate_readout_mitigation_qubit_map(
            qubit_id="0",
            qubit_map={"bad": 1.0},
            qubit_indices={0},
        )

    with pytest.raises(SourceValidationError, match="must be numeric"):
        _validate_readout_mitigation_qubit_map(
            qubit_id="0",
            qubit_map={"0|0": "x"},
            qubit_indices={0},
        )

    with pytest.raises(SourceValidationError, match=r"must lie in \[0, 1\]"):
        _validate_readout_mitigation_qubit_map(
            qubit_id="0",
            qubit_map={"0|0": 2.0},
            qubit_indices={0},
        )

    with pytest.raises(SourceValidationError, match="must sum to 1"):
        _validate_readout_mitigation_qubit_map(
            qubit_id="0",
            qubit_map={"0|0": 0.5, "1|0": 0.1, "0|1": 0.9, "1|1": 0.9},
            qubit_indices={0},
        )

    with pytest.raises(SourceValidationError, match="must be a mapping when provided"):
        _validate_readout_mitigation({"readout_mitigation": {"linear": "bad"}}, {0})

    # Out-of-scope qubit entries are ignored by design.
    _validate_readout_mitigation_qubit_map(
        qubit_id="9",
        qubit_map={"invalid": "shape"},
        qubit_indices={0},
    )


def test_postprocess_helper_validation_error_branches():
    with pytest.raises(SourceValidationError, match="exactly two entries"):
        _validate_linear_post_process_method_payload(
            method_payload={"mean_z_map_args": [1]},
            path_root="$.x",
        )

    with pytest.raises(SourceValidationError, match="int, float, or complex"):
        _validate_linear_post_process_method_payload(
            method_payload={"mean_z_map_args": [1, "x"]},
            path_root="$.x",
        )

    with pytest.raises(SourceValidationError, match="non-empty states mapping"):
        _validate_max_likelihood_states(states={}, path_root="$.x")

    with pytest.raises(SourceValidationError, match="coercible to integers"):
        _validate_max_likelihood_states(states={"bad": {"location": 1}}, path_root="$.x")

    with pytest.raises(SourceValidationError, match="must be a mapping"):
        _validate_max_likelihood_states(states={"0": "bad"}, path_root="$.x")

    with pytest.raises(
        SourceValidationError, match="location must be int, float, or complex"
    ):
        _validate_max_likelihood_states(
            states={"0": {"location": "x"}},
            path_root="$.x",
        )

    with pytest.raises(SourceValidationError, match="label must be a string"):
        _validate_max_likelihood_states(
            states={"0": {"location": 1, "label": 1}},
            path_root="$.x",
        )

    with pytest.raises(SourceValidationError, match="noise_est must be numeric"):
        _validate_max_likelihood_post_process_method_payload(
            method_payload={"states": {"0": {"location": 1}}, "noise_est": "bad"},
            path_root="$.x",
        )

    with pytest.raises(SourceValidationError, match="p_min must be numeric"):
        _validate_max_likelihood_post_process_method_payload(
            method_payload={"states": {"0": {"location": 1}}, "p_min": "bad"},
            path_root="$.x",
        )

    with pytest.raises(SourceValidationError, match="transform must be a 2x2"):
        _validate_max_likelihood_post_process_method_payload(
            method_payload={"states": {"0": {"location": 1}}, "transform": [1]},
            path_root="$.x",
        )

    with pytest.raises(SourceValidationError, match="offset must be a 2-element"):
        _validate_max_likelihood_post_process_method_payload(
            method_payload={"states": {"0": {"location": 1}}, "offset": [1]},
            path_root="$.x",
        )


def test_validate_mean_z_map_and_post_process_method_error_branches():
    dto_mean = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"].__setitem__("mean_z_map_args", [1])
    )
    with pytest.raises(SourceValidationError, match="exactly two entries"):
        _validate_mean_z_map_args(dto_mean)

    dto_mean_bad_type = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"].__setitem__("mean_z_map_args", [1, "x"])
    )
    with pytest.raises(SourceValidationError, match="int, float, or complex"):
        _validate_mean_z_map_args(dto_mean_bad_type)

    dto_method_not_mapping = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"].__setitem__("post_process_method", "bad")
    )
    with pytest.raises(SourceValidationError, match="must be a mapping"):
        _validate_post_process_method(dto_method_not_mapping)

    dto_linear_bad = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"].__setitem__(
            "post_process_method",
            {"method": "linear_map_complex_to_real", "mean_z_map_args": [1]},
        )
    )
    with pytest.raises(SourceValidationError, match="exactly two entries"):
        _validate_post_process_method(dto_linear_bad)

    dto_linear_ok = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"].__setitem__(
            "post_process_method",
            {
                "method": "linear_map_complex_to_real",
                "mean_z_map_args": [1 + 0j, 0 + 0j],
            },
        )
    )
    _validate_post_process_method(dto_linear_ok)

    _warn_extra_mitigation_entries({"readout_mitigation": {"linear": "bad"}}, {0, 1})
    _warn_extra_mitigation_entries(
        {
            "readout_mitigation": {
                "linear": {
                    "bad": {"0|0": 1.0},
                }
            }
        },
        {0, 1},
    )


def test_signal_path_helper_validation_error_branches_and_warnings(caplog):
    with pytest.raises(SourceValidationError, match="non-negative"):
        _validate_passive_reset_time(-1.0)

    with pytest.raises(SourceValidationError, match="strictly positive"):
        _validate_physical_channel_payload("p", {"sample_time": 0})

    with pytest.raises(SourceValidationError, match="integer >= 1"):
        _validate_physical_channel_payload("p", {"sample_time": 1e-9, "block_size": 0})

    with pytest.raises(SourceValidationError, match="min_blocks must be an integer"):
        _validate_physical_channel_payload("p", {"sample_time": 1e-9, "min_blocks": 0})

    with pytest.raises(SourceValidationError, match="max_blocks must be an integer"):
        _validate_physical_channel_payload("p", {"sample_time": 1e-9, "max_blocks": 0})

    with pytest.raises(SourceValidationError, match="min_blocks must be <= max_blocks"):
        _validate_physical_channel_payload(
            "p",
            {"sample_time": 1e-9, "min_blocks": 3, "max_blocks": 2},
        )

    with pytest.raises(SourceValidationError, match="must be a boolean"):
        _validate_physical_channel_payload(
            "p",
            {"sample_time": 1e-9, "acquire_allowed": "yes"},
        )

    with pytest.raises(
        SourceValidationError, match="pulse_duration_min must be a finite strictly positive"
    ):
        _validate_physical_channel_payload(
            "p",
            {"sample_time": 1e-9, "pulse_duration_min": 0},
        )

    with pytest.raises(
        SourceValidationError, match="pulse_duration_max must be a finite strictly positive"
    ):
        _validate_physical_channel_payload(
            "p",
            {"sample_time": 1e-9, "pulse_duration_max": 0},
        )

    with pytest.raises(
        SourceValidationError, match="pulse_duration_min must be <= pulse_duration_max"
    ):
        _validate_physical_channel_payload(
            "p",
            {"sample_time": 1e-9, "pulse_duration_min": 2e-9, "pulse_duration_max": 1e-9},
        )

    with pytest.raises(SourceValidationError, match="strictly positive real number"):
        _validate_baseband_payload("bb", {"frequency": 0.0})

    with pytest.raises(SourceValidationError, match="finite strictly positive"):
        _validate_physical_channel_payload("p", {"sample_time": math.inf})

    with pytest.raises(SourceValidationError, match="finite strictly positive"):
        _validate_physical_channel_payload("p", {"sample_time": math.nan})

    with pytest.raises(SourceValidationError, match="finite strictly positive"):
        _validate_baseband_payload("bb", {"frequency": math.inf})

    with pytest.raises(SourceValidationError, match="finite strictly positive"):
        _validate_baseband_payload("bb", {"frequency": math.nan})

    dto_freq_bad = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"]["pulse_channels"]["drive"][
            "pulse_channel"
        ].__setitem__("frequency", -1)
    )
    with pytest.raises(SourceValidationError, match="must be a finite non-negative number"):
        _validate_pulse_channel_frequencies(dto_freq_bad)

    dto_freq_inf = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"]["pulse_channels"]["drive"][
            "pulse_channel"
        ].__setitem__("frequency", math.inf)
    )
    with pytest.raises(SourceValidationError, match="must be a finite non-negative number"):
        _validate_pulse_channel_frequencies(dto_freq_inf)

    dto_freq_nan = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"]["pulse_channels"]["drive"][
            "pulse_channel"
        ].__setitem__("frequency", math.nan)
    )
    with pytest.raises(SourceValidationError, match="must be a finite non-negative number"):
        _validate_pulse_channel_frequencies(dto_freq_nan)

    dto_freq_low = _mutated_dto(
        lambda p: (
            p["quantum_devices"]["Q0"]["pulse_channels"]["drive"][
                "pulse_channel"
            ].__setitem__("frequency", 1.0),
            p["quantum_devices"]["Q0"]["pulse_channels"]["drive"]["pulse_channel"][
                "physical_channel"
            ].update(
                {
                    "pulse_channel_min_frequency": 4.9e9,
                    "pulse_channel_max_frequency": 5.2e9,
                }
            ),
        )
    )
    with pytest.raises(SourceValidationError, match="below the declared minimum"):
        _validate_pulse_channel_frequencies(dto_freq_low)

    dto_freq_high = _mutated_dto(
        lambda p: (
            p["quantum_devices"]["Q0"]["pulse_channels"]["drive"][
                "pulse_channel"
            ].__setitem__("frequency", 9.9e9),
            p["quantum_devices"]["Q0"]["pulse_channels"]["drive"]["pulse_channel"][
                "physical_channel"
            ].update(
                {
                    "pulse_channel_min_frequency": 4.9e9,
                    "pulse_channel_max_frequency": 5.2e9,
                }
            ),
        )
    )
    with pytest.raises(SourceValidationError, match="above the declared maximum"):
        _validate_pulse_channel_frequencies(dto_freq_high)

    dto_scale_bad = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"]["pulse_channels"]["drive"][
            "pulse_channel"
        ].__setitem__("scale", "bad")
    )
    with pytest.raises(SourceValidationError, match="must be numeric or complex"):
        _validate_pulse_channel_scales(dto_scale_bad)

    dto_warn = _make_dto()
    _warn_sample_time_consistency(dto_warn)
    assert any("inconsistent sample_time values" in rec.message for rec in caplog.records)

    dto_resonator_warn = _mutated_dto(
        lambda p: (
            p["quantum_devices"].__setitem__(
                "R1",
                {
                    "id": "R1",
                    "pulse_channels": {
                        "measure": {
                            "pulse_channel": {
                                "id": "R1.measure",
                                "physical_channel": {"id": "p_r1"},
                                "frequency": 6.1e9,
                            }
                        }
                    },
                },
            )
            or p["physical_channels"].__setitem__(
                "p_r1",
                {
                    "id": "p_r1",
                    "sample_time": 5e-9,
                    "block_size": 4,
                    "acquire_allowed": True,
                },
            )
        )
    )
    _warn_sample_time_consistency(dto_resonator_warn)
    assert any(
        "Resonator channels have inconsistent sample_time values" in rec.message
        for rec in caplog.records
    )


def test_signal_path_frequency_and_sample_time_skip_branches_do_not_raise(caplog):
    _validate_pulse_channel_reference(
        pulse_id="p0",
        pulse_payload={"physical_channel": {"id": "p_q0"}},
        physical_channel_ids=frozenset({"p_q0"}),
    )

    dto_missing_physical_dict = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"]["pulse_channels"]["drive"][
            "pulse_channel"
        ].__setitem__("physical_channel", "bad")
    )
    _validate_pulse_channel_frequencies(dto_missing_physical_dict)

    dto_non_numeric_frequency = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"]["pulse_channels"]["drive"][
            "pulse_channel"
        ].__setitem__("frequency", None)
    )
    _validate_pulse_channel_frequencies(dto_non_numeric_frequency)

    dto_no_resonator_warning = _mutated_dto(
        lambda p: p["quantum_devices"]["R0"]["pulse_channels"]["measure"][
            "pulse_channel"
        ].__setitem__("physical_channel", {"id": "missing"})
    )
    _warn_sample_time_consistency(dto_no_resonator_warning)
    assert not any(
        "Resonator channels have inconsistent sample_time values" in rec.message
        for rec in caplog.records
    )


def test_signal_path_warning_helper_covers_non_dict_skip_branches():
    fake_dto = SimpleNamespace(
        quantum_devices={
            "raw": "bad",
            "q0": {"pulse_channels": "bad"},
            "q1": {"pulse_channels": {"raw": "bad", "v": {"pulse_channel": "bad"}}},
            "q2": {"pulse_channels": {"v": {"pulse_channel": {"physical_channel": 7}}}},
            "q3": {
                "pulse_channels": {
                    "v": {"pulse_channel": {"physical_channel": {"id": "p0"}}}
                }
            },
        },
        physical_channels={"p0": "bad"},
    )

    _warn_sample_time_consistency(fake_dto)


def test_waveform_validator_error_branches():
    dto_width_bad = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"]["pulse_hw_x_pi_2"].__setitem__("width", -1)
    )
    with pytest.raises(SourceValidationError, match="Waveform width"):
        _validate_waveform_payloads(dto_width_bad)

    dto_width_inf = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"]["pulse_hw_x_pi_2"].__setitem__(
            "width", math.inf
        )
    )
    with pytest.raises(SourceValidationError, match="Waveform width"):
        _validate_waveform_payloads(dto_width_inf)

    dto_width_nan = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"]["pulse_hw_x_pi_2"].__setitem__(
            "width", math.nan
        )
    )
    with pytest.raises(SourceValidationError, match="Waveform width"):
        _validate_waveform_payloads(dto_width_nan)

    dto_rise_bad = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"]["pulse_hw_x_pi_2"].__setitem__("rise", -1)
    )
    with pytest.raises(SourceValidationError, match="Waveform rise"):
        _validate_waveform_payloads(dto_rise_bad)

    dto_rise_inf = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"]["pulse_hw_x_pi_2"].__setitem__(
            "rise", math.inf
        )
    )
    with pytest.raises(SourceValidationError, match="Waveform rise"):
        _validate_waveform_payloads(dto_rise_inf)

    dto_rise_nan = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"]["pulse_hw_x_pi_2"].__setitem__(
            "rise", math.nan
        )
    )
    with pytest.raises(SourceValidationError, match="Waveform rise"):
        _validate_waveform_payloads(dto_rise_nan)

    dto_zx_bad = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"]["pulse_hw_zx_pi_4"]["Q1"].__setitem__(
            "width", -1
        )
    )
    with pytest.raises(SourceValidationError, match="Cross-resonance waveform width"):
        _validate_waveform_payloads(dto_zx_bad)

    dto_zx_width_inf = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"]["pulse_hw_zx_pi_4"]["Q1"].__setitem__(
            "width", math.inf
        )
    )
    with pytest.raises(SourceValidationError, match="Cross-resonance waveform width"):
        _validate_waveform_payloads(dto_zx_width_inf)

    dto_zx_width_nan = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"]["pulse_hw_zx_pi_4"]["Q1"].__setitem__(
            "width", math.nan
        )
    )
    with pytest.raises(SourceValidationError, match="Cross-resonance waveform width"):
        _validate_waveform_payloads(dto_zx_width_nan)

    dto_acq_delay_bad = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"]["measure_acquire"].__setitem__("delay", -1)
    )
    with pytest.raises(SourceValidationError, match="Acquire delay"):
        _validate_measure_acquire_payloads(dto_acq_delay_bad)

    dto_acq_delay_inf = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"]["measure_acquire"].__setitem__(
            "delay", math.inf
        )
    )
    with pytest.raises(SourceValidationError, match="Acquire delay"):
        _validate_measure_acquire_payloads(dto_acq_delay_inf)

    dto_acq_width_nan = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"]["measure_acquire"].__setitem__(
            "width", math.nan
        )
    )
    with pytest.raises(SourceValidationError, match="Acquire width"):
        _validate_measure_acquire_payloads(dto_acq_width_nan)

    dto_acq_weights_type_bad = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"]["measure_acquire"].__setitem__(
            "weights", "bad"
        )
    )
    with pytest.raises(SourceValidationError, match="must be a list"):
        _validate_measure_acquire_payloads(dto_acq_weights_type_bad)

    dto_acq_weights_value_bad = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"]["measure_acquire"].__setitem__(
            "weights", [1.0, "x"]
        )
    )
    with pytest.raises(SourceValidationError, match="must be numeric or complex"):
        _validate_measure_acquire_payloads(dto_acq_weights_value_bad)

    dto_amp_bad = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"]["pulse_measure"].__setitem__("amp", math.inf)
    )
    with pytest.raises(SourceValidationError, match="must be a finite number"):
        _validate_waveform_numeric_fields(dto_amp_bad)

    dto_zx_phase_bad = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"]["pulse_hw_zx_pi_4"]["Q1"].__setitem__(
            "phase", math.nan
        )
    )
    with pytest.raises(SourceValidationError, match="Cross-resonance waveform phase"):
        _validate_waveform_numeric_fields(dto_zx_phase_bad)

    dto_zx_non_dict = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"]["pulse_hw_zx_pi_4"].__setitem__("Qx", "bad")
    )
    _validate_waveform_payloads(dto_zx_non_dict)
    _validate_waveform_numeric_fields(dto_zx_non_dict)


def test_signal_path_validator_finite_value_edge_branches():
    dto_baseband_bad = _mutated_dto(
        lambda p: p["basebands"]["bb0"].__setitem__("frequency", math.inf)
    )
    with pytest.raises(SourceValidationError, match="Baseband frequency"):
        _validate_baseband_payload("bb0", dto_baseband_bad.basebands["bb0"])

    dto_channel_bad = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"]["pulse_channels"]["drive"][
            "pulse_channel"
        ].__setitem__("frequency", math.inf)
    )
    with pytest.raises(SourceValidationError, match="Pulse channel frequency"):
        _validate_pulse_channel_frequencies(dto_channel_bad)

    dto_physical_bad = _mutated_dto(
        lambda p: p["physical_channels"]["p_q0"].__setitem__("sample_time", math.nan)
    )
    with pytest.raises(SourceValidationError, match="sample_time"):
        _validate_physical_channel_payload(
            "p_q0", dto_physical_bad.physical_channels["p_q0"]
        )


def test_validator_skip_and_error_branches_for_couplings_signal_paths_and_weights():
    dto_bad_key = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"]["pulse_channels"].__setitem__(
            "invalid_key",
            p["quantum_devices"]["Q0"]["pulse_channels"].pop("Q1.cross_resonance"),
        )
    )
    # Invalid CR/CRC key format is skipped by normalized-entry iterator.
    entries = list(_iter_normalized_cr_crc_entries(dto_bad_key))
    assert all(
        entry["suffix"] in {"cross_resonance", "cross_resonance_cancellation"}
        for entry in entries
    )

    dto_bad_channel = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"]["pulse_channels"][
            "Q1.cross_resonance"
        ].__setitem__("pulse_channel", "bad")
    )
    assert list(_iter_normalized_cr_crc_entries(dto_bad_channel))

    dto_scale_skip = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"]["pulse_channels"]["drive"].__setitem__(
            "pulse_channel", "bad"
        )
    )
    _validate_pulse_channel_scales(dto_scale_skip)

    dto_warn_skip = _mutated_dto(
        lambda p: p["physical_channels"]["p_q0"].__setitem__("sample_time", math.nan)
    )
    _warn_sample_time_consistency(dto_warn_skip)

    dto_freq_below_min = _mutated_dto(
        lambda p: (
            p["quantum_devices"]["Q0"]["pulse_channels"]["drive"][
                "pulse_channel"
            ].__setitem__("frequency", 4.0e9),
            p["quantum_devices"]["Q0"]["pulse_channels"]["drive"]["pulse_channel"].update(
                {
                    "physical_channel": {
                        "id": "p_q0",
                        "pulse_channel_min_frequency": 4.9e9,
                    }
                }
            ),
        )
    )
    with pytest.raises(SourceValidationError, match="below the declared minimum"):
        _validate_pulse_channel_frequencies(dto_freq_below_min)

    dto_weights_bad_custom_pulse = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"]["measure_acquire"].__setitem__(
            "weights", {"object_type": "not.Pulse", "samples": [1.0]}
        )
    )
    with pytest.raises(SourceValidationError, match="must be a CustomPulse with samples"):
        _validate_measure_acquire_payloads(dto_weights_bad_custom_pulse)
