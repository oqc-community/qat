# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from qat.experimental.system_data.materialisers.purr.materialisers.common import (
    _seconds_to_picoseconds,
)
from qat.experimental.system_data.materialisers.purr.materialisers.qubits import (
    _build_qubit_modes,
    _build_waveforms_for_mode,
    _get_control_qubit_ids,
    _get_coupled_qubit_ids,
)


def test_seconds_to_picoseconds_logs_warning_on_lossy_rounding(caplog):
    with caplog.at_level("WARNING", logger="qat.purr.utils.logger"):
        result = _seconds_to_picoseconds(1.2345e-12)

    assert result == 1
    assert "Rounded duration from scaled picoseconds value" in caplog.text


def test_seconds_to_picoseconds_skips_warning_for_exact_scaling(caplog):
    with caplog.at_level("WARNING", logger="qat.purr.utils.logger"):
        result = _seconds_to_picoseconds(2e-12)

    assert result == 2
    assert "Rounded duration from scaled picoseconds value" not in caplog.text


def test_get_coupled_qubit_ids_returns_empty_tuple_when_pulse_channels_absent():
    assert _get_coupled_qubit_ids({}) == ()
    assert _get_coupled_qubit_ids({"pulse_channels": "bad"}) == ()


def test_get_control_qubit_ids_returns_empty_tuple_when_pulse_channels_absent():
    assert _get_control_qubit_ids({}) == ()
    assert _get_control_qubit_ids({"pulse_channels": "bad"}) == ()


def test_get_coupled_qubit_ids_deduplicates_same_peer():
    """Two .cross_resonance keys with the same first segment are counted once."""
    result = _get_coupled_qubit_ids(
        {
            "pulse_channels": {
                "Q1.cross_resonance": {},
                "Q1.extra.cross_resonance": {},
            }
        }
    )
    assert result == ("Q1",)


def test_get_control_qubit_ids_deduplicates_same_peer():
    """Two .cross_resonance_cancellation keys with the same first segment are counted
    once."""
    result = _get_control_qubit_ids(
        {
            "pulse_channels": {
                "Q0.cross_resonance_cancellation": {},
                "Q0.extra.cross_resonance_cancellation": {},
            }
        }
    )
    assert result == ("Q0",)


def test_build_waveforms_for_mode_cr_without_zx_map():
    """No ZX waveforms when pulse_hw_zx_pi_4 is absent."""
    result = _build_waveforms_for_mode(
        {}, "Q1.cross_resonance", {"id": "Q0.Q1.cross_resonance"}
    )
    assert result == ()


def test_build_waveforms_for_mode_cr_with_missing_zx_target():
    """No ZX waveforms when pulse_hw_zx_pi_4 doesn't contain the target qubit."""
    result = _build_waveforms_for_mode(
        {"pulse_hw_zx_pi_4": {"Q2": {"width": 20e-9, "rise": 5e-9, "amp": 0.2}}},
        "Q1.cross_resonance",
        {"id": "Q0.Q1.cross_resonance"},
    )
    assert result == ()


def test_build_waveforms_for_mode_reset_without_ddrop_config():
    """No reset waveforms when ddrop_reset config is absent."""
    result = _build_waveforms_for_mode({}, "reset", {"id": "Q0.reset"})
    assert result == ()


def test_build_qubit_modes_skips_none_mode_from_resonator_channel():
    """Resonator channels with invalid pulse_channel dict produce no mode."""
    modes = _build_qubit_modes(
        quantum_devices={},
        qubit_payload={
            "id": "Q0",
            "pulse_channels": {},
            "measure_device": {
                "id": "R0",
                "pulse_channels": {
                    "measure": {"pulse_channel": "not_a_dict"},
                },
            },
        },
    )
    assert not any(m.id == "readout_measure" for m in modes)
