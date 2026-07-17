# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Qubit-, mode-, waveform-, and readout-probability builders for PuRR materialisation."""

from typing import Any

from qat.experimental.system_data.canonical.schema import (
    AcquireDefinitionData,
    ModeData,
    ProbabilityEntry,
    QubitData,
    ReadoutProbabilityData,
    WaveformData,
)
from qat.experimental.system_data.materialisers.purr.materialisers.common import (
    _as_float,
    _seconds_to_picoseconds,
)
from qat.experimental.system_data.materialisers.purr.materialisers.postprocess import (
    _build_post_process_method,
)
from qat.experimental.utils.logging import get_logger

logger = get_logger(__name__)

_RISE_RATIO_SHAPES = {"gaussian", "softer_gaussian"}
_RISE_TIME_SHAPES = {
    "soft_square",
    "softer_square",
    "extra_soft_square",
    "rounded_square",
    "setup_hold",
}


def _normalise_waveform_rise(payload: dict[str, Any]) -> int | float | None:
    """Normalise rise by shape semantics.

    PuRR uses mixed rise semantics:
    - Gaussian-like shapes use a dimensionless ratio.
    - Soft-square/rounded-like shapes use rise as time in seconds.
    """

    rise = payload.get("rise")
    if rise is None or not isinstance(rise, int | float):
        return rise

    shape = payload.get("shape")
    shape_name = shape.value if hasattr(shape, "value") else shape
    if isinstance(shape_name, str):
        shape_name = shape_name.lower()

    if shape_name in _RISE_RATIO_SHAPES:
        return float(rise)
    if shape_name in _RISE_TIME_SHAPES:
        return _seconds_to_picoseconds(float(rise))

    return rise


def _build_waveform_data(waveform_id: str, payload: dict[str, Any]) -> WaveformData:
    """Convert one PuRR pulse-parameter mapping into canonical waveform data."""

    return WaveformData(
        id=waveform_id,
        shape=payload.get("shape"),
        width=_seconds_to_picoseconds(payload.get("width")),
        rise=_normalise_waveform_rise(payload),
        amp=_as_float(payload.get("amp"), default=None),
        drag=_as_float(payload.get("drag"), default=None),
        phase=_as_float(payload.get("phase"), default=None),
        amp_setup=_as_float(payload.get("amp_setup"), default=None),
    )


def _build_waveforms_for_mode(
    qubit_payload: dict[str, Any],
    pulse_key: str,
    pulse_channel: dict[str, Any],
    is_readout: bool = False,
) -> tuple[WaveformData, ...]:
    """Select and convert waveform definitions relevant to a given canonical mode."""

    waveforms: list[WaveformData] = []

    if pulse_key in {"drive", "second_state"}:
        pulse_half = qubit_payload.get("pulse_hw_x_pi_2")
        if isinstance(pulse_half, dict):
            waveforms.append(_build_waveform_data("x_pi_2", pulse_half))
        pulse_full = qubit_payload.get("pulse_hw_x_pi")
        if isinstance(pulse_full, dict):
            waveforms.append(_build_waveform_data("x_pi", pulse_full))
    elif pulse_key in {"measure", "macq"}:
        pulse_measure = qubit_payload.get("pulse_measure")
        if isinstance(pulse_measure, dict):
            waveforms.append(_build_waveform_data("measure", pulse_measure))
    elif pulse_key.endswith("cross_resonance") or pulse_key.endswith(
        "cross_resonance_cancellation"
    ):
        target_id = pulse_key.split(".")[0]
        zx_map = qubit_payload.get("pulse_hw_zx_pi_4")
        if isinstance(zx_map, dict):
            zx_pulse = zx_map.get(target_id)
            if isinstance(zx_pulse, dict):
                waveforms.append(_build_waveform_data("zx_pi_4", zx_pulse))
    elif pulse_key == "reset":
        pulse_reset = qubit_payload.get("ddrop_reset")
        if isinstance(pulse_reset, dict):
            pulse_data = pulse_reset.copy()
            q_amp = pulse_data.pop("qubit_amp", None)
            r_amp = pulse_data.pop("res_amp", None)
            amp = r_amp if is_readout else q_amp
            if amp is not None:
                pulse_data["amp"] = amp
                waveforms.append(_build_waveform_data("ddrop_reset", pulse_data))
    elif pulse_key == "freq_shift":
        pulse_data = {
            "shape": "square",
            "amp": pulse_channel.get("amp"),
            "phase": pulse_channel.get("phase"),
            "width": None,
        }
        waveforms.append(_build_waveform_data("freq_shift", pulse_data))
    elif pulse_key == "acquire":
        pass
    else:
        logger.warning(
            "Unexpected pulse channel type encountered when building waveforms: %s",
            pulse_key,
        )

    return tuple(waveforms)


def _build_acquire_definitions_for_mode(
    qubit_payload: dict[str, Any],
    pulse_key: str,
) -> tuple[AcquireDefinitionData, ...] | None:
    """Build canonical acquisition definitions for readout/acquire modes."""

    if pulse_key not in {"acquire", "macq"}:
        return None

    acquire_payload = qubit_payload.get("measure_acquire")
    if not isinstance(acquire_payload, dict):
        return None

    weights = acquire_payload.get("weights")
    canonical_weights = None
    if isinstance(weights, dict):
        weights = weights.get("samples")
    if isinstance(weights, list):
        canonical_weights = tuple(
            value for value in weights if isinstance(value, int | float | complex)
        )

    # TODO: Resolve the issue with calibration data having d.5 clock cycle delays, which are
    #       not representable in canonical picoseconds. That should likely be handled by the
    #       validation layer.
    #       COMPILER-1336
    # TODO: Resolve open question around operation schema. COMPILER-1338
    return (
        AcquireDefinitionData(
            id="acquire",
            delay=_seconds_to_picoseconds(acquire_payload.get("delay")),
            sync=acquire_payload.get("sync"),
            width=_seconds_to_picoseconds(acquire_payload.get("width")),
            weights=canonical_weights,
        ),
    )


def _build_mode_from_pulse_view(
    *,
    qubit_payload: dict[str, Any],
    pulse_key: str,
    pulse_view: dict[str, Any],
    mode_id: str,
) -> ModeData | None:
    """Build a canonical mode from a PuRR pulse-channel view payload."""

    pulse_channel = pulse_view.get("pulse_channel")
    if not isinstance(pulse_channel, dict) or not isinstance(pulse_channel.get("id"), str):
        return None

    return ModeData(
        id=mode_id,
        channel_id=pulse_channel["id"],
        waveform_definitions=_build_waveforms_for_mode(
            qubit_payload,
            pulse_key,
            pulse_channel,
            is_readout=mode_id.startswith("readout_"),
        ),
        acquire_definitions=_build_acquire_definitions_for_mode(qubit_payload, pulse_key),
        post_process_method=_build_post_process_method(qubit_payload, pulse_key),
    )


def _resolve_resonator_payload(
    *,
    qubit_payload: dict[str, Any],
    quantum_devices: dict[str, Any],
) -> dict[str, Any] | None:
    """Resolve readout resonator payload from inline or top-level references."""

    measure_device = qubit_payload.get("measure_device")
    resonator_payload = measure_device if isinstance(measure_device, dict) else None
    measure_device_id = (
        resonator_payload.get("id") if isinstance(resonator_payload, dict) else None
    )

    if not isinstance(resonator_payload, dict) or not isinstance(
        resonator_payload.get("pulse_channels"), dict
    ):
        resonator_payload = quantum_devices.get(measure_device_id)

    if isinstance(resonator_payload, dict):
        return resonator_payload
    return None


def _build_qubit_modes(
    *,
    quantum_devices: dict[str, Any],
    qubit_payload: dict[str, Any],
) -> tuple[ModeData, ...]:
    """Build canonical modes for a qubit and its associated readout resonator."""

    qubit_id = qubit_payload.get("id")
    if not isinstance(qubit_id, str):
        return ()

    modes: list[ModeData] = []

    pulse_channels = qubit_payload.get("pulse_channels")
    if isinstance(pulse_channels, dict):
        for pulse_key, pulse_view in pulse_channels.items():
            if not isinstance(pulse_view, dict):
                continue

            mode = _build_mode_from_pulse_view(
                qubit_payload=qubit_payload,
                pulse_key=pulse_key,
                pulse_view=pulse_view,
                mode_id=pulse_key,
            )
            if mode is not None:
                modes.append(mode)

    resonator_payload = _resolve_resonator_payload(
        qubit_payload=qubit_payload,
        quantum_devices=quantum_devices,
    )
    if isinstance(resonator_payload, dict):
        resonator_channels = resonator_payload.get("pulse_channels")
        if isinstance(resonator_channels, dict):
            for pulse_key, pulse_view in resonator_channels.items():
                if not isinstance(pulse_view, dict):
                    continue

                mode = _build_mode_from_pulse_view(
                    qubit_payload=qubit_payload,
                    pulse_key=pulse_key,
                    pulse_view=pulse_view,
                    mode_id=f"readout_{pulse_key}",
                )
                if mode is not None:
                    modes.append(mode)

    return tuple(modes)


def _build_readout_probability(
    *,
    error_mitigation: Any,
    qubit_payload: dict[str, Any],
) -> ReadoutProbabilityData | None:
    """Build readout confusion probabilities from PuRR linear mitigation data."""

    qubit_index = qubit_payload.get("index")
    if not isinstance(qubit_index, int):
        return None

    linear_maps = (error_mitigation or {}).get("readout_mitigation", {}).get("linear", {})
    if not isinstance(linear_maps, dict):
        return None

    qubit_map = linear_maps.get(str(qubit_index), linear_maps.get(qubit_index))
    if not isinstance(qubit_map, dict):
        return None

    probability_entries: list[ProbabilityEntry] = []
    for key, probability in qubit_map.items():
        if (
            not isinstance(key, str)
            or "|" not in key
            or not isinstance(probability, int | float)
        ):
            continue

        measured_state, prepared_state = key.split("|", maxsplit=1)
        if not (
            measured_state.lstrip("-").isdigit() and prepared_state.lstrip("-").isdigit()
        ):
            continue

        probability_entries.append(
            ProbabilityEntry(
                prepared_state=int(prepared_state),
                measured_state=int(measured_state),
                probability=float(probability),
            )
        )

    if not probability_entries:
        return None

    return ReadoutProbabilityData(probability_entries=tuple(probability_entries))


def _build_qubits(
    *,
    quantum_devices: dict[str, Any],
    error_mitigation: Any,
) -> tuple[QubitData, ...]:
    """Build canonical qubit records from PuRR quantum-device payloads."""

    qubits: list[QubitData] = []
    for device_payload in quantum_devices.values():
        if not isinstance(device_payload, dict):
            continue
        qubit_index = device_payload.get("index")
        qubit_id = device_payload.get("id")
        if not isinstance(qubit_index, int) or not isinstance(qubit_id, str):
            continue

        qubits.append(
            QubitData(
                id=qubit_id,
                index=qubit_index,
                modes=_build_qubit_modes(
                    quantum_devices=quantum_devices,
                    qubit_payload=device_payload,
                ),
                readout_probability=_build_readout_probability(
                    error_mitigation=error_mitigation,
                    qubit_payload=device_payload,
                ),
            )
        )

    return tuple(sorted(qubits, key=lambda qubit: qubit.index))
