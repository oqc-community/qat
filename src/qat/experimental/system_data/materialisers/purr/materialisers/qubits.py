# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Qubit-, mode-, waveform-, and readout-probability builders for PuRR materialisation."""

from math import pi
from typing import Any

from qat.experimental.system_data.canonical.schema import (
    AcquireDefinitionData,
    AttributeEntry,
    ModeData,
    OperationData,
    ProbabilityEntry,
    QubitData,
    ReadoutProbabilityData,
    ResetData,
    WaveformData,
)
from qat.experimental.system_data.materialisers.operations.defaults import (
    DefaultOperationBuilder,
)
from qat.experimental.system_data.materialisers.operations.operation_builder import (
    AbstractOperationBuilder,
)
from qat.experimental.system_data.materialisers.purr.materialisers.common import (
    _as_complex,
    _as_float,
    _seconds_to_picoseconds,
)
from qat.experimental.system_data.materialisers.purr.materialisers.postprocess import (
    _build_post_process_method,
)
from qat.experimental.utils.logging import get_logger

logger = get_logger(__name__)


def _build_shape_name_and_parameters(
    shape: str | None,
    amp: float | None,
    width: float | None,
    extra_parameters: dict[str, Any],
) -> tuple[str, tuple[AttributeEntry, ...]] | None:
    """Build a shape attribute entry from the shape name and its parameters.

    The shape name is normalised to lower-case, and the expected parameters for that shape
    are extracted from the waveform payload. Any other parameters are ignored.

    Validation happens prior to manifestation, so just returns None if the shape is not
    recognised, which practically is an impossibility.
    """

    if shape is None:
        return None

    match shape.lower():
        case "blackman":
            return "blackman", ()
        case "cos":
            if width is None:
                return None
            internal_phase = extra_parameters.get("internal_phase", 0.0) + pi / 2
            freq = extra_parameters.get("frequency", 0.0)
            number_of_periods = freq * width
            return "sinusoidal", (
                AttributeEntry(key="internal_phase", value=internal_phase),
                AttributeEntry(key="number_of_periods", value=number_of_periods),
            )
        case "drag_gaussian":
            # The DragGaussian is equivalent to a breadth of 1.0
            return "gaussian", (AttributeEntry(key="fractional_breadth", value=1.0),)
        case "extra_soft_square":
            if width is None:
                return None
            rise = extra_parameters.get("rise", 0.0)
            std_dev = extra_parameters.get("std_dev", 0.0)
            return "soft_square", (
                AttributeEntry(
                    key="fractional_top_width", value=(std_dev - 4.0 * rise) / width
                ),
                AttributeEntry(key="fractional_rise", value=2.0 * rise / width),
                AttributeEntry(key="regularize", value=True),
            )
        case "gaussian":
            rise = extra_parameters.get("rise", 0.0)
            return "gaussian", (
                AttributeEntry(key="fractional_breadth", value=2.0**0.5 * rise),
                AttributeEntry(key="regularize", value=False),
            )
        case "gaussian_zero_edges":
            if width is None:
                return None
            rise = extra_parameters.get("rise", 0.0)
            zero_at_edges = extra_parameters.get("zero_at_edges", False)
            return "gaussian", (
                AttributeEntry(key="fractional_breadth", value=2.0 * rise / width),
                AttributeEntry(key="regularize", value=zero_at_edges),
            )
        case "gaussian_square":
            if width is None:
                return None
            rise = extra_parameters.get("rise", 0.0)
            std_dev = extra_parameters.get("std_dev", 0.0)
            zero_at_edges = extra_parameters.get("zero_at_edges", False)
            return "gaussian_square", (
                AttributeEntry(key="fractional_rise", value=2.0 * rise / width),
                AttributeEntry(key="fractional_top_width", value=std_dev / width),
                AttributeEntry(key="regularize", value=zero_at_edges),
            )
        case "rounded_square":
            if width is None:
                return None
            rise = extra_parameters.get("rise", 0.0)
            std_dev = extra_parameters.get("std_dev", 0.0)
            return "rounded_square", (
                AttributeEntry(key="fractional_top_width", value=std_dev / width),
                AttributeEntry(key="fractional_rise", value=2.0 * rise / width),
            )
        case "sech":
            if width is None:
                return None
            std_dev = extra_parameters.get("std_dev", 0.0)
            return "sech", (
                AttributeEntry(key="fractional_breadth", value=2.0 * std_dev / width),
            )
        case "setup_hold":
            if width is None or amp is None:
                return None
            rise = extra_parameters.get("rise", 0.0)
            amp_setup = extra_parameters.get("amp_setup", 0.0)
            return "setup_hold", (
                AttributeEntry(key="setup", value=amp_setup / amp),
                AttributeEntry(key="rise_location", value=rise / width),
            )
        case "sin":
            if width is None:
                return None
            internal_phase = extra_parameters.get("internal_phase", 0.0)
            freq = extra_parameters.get("frequency", 0.0)
            number_of_periods = freq * width
            return "sinusoidal", (
                AttributeEntry(key="internal_phase", value=internal_phase),
                AttributeEntry(key="number_of_periods", value=number_of_periods),
            )
        case "soft_square":
            if width is None:
                return None
            rise = extra_parameters.get("rise", 0.0)
            return "soft_square", (
                AttributeEntry(key="fractional_top_width", value=1.0 - rise / width),
                AttributeEntry(key="fractional_rise", value=2.0 * rise / width),
            )
        case "softer_gaussian":
            rise = extra_parameters.get("rise", 0.0)
            return "gaussian", (
                AttributeEntry(key="fractional_breadth", value=2.0**0.5 * rise),
                AttributeEntry(key="regularize", value=True),
            )
        case "softer_square":
            if width is None:
                return None
            rise = extra_parameters.get("rise", 0.0)
            std_dev = extra_parameters.get("std_dev", 0.0)
            return "soft_square", (
                AttributeEntry(
                    key="fractional_top_width", value=(std_dev - 2.0 * rise) / width
                ),
                AttributeEntry(key="fractional_rise", value=2.0 * rise / width),
                AttributeEntry(key="regularize", value=True),
            )
        case "square":
            return "square", ()
        case _:
            return None


def _build_waveform_data(waveform_id: str, payload: dict[str, Any]) -> WaveformData | None:
    """Convert one PuRR pulse-parameter mapping into canonical waveform data.

    Extracts the expected parameters for any waveform, then bundles any other parameters
    into a tuple of attribute entries.
    """

    width = _as_float(payload.get("width"), default=None)
    amp = _as_complex(payload.get("amp"), default=None)
    phase = _as_float(payload.get("phase"), default=None)
    purr_shape = payload.get("shape")

    # For drag_gaussian, beta is mapped to drag in the canonical form
    if isinstance(purr_shape, str) and purr_shape.lower() == "drag_gaussian":
        drag = _as_float(payload.get("beta"), default=None)
    else:
        drag = _as_float(payload.get("drag"), default=None)

    purr_parameters = {
        key: value
        for key, value in payload.items()
        if key not in {"shape", "width", "amp", "phase", "drag"}
    }

    built_shape = _build_shape_name_and_parameters(purr_shape, amp, width, purr_parameters)
    if built_shape is None:
        return None
    built_name, built_parameters = built_shape

    return WaveformData(
        id=waveform_id,
        width=_seconds_to_picoseconds(width),
        amp=amp,
        phase=phase,
        drag=drag,
        shape=built_name,
        shape_parameters=built_parameters,
    )


def _append_to_pulse_payload(
    waveforms: list[WaveformData], waveform_id: str, pulse_payload: dict[str, Any]
) -> None:
    """Append a canonical waveform to the list of waveforms for a given pulse channel."""
    waveform = _build_waveform_data(waveform_id, pulse_payload)
    if waveform is not None:
        waveforms.append(waveform)


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
            _append_to_pulse_payload(waveforms, "x_pi_2", pulse_half)
        pulse_full = qubit_payload.get("pulse_hw_x_pi")
        if isinstance(pulse_full, dict):
            _append_to_pulse_payload(waveforms, "x_pi", pulse_full)
    elif pulse_key in {"measure", "macq"}:
        pulse_measure = qubit_payload.get("pulse_measure")
        if isinstance(pulse_measure, dict):
            _append_to_pulse_payload(waveforms, "measure", pulse_measure)
    elif pulse_key.endswith("cross_resonance") or pulse_key.endswith(
        "cross_resonance_cancellation"
    ):
        target_id = pulse_key.split(".")[0]
        zx_map = qubit_payload.get("pulse_hw_zx_pi_4")
        if isinstance(zx_map, dict):
            zx_pulse = zx_map.get(target_id)
            if isinstance(zx_pulse, dict):
                _append_to_pulse_payload(waveforms, "zx_pi_4", zx_pulse)
    elif pulse_key == "reset":
        pulse_reset = qubit_payload.get("ddrop_reset")
        if isinstance(pulse_reset, dict):
            pulse_data = pulse_reset.copy()
            q_amp = pulse_data.pop("qubit_amp", None)
            r_amp = pulse_data.pop("res_amp", None)
            amp = r_amp if is_readout else q_amp
            if amp is not None:
                pulse_data["amp"] = amp
                _append_to_pulse_payload(waveforms, "ddrop_reset", pulse_data)
    elif pulse_key == "freq_shift":
        pulse_data = {
            "shape": "square",
            "amp": pulse_channel.get("amp"),
            "phase": pulse_channel.get("phase"),
            "width": None,
        }
        _append_to_pulse_payload(waveforms, "freq_shift", pulse_data)
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


def _get_coupled_qubit_ids(qubit_payload: dict[str, Any]) -> tuple[str, ...]:
    """Extract target qubit IDs from cross-resonance pulse channels.

    PuRR encodes two-qubit coupling as pulse channels named
    ``"{target_id}.cross_resonance"``. This helper collects the unique set of
    target IDs in definition order, preserving the first occurrence.
    """
    pulse_channels = qubit_payload.get("pulse_channels")
    if not isinstance(pulse_channels, dict):
        return ()
    coupled: list[str] = []
    for pulse_key in pulse_channels:
        if isinstance(pulse_key, str) and pulse_key.endswith(".cross_resonance"):
            target_id = pulse_key.split(".")[0]
            if target_id not in coupled:
                coupled.append(target_id)
    return tuple(coupled)


def _get_control_qubit_ids(qubit_payload: dict[str, Any]) -> tuple[str, ...]:
    """Extract control qubit IDs from cross-resonance-cancellation pulse channels.

    PuRR encodes the target side of a two-qubit coupling as pulse channels named
    ``"{control_id}.cross_resonance_cancellation"``. This helper collects the unique
    set of control IDs in definition order.
    """
    pulse_channels = qubit_payload.get("pulse_channels")
    if not isinstance(pulse_channels, dict):
        return ()
    controls: list[str] = []
    for pulse_key in pulse_channels:
        if isinstance(pulse_key, str) and pulse_key.endswith(
            ".cross_resonance_cancellation"
        ):
            control_id = pulse_key.split(".")[0]
            if control_id not in controls:
                controls.append(control_id)
    return tuple(controls)


def _has_x_pi_waveform(qubit_payload: dict[str, Any]) -> bool:
    """Return ``True`` when the qubit payload contains a calibrated X(π) pulse.

    PuRR stores the full-pi X pulse parameters under ``pulse_hw_x_pi``. When absent
    the ``X_pi`` operation and the ``X``/``Y`` gate variants that reference it must
    be omitted to avoid unresolvable operation references.
    """
    return isinstance(qubit_payload.get("pulse_hw_x_pi"), dict)


def _build_operations(
    qubit_payload: dict[str, Any],
    operation_builder_type: type[AbstractOperationBuilder] = DefaultOperationBuilder,
    extra_operations: tuple[OperationData, ...] = (),
    reset_methods: tuple[ResetData, ...] = (),
    default_reset_method: str | None = None,
) -> tuple[OperationData, ...]:
    """Build the canonical operation set for a qubit.

    Single-qubit operations (X_pi_2, Z, X, Y, U, H, SX, SXdg, S, Sdg, T, Tdg,
    measure, initiate) are always included. ``X_pi`` and the ``X``/``Y`` gate
    variants that reference it are included only when the qubit has a calibrated
    X(π) pulse (``pulse_hw_x_pi`` present in the PuRR payload).

    For each coupled target qubit (detected via cross-resonance pulse channels),
    ZX(±π/4), ECR, and CNOT operations are appended. For each control qubit that
    drives this qubit (detected via cross-resonance-cancellation channels), ZX
    cancellation-tone primitives are appended.

    :param qubit_payload: PuRR quantum-device payload dict for the qubit.
    :param operation_builder_type: Builder class to instantiate. Subclass
        :class:`~qat.experimental.system_data.materialisers.operations.defaults.DefaultOperationBuilder`
        to customise individual operations for a specific hardware target.
    :param extra_operations: Additional or replacement operations (last-wins by ID).
    :param reset_methods: Supported reset strategies from top-level canonical metadata.
    :param default_reset_method: Default reset method type.
    """
    ddrop_reset = qubit_payload.get("ddrop_reset")
    ddrop_delay_ps: int | None = None
    if isinstance(ddrop_reset, dict):
        delay_s = ddrop_reset.get("delay")
        if delay_s is not None:
            ddrop_delay_ps = int(_seconds_to_picoseconds(delay_s))

    builder = operation_builder_type(
        qubit_id=qubit_payload.get("id"),
        coupled_qubit_ids=_get_coupled_qubit_ids(qubit_payload),
        control_qubit_ids=_get_control_qubit_ids(qubit_payload),
        has_x_pi=_has_x_pi_waveform(qubit_payload),
        reset_methods=reset_methods,
        default_reset_method=default_reset_method,
        ddrop_delay_ps=ddrop_delay_ps,
    )
    return builder.build(extra_operations=extra_operations)


def _build_qubits(
    *,
    quantum_devices: dict[str, Any],
    error_mitigation: Any,
    operation_builder_type: type[AbstractOperationBuilder] = DefaultOperationBuilder,
    extra_operations: tuple[OperationData, ...] = (),
    reset_methods: tuple[ResetData, ...] = (),
    default_reset_method: str | None = None,
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
                operations=_build_operations(
                    device_payload,
                    operation_builder_type=operation_builder_type,
                    extra_operations=extra_operations,
                    reset_methods=reset_methods,
                    default_reset_method=default_reset_method,
                ),
                readout_probability=_build_readout_probability(
                    error_mitigation=error_mitigation,
                    qubit_payload=device_payload,
                ),
            )
        )

    return tuple(sorted(qubits, key=lambda qubit: qubit.index))
