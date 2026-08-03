# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Structural and referential validation for canonical system data.

The public entry point is :func:`validate`.  All other functions are internal helpers.

Validation rules
----------------

* **Structural presence** -- qubits, ports, channels, and oscillators must be present.
* **Acquire limit** -- ``acquire_limit`` must be ``-1`` (unlimited) or strictly positive.
* **Ports** -- ``sample_time`` strictly positive; ``block_size`` / ``min_blocks`` at least
  one; ``max_blocks`` either ``-1`` or at least one; ``min_blocks <= max_blocks`` when
  bounded.
* **Oscillators** -- ``frequency`` strictly positive.
* **Channels** -- ``frequency`` non-negative; ``port_id`` and ``oscillator_reference``
  must resolve to declared resources.
* **Modes** -- ``channel_id`` must resolve to a declared channel; nested waveform and
  acquire definitions and post-processing methods are bounds-checked.
* **Couplings** -- ``source_qubit_id`` / ``target_qubit_id`` must resolve to declared
  qubits.
* **Readout probabilities** -- probabilities within ``[0, 1]`` and normalised per prepared
  state.

Warning-level rules flag missing coupling fidelities and inconsistent port sample times.
"""

from __future__ import annotations

import math
from typing import Any

from qat.experimental.system_data.canonical.schema import (
    AcquireDefinitionData,
    CanonicalSystemData,
    ChannelData,
    LinearMapToRealMethodData,
    MaxLikelihoodMethodData,
    ModeData,
    QubitData,
    ReadoutProbabilityData,
    WaveformData,
)
from qat.experimental.system_data.materialisers.errors import (
    MaterialisationConsistencyError,
    MaterialisationValidationError,
)
from qat.experimental.utils.logging import get_logger

logger = get_logger(__name__)
PROBABILITY_TOLERANCE = 1e-6


def _is_finite_real(value: Any) -> bool:
    """Return ``True`` for a finite real scalar (excluding ``bool``)."""
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _is_finite_number(value: Any) -> bool:
    """Return ``True`` for a finite real or complex scalar (excluding ``bool``)."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int | float):
        return math.isfinite(value)
    if isinstance(value, complex):
        return math.isfinite(value.real) and math.isfinite(value.imag)
    return False


def _raise_validation_error(message: str, *, path: str, details: dict[str, Any]) -> None:
    raise MaterialisationValidationError(
        message,
        path=path,
        details=details,
    )


def _raise_consistency_error(message: str, *, path: str, details: dict[str, Any]) -> None:
    raise MaterialisationConsistencyError(
        message,
        path=path,
        details=details,
    )


def _validate_top_level_collections(canonical: CanonicalSystemData) -> None:
    """Require the core resource collections needed by a usable system model."""
    if not canonical.qubits:
        _raise_consistency_error(
            "Canonical system data contains no qubits.",
            path="$.qubits",
            details={},
        )
    if not canonical.ports:
        _raise_consistency_error(
            "Canonical system data contains no ports.",
            path="$.ports",
            details={},
        )
    if not canonical.channels:
        _raise_consistency_error(
            "Canonical system data contains no channels.",
            path="$.channels",
            details={},
        )
    if not canonical.oscillators:
        _raise_consistency_error(
            "Canonical system data contains no oscillators.",
            path="$.oscillators",
            details={},
        )


def _validate_acquire_limit(canonical: CanonicalSystemData) -> None:
    """Validate the acquire limit.

    ``-1`` denotes an unlimited batch; any other value must be strictly positive.
    """
    acquire_limit = canonical.acquire_limit
    if acquire_limit != -1 and acquire_limit <= 0:
        _raise_validation_error(
            "acquire_limit must be -1 (unlimited) or strictly positive.",
            path="$.acquire_limit",
            details={"value": acquire_limit},
        )


def _validate_ports(canonical: CanonicalSystemData) -> None:
    """Validate port timing and block granularity constraints."""
    for port in canonical.ports:
        path_root = f"$.ports[{port.id}]"
        if port.sample_time <= 0:
            _raise_validation_error(
                "Port sample_time must be strictly positive.",
                path=f"{path_root}.sample_time",
                details={"value": port.sample_time},
            )
        if port.block_size < 1:
            _raise_validation_error(
                "Port block_size must be an integer >= 1.",
                path=f"{path_root}.block_size",
                details={"value": port.block_size},
            )
        if port.min_blocks < 1:
            _raise_validation_error(
                "Port min_blocks must be an integer >= 1.",
                path=f"{path_root}.min_blocks",
                details={"value": port.min_blocks},
            )
        if port.max_blocks != -1 and port.max_blocks < 1:
            _raise_validation_error(
                "Port max_blocks must be -1 (unbounded) or an integer >= 1.",
                path=f"{path_root}.max_blocks",
                details={"value": port.max_blocks},
            )
        if port.max_blocks != -1 and port.min_blocks > port.max_blocks:
            _raise_validation_error(
                "Port min_blocks must be <= max_blocks.",
                path=path_root,
                details={
                    "min_blocks": port.min_blocks,
                    "max_blocks": port.max_blocks,
                },
            )


def _validate_oscillators(canonical: CanonicalSystemData) -> None:
    """Validate oscillator frequencies."""
    for oscillator in canonical.oscillators:
        if oscillator.frequency <= 0:
            _raise_validation_error(
                "Oscillator frequency must be strictly positive.",
                path=f"$.oscillators[{oscillator.id}].frequency",
                details={"value": oscillator.frequency},
            )


def _validate_channel(
    channel: ChannelData,
    *,
    port_ids: frozenset[str],
    oscillator_ids: frozenset[str],
) -> None:
    """Validate a single channel's frequency and resource references."""
    path_root = f"$.channels[{channel.id}]"
    if channel.frequency < 0:
        _raise_validation_error(
            "Channel frequency must be non-negative.",
            path=f"{path_root}.frequency",
            details={"value": channel.frequency},
        )
    if channel.port_id not in port_ids:
        _raise_consistency_error(
            "Channel references unknown port.",
            path=f"{path_root}.port_id",
            details={"port_id": channel.port_id},
        )
    if (
        channel.oscillator_reference is not None
        and channel.oscillator_reference not in oscillator_ids
    ):
        _raise_consistency_error(
            "Channel references unknown oscillator.",
            path=f"{path_root}.oscillator_reference",
            details={"oscillator_reference": channel.oscillator_reference},
        )


def _validate_channels(
    canonical: CanonicalSystemData,
    *,
    port_ids: frozenset[str],
    oscillator_ids: frozenset[str],
) -> None:
    """Validate channel frequencies and resource references."""
    for channel in canonical.channels:
        _validate_channel(channel, port_ids=port_ids, oscillator_ids=oscillator_ids)


def _validate_waveform(*, waveform: WaveformData, path_root: str) -> None:
    """Validate waveform timing and numeric bounds."""
    path = f"{path_root}.waveform_definitions[{waveform.id}]"
    if waveform.width is not None and waveform.width < 0:
        _raise_validation_error(
            "Waveform width must be non-negative.",
            path=f"{path}.width",
            details={"value": waveform.width},
        )
    if waveform.rise is not None and (
        not _is_finite_real(waveform.rise) or waveform.rise < 0
    ):
        _raise_validation_error(
            "Waveform rise must be a finite non-negative number.",
            path=f"{path}.rise",
            details={"value": waveform.rise},
        )
    for field_name in ("amp", "drag", "phase", "amp_setup"):
        value = getattr(waveform, field_name)
        if value is not None and not _is_finite_real(value):
            _raise_validation_error(
                f"Waveform {field_name} must be a finite number.",
                path=f"{path}.{field_name}",
                details={"value": value},
            )


def _validate_acquire_definition(
    *,
    acquire: AcquireDefinitionData,
    path_root: str,
) -> None:
    """Validate acquisition timing and weights."""
    path = f"{path_root}.acquire_definitions[{acquire.id}]"

    def validate_field(value, field_name):  # noqa: ANN001
        if value is not None and value < 0:
            _raise_validation_error(
                f"Acquire {field_name} must be non-negative.",
                path=f"{path}.{field_name}",
                details={"value": value},
            )

    validate_field(acquire.delay, "delay")
    validate_field(acquire.width, "width")

    if acquire.weights is not None:
        for position, weight in enumerate(acquire.weights):
            if not _is_finite_number(weight):
                _raise_validation_error(
                    "Acquire weights entries must be finite numeric values.",
                    path=f"{path}.weights[{position}]",
                    details={"value": weight},
                )


def _validate_max_likelihood_method(
    *,
    method: MaxLikelihoodMethodData,
    path: str,
) -> None:
    """Validate a max-likelihood discriminator payload."""
    if not method.states:
        _raise_validation_error(
            "max_likelihood requires a non-empty states mapping.",
            path=f"{path}.states",
            details={},
        )
    for key, params in method.states:
        if not _is_finite_number(params.location):
            _raise_validation_error(
                "max_likelihood state location must be a finite numeric value.",
                path=f"{path}.states[{key}].location",
                details={"value": params.location},
            )
    if not _is_finite_real(method.noise_est):
        _raise_validation_error(
            "max_likelihood noise_est must be a finite number.",
            path=f"{path}.noise_est",
            details={"value": method.noise_est},
        )
    if not _is_finite_real(method.p_min) or method.p_min < 0.0 or method.p_min > 1.0:
        _raise_validation_error(
            "max_likelihood p_min must be a finite number in [0, 1].",
            path=f"{path}.p_min",
            details={"value": method.p_min},
        )
    if method.transform is not None and not all(
        _is_finite_real(value) for row in method.transform for value in row
    ):
        _raise_validation_error(
            "max_likelihood transform entries must be finite numbers.",
            path=f"{path}.transform",
            details={"value": method.transform},
        )
    if method.offset is not None and not all(
        _is_finite_real(value) for value in method.offset
    ):
        _raise_validation_error(
            "max_likelihood offset entries must be finite numbers.",
            path=f"{path}.offset",
            details={"value": method.offset},
        )


def _validate_post_process_method(
    *,
    method: LinearMapToRealMethodData | MaxLikelihoodMethodData,
    path_root: str,
) -> None:
    """Validate a canonical post-processing method payload."""
    path = f"{path_root}.post_process_method"
    if isinstance(method, LinearMapToRealMethodData):
        if len(method.mean_z_map_args) != 2 or not all(
            _is_finite_number(value) for value in method.mean_z_map_args
        ):
            _raise_validation_error(
                "mean_z_map_args must contain exactly two finite numeric entries.",
                path=f"{path}.mean_z_map_args",
                details={"value": method.mean_z_map_args},
            )
        return
    _validate_max_likelihood_method(method=method, path=path)


def _validate_mode(
    *,
    qubit: QubitData,
    mode: ModeData,
    channel_ids: frozenset[str],
) -> None:
    """Validate one mode's channel reference and nested definitions."""
    path_root = f"$.qubits[{qubit.id}].modes[{mode.id}]"
    if mode.channel_id not in channel_ids:
        _raise_consistency_error(
            "Mode references unknown channel.",
            path=f"{path_root}.channel_id",
            details={"channel_id": mode.channel_id},
        )
    for waveform in mode.waveform_definitions:
        _validate_waveform(waveform=waveform, path_root=path_root)
    for acquire in mode.acquire_definitions or ():
        _validate_acquire_definition(acquire=acquire, path_root=path_root)
    if mode.post_process_method is not None:
        _validate_post_process_method(
            method=mode.post_process_method,
            path_root=path_root,
        )


def _validate_readout_probability(
    *,
    qubit: QubitData,
    readout_probability: ReadoutProbabilityData,
) -> None:
    """Validate readout confusion probabilities."""
    path_root = f"$.qubits[{qubit.id}].readout_probability"
    prepared_sums: dict[int, float] = {}
    for position, entry in enumerate(readout_probability.probability_entries):
        if not _is_finite_real(entry.probability):
            _raise_validation_error(
                "Readout probability must be a finite number.",
                path=f"{path_root}.probability_entries[{position}].probability",
                details={"value": entry.probability},
            )
        if entry.probability < 0.0 or entry.probability > 1.0:
            _raise_validation_error(
                "Readout probability must lie in [0, 1].",
                path=f"{path_root}.probability_entries[{position}].probability",
                details={"value": entry.probability},
            )
        prepared_sums[entry.prepared_state] = (
            prepared_sums.get(entry.prepared_state, 0.0) + entry.probability
        )
    for prepared_state, total in prepared_sums.items():
        if abs(total - 1.0) > PROBABILITY_TOLERANCE:
            _raise_consistency_error(
                "Readout probabilities must sum to 1 for each prepared state.",
                path=path_root,
                details={"prepared_state": prepared_state, "sum": total},
            )


def _validate_qubits(
    canonical: CanonicalSystemData,
    *,
    channel_ids: frozenset[str],
) -> None:
    """Validate each qubit's modes and readout probabilities."""
    for qubit in canonical.qubits:
        for mode in qubit.modes:
            _validate_mode(qubit=qubit, mode=mode, channel_ids=channel_ids)
        if qubit.readout_probability is not None:
            _validate_readout_probability(
                qubit=qubit,
                readout_probability=qubit.readout_probability,
            )


def _validate_couplings(
    canonical: CanonicalSystemData,
    *,
    qubit_ids: frozenset[str],
) -> None:
    """Validate coupling references against declared qubits."""
    for position, coupling in enumerate(canonical.couplings):
        for role, qubit_id in (
            ("source_qubit_id", coupling.source_qubit_id),
            ("target_qubit_id", coupling.target_qubit_id),
        ):
            if qubit_id not in qubit_ids:
                _raise_consistency_error(
                    "Coupling references unknown qubit.",
                    path=f"$.couplings[{position}].{role}",
                    details={role: qubit_id},
                )


def _warn_missing_coupling_fidelity(canonical: CanonicalSystemData) -> None:
    """Warn when a coupling declares no gate fidelities."""
    missing = [
        (coupling.source_qubit_id, coupling.target_qubit_id)
        for coupling in canonical.couplings
        if not coupling.gate_fidelities
    ]
    if missing:
        logger.warning(
            "Coupling entries are missing gate fidelity values. Affected qubit pairs: %s",
            missing,
        )


def _warn_sample_time_consistency(canonical: CanonicalSystemData) -> None:
    """Warn when ports of the same acquisition role have heterogeneous sample times."""
    acquire_times = {port.sample_time for port in canonical.ports if port.acquire_allowed}
    drive_times = {port.sample_time for port in canonical.ports if not port.acquire_allowed}
    if len(acquire_times) > 1:
        logger.warning(
            "Acquire-capable ports have inconsistent sample_time values. "
            "Sample times observed: %s",
            sorted(acquire_times),
        )
    if len(drive_times) > 1:
        logger.warning(
            "Drive ports have inconsistent sample_time values. Sample times observed: %s",
            sorted(drive_times),
        )


def validate(model: CanonicalSystemData) -> None:
    """Validate a canonical system data model.

    Enforces structural, field-level, and referential invariants on
    :class:`~qat.experimental.system_data.canonical.schema.CanonicalSystemData`.
    Fatal issues raise
    :class:`~qat.experimental.system_data.materialisers.errors.MaterialisationValidationError`
    (field-level bounds) or
    :class:`~qat.experimental.system_data.materialisers.errors.MaterialisationConsistencyError`
    (referential integrity and cross-entry consistency).  Non-fatal issues are logged
    as warnings and do not block usage.

    :param model: The :class:`CanonicalSystemData` model to validate.
    """
    _validate_top_level_collections(model)
    _validate_acquire_limit(model)
    _validate_ports(model)
    _validate_oscillators(model)

    port_ids = frozenset(port.id for port in model.ports)
    oscillator_ids = frozenset(osc.id for osc in model.oscillators)
    channel_ids = frozenset(channel.id for channel in model.channels)
    qubit_ids = frozenset(qubit.id for qubit in model.qubits)

    _validate_channels(model, port_ids=port_ids, oscillator_ids=oscillator_ids)
    _validate_qubits(model, channel_ids=channel_ids)
    _validate_couplings(model, qubit_ids=qubit_ids)

    # Warning-level checks: log non-fatal issues that may indicate payload
    # incompleteness or unintended heterogeneity, but do not block usage.
    _warn_missing_coupling_fidelity(model)
    _warn_sample_time_consistency(model)
