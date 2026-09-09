# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Resolve supplied Qblox configuration into Q1 configuration attributes.

Resolution projects the values a source supplied, the routing it selected, and the
calibration of the canonical channel onto :class:`ModuleConfigAttr` and
:class:`SequencerConfigAttr`. It never invents a value: a field the source left unset stays
absent so a later pass, or the instrument itself, still owns the default. It also never
duplicates target data - the module and sequencer capabilities used to validate the
supplied configuration are read from :data:`DEFAULT_QBLOX_TARGET`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from math import degrees, isclose
from typing import TypeVar

from frozendict import frozendict
from xdsl.dialects.builtin import NoneAttr
from xdsl.ir import Attribute

from qat.experimental.conversion.pulse_to_q1.qblox_configuration.allocation import (
    allocate_sequencers,
)
from qat.experimental.conversion.pulse_to_q1.qblox_configuration.models import (
    ReconciledModuleConfiguration,
    SequencerBinding,
    SequencerPlacement,
)
from qat.experimental.conversion.pulse_to_q1.qblox_configuration.reconciliation import (
    reconcile_configurations,
)
from qat.experimental.dialect.q1_sequence.ir.attrs import (
    AcquireConfigAttr,
    AcquisitionPathConnectionAttr,
    AwgConfigAttr,
    ConnectionAttr,
    InputConfigAttr,
    InputSignalConfigAttr,
    LocalOscillatorConfigAttr,
    MarkerOverrideConfigAttr,
    MixerCorrectionConfigAttr,
    ModuleConfigAttr,
    NcoConfigAttr,
    OutputConfigAttr,
    OutputPathConnectionAttr,
    OutputSignalConfigAttr,
    RealTimePredistortionConfigAttr,
    ScopeAcquireConfigAttr,
    SequencerConfigAttr,
    ThresholdedAcquireConfigAttr,
    UnweightedAcquireConfigAttr,
)
from qat.experimental.system_data.hardware.qblox.configuration import (
    ConfigValue,
    QbloxSequencerConfiguration,
    QbloxSuppliedConfiguration,
    SequencerConnection,
)
from qat.experimental.system_data.hardware.qblox.models import (
    QbloxChannelBinding,
    QbloxModuleKind,
    SignalPath,
    connection_input_ids,
    connection_output_ids,
)
from qat.experimental.system_data.hardware.qblox.target import (
    DEFAULT_QBLOX_TARGET,
    ModuleSpec,
    Q1SequencerFeature,
    Q1SequencerType,
)
from qat.experimental.system_data.hardware.qblox.view import QbloxHardwareView

_ACQUISITION_VALUE_KEYS = (
    "square_weight_acq",
    "thresholded_acq",
    "ttl_acq",
    "demod_en_acq",
)
_PREDISTORTION_GROUPS = ("fir", "exp0", "exp1", "exp2", "exp3")
_LANE_FIELD = re.compile(r"(?:out|in|io)\d+(?:_(?:path[01]|in\d+))?")
_SCOPE_FIELD = re.compile(r"sequencer_select|avg_mode_en_path\d+")
_MODULE_FIELDS: frozendict[str, re.Pattern[str] | None] = frozendict(
    {
        "offset": _LANE_FIELD,
        "attenuation": _LANE_FIELD,
        "gain": _LANE_FIELD,
        "lo": re.compile(r"(?:out|in|io)\d+(?:_in\d+)?_(?:en|freq)"),
        "scope_acq": _SCOPE_FIELD,
        **dict.fromkeys(_PREDISTORTION_GROUPS, _LANE_FIELD),
    }
)
_SEQUENCER_FIELDS: frozendict[str, re.Pattern[str] | None] = frozendict(
    {
        "sync_en": None,
        "marker_ovr_en": None,
        "marker_ovr_value": None,
        "demod_en_acq": None,
        "nco": re.compile(r"freq|phase_offs|prop_delay_comp|prop_delay_comp_en"),
        "awg": re.compile(r"(?:gain|offset)_path[01]|mod_en"),
        "mixer": re.compile(r"phase_offset|gain_ratio"),
        "square_weight_acq": re.compile(r"integration_length"),
        "thresholded_acq": re.compile(r"rotation|threshold"),
        "ttl_acq": re.compile(r"auto_bin_incr_en"),
    }
)
_ValueT = TypeVar("_ValueT", bool, float)
_AttributeT = TypeVar("_AttributeT", bound=Attribute)


def resolve_sequencer_bindings(
    hardware_view: QbloxHardwareView,
    supplied_configurations_by_resource: Mapping[str, QbloxSuppliedConfiguration],
) -> frozendict[str, SequencerBinding]:
    """Resolve every canonical channel of a hardware view onto Q1 configuration.

    :param hardware_view: Derived Qblox projection of the canonical hardware.
    :param supplied_configurations_by_resource: Supplied configurations keyed by external-
        resource identifier.
    :returns: The resolved binding of each canonical channel, keyed by channel identifier.
    :raises ValueError: If the supplied configuration is incomplete, conflicts with the
        canonical calibration, or is invalid for the installed hardware.
    """

    module_configurations = reconcile_configurations(
        hardware_view, supplied_configurations_by_resource
    )
    for module_configuration in module_configurations.values():
        _validate_supplied_configuration(module_configuration)
    placements = allocate_sequencers(module_configurations)
    bindings: dict[str, SequencerBinding] = {}
    for module_configuration in module_configurations.values():
        bindings.update(_module_bindings(module_configuration, placements))
    return frozendict(bindings)


def _validate_supplied_configuration(
    module_configuration: ReconciledModuleConfiguration,
) -> None:
    """Validate everything the canonical ports of one physical module supply.

    Validation runs over the reconciled module values and the whole supplied bank rather
    than only the sequencers allocation happens to pick, because an unusable entry is a
    defect in the source configuration whether or not this program allocates it. Deferring
    the check to allocation would let an invalid entry pass silently until some later
    program with more channels selected it.

    :param module_configuration: Configuration assembled from every canonical port exposing
        the physical module.
    :raises ValueError: If the module values or any supplied sequencer are invalid for the
        installed hardware.
    """

    module_view = module_configuration.module_view
    module_spec = DEFAULT_QBLOX_TARGET.module_spec(module_view.kind)
    _validate_supplied_module(module_configuration, module_spec)
    for port_id, sequencer_bank in module_configuration.sequencer_banks.items():
        for index, sequencer_configuration in sorted(sequencer_bank.items()):
            origin = (
                f"Supplied sequencer {index} of Qblox port {port_id!r} on module "
                f"{module_view.location!r}"
            )
            _validate_supplied_sequencer(
                sequencer_configuration, module_view.kind, module_spec, origin
            )


def _validate_supplied_module(
    module_configuration: ReconciledModuleConfiguration, module_spec: ModuleSpec
) -> None:
    """Validate the module-wide values reconciled for one physical module.

    Scope acquisition is checked here rather than while resolving the lanes a program
    drives, because a module without an acquisition path has no input configuration to carry
    the values onto and would otherwise discard them silently.

    :param module_configuration: Configuration assembled from every canonical port exposing
        the physical module.
    :param module_spec: Target description of the installed module kind.
    :raises ValueError: If a supplied field has no Q1 representation, a configuration group
        is malformed, or the module cannot honour its scope acquisition values.
    """

    origin = f"Module {module_configuration.module_view.location!r}"
    _validate_supplied_fields(module_configuration.module_values, _MODULE_FIELDS, origin)
    scope = _group(module_configuration.module_values, "scope_acq")
    if not scope:
        return
    readout_indices = module_spec.sequencer_indices(Q1SequencerType.readout)
    if not readout_indices or not module_spec.input_count:
        raise ValueError(
            f"{origin} supplies scope acquisition configuration, which a "
            f"{module_spec.kind.value} module has no acquisition path to apply"
        )
    selected = _integer(scope, "sequencer_select", f"{origin} scope acquisition")
    if selected is not None and selected not in readout_indices:
        raise ValueError(
            f"{origin} scope acquisition selects sequencer {selected}, which is not one "
            f"of the acquisition-capable sequencers {list(readout_indices)} of a "
            f"{module_spec.kind.value} module"
        )


def _module_bindings(
    module_configuration: ReconciledModuleConfiguration,
    placements: Mapping[str, SequencerPlacement],
) -> dict[str, SequencerBinding]:
    """Resolve every canonical channel routed through one physical module.

    :param module_configuration: Configuration assembled from every canonical port exposing
        the physical module.
    :param placements: Allocated placements keyed by canonical channel identifier.
    :returns: The resolved binding of each channel routed through the module.
    :raises ValueError: If the module's supplied configuration is invalid.
    """

    module_view = module_configuration.module_view
    module_spec = DEFAULT_QBLOX_TARGET.module_spec(module_view.kind)
    channel_bindings = module_view.channel_bindings
    module_placements = [
        placements[channel_binding.channel_id] for channel_binding in channel_bindings
    ]
    module_config = _module_config(
        module_configuration, module_spec, channel_bindings, module_placements
    )
    return {
        channel_binding.channel_id: SequencerBinding(
            channel_id=channel_binding.channel_id,
            port_id=channel_binding.port_id,
            module_location=placement.module_location,
            sequencer_index=placement.sequencer_index,
            sequencer_config=_sequencer_config(channel_binding, placement, module_spec),
            module_config=module_config,
        )
        for channel_binding, placement in zip(
            channel_bindings, module_placements, strict=True
        )
    }


def _validate_supplied_sequencer(
    sequencer_configuration: QbloxSequencerConfiguration,
    kind: QbloxModuleKind,
    module_spec: ModuleSpec,
    origin: str,
) -> None:
    """Validate one supplied sequencer against the installed module's target description.

    :param sequencer_configuration: Configuration a canonical port supplies for one
        sequencer.
    :param kind: Installed module kind.
    :param module_spec: Target description of the installed module kind.
    :param origin: Human-readable description of the sequencer, used in error messages.
    :raises ValueError: If the sequencer, its routing, or its acquisition configuration is
        invalid for the installed hardware.
    """

    index = sequencer_configuration.index
    if index >= module_spec.sequencer_count:
        raise ValueError(
            f"{origin} is outside the {module_spec.sequencer_count} sequencers of a "
            f"{kind.value} module"
        )
    _validate_supplied_fields(sequencer_configuration.values, _SEQUENCER_FIELDS, origin)
    connection = sequencer_configuration.connection
    if connection is None:
        raise ValueError(f"{origin} supplies no routing")
    _validate_connection_routing(connection, origin)
    if module_spec.is_rf:
        _validate_rf_connection(connection, kind, origin)

    for output_id in sorted(connection.output_ids):
        if output_id >= module_spec.output_count:
            raise ValueError(
                f"{origin} routes to output {output_id}, absent on {kind.value}"
            )
        if index not in DEFAULT_QBLOX_TARGET.output_sequencers(kind, output_id):
            raise ValueError(f"{origin} cannot drive output {output_id} on {kind.value}")
    for input_id in sorted(connection.input_ids):
        if input_id >= module_spec.input_count:
            raise ValueError(
                f"{origin} routes from input {input_id}, absent on {kind.value}"
            )
        if index not in DEFAULT_QBLOX_TARGET.input_sequencers(kind, input_id):
            raise ValueError(f"{origin} cannot read input {input_id} on {kind.value}")

    uses_acquisition = connection.uses_acquisition or any(
        key in sequencer_configuration.values for key in _ACQUISITION_VALUE_KEYS
    )
    if uses_acquisition and not DEFAULT_QBLOX_TARGET.supports(
        kind, index, Q1SequencerFeature.acquisition
    ):
        raise ValueError(f"{origin} configures acquisition but is not acquisition-capable")


def _validate_connection_routing(connection: SequencerConnection, origin: str) -> None:
    """Validate the routing invariants the connections of one sequencer must satisfy.

    A physical lane carries one routing decision: ``connect_out<n>``, ``connect_acq``, and
    the ``connect_acq_I``/``connect_acq_Q`` pair are all single-valued parameters. Two
    supplied entries claiming the same lane, or one entry that both claims and disables a
    lane, therefore describe instrument state that cannot exist.

    :param connection: Routing supplied for the sequencer.
    :param origin: Description of the sequencer, used to describe validation failures.
    :raises ValueError: If two supplied entries make conflicting routing decisions.
    """

    outputs: set[int] = set()
    inputs: set[int] = set()
    for entry in connection.connections:
        entry_outputs = connection_output_ids(entry.direction, entry.port_ids)
        if overlap := outputs.intersection(entry_outputs):
            raise ValueError(f"{origin} connects outputs {sorted(overlap)} more than once")
        outputs.update(entry_outputs)
        entry_inputs = connection_input_ids(entry.direction, entry.port_ids)
        if overlap := inputs.intersection(entry_inputs):
            raise ValueError(f"{origin} connects inputs {sorted(overlap)} more than once")
        inputs.update(entry_inputs)

    bound_outputs = outputs | {
        entry.output_id for entry in connection.output_path_connections
    }
    if overlap := bound_outputs & connection.disabled_outputs:
        raise ValueError(f"{origin} both connects and disables outputs {sorted(overlap)}")
    bound_paths = {entry.path for entry in connection.acquisition_path_connections}
    if overlap := bound_paths & connection.disabled_acquisition_paths:
        raise ValueError(
            f"{origin} both connects and disables acquisition paths "
            f"{[path.value for path in _sorted_paths(overlap)]}"
        )
    if connection.acquisition_enabled and connection.acquisition_disabled:
        raise ValueError(f"{origin} both enables and disables acquisition")
    if connection.acquisition_disabled and (inputs or bound_paths):
        raise ValueError(f"{origin} disables acquisition while connecting inputs")


def _validate_rf_connection(
    connection: SequencerConnection, kind: QbloxModuleKind, origin: str
) -> None:
    """Reject routing an RF module cannot express.

    An RF module exposes one logical I/O port per RF chain and derives the I and Q lanes
    from it internally. Its driver therefore accepts a single port index per connection,
    offers only the combined ``IQ`` state on ``connect_out<n>``, and selects acquisition
    through one combined ``connect_acq`` parameter instead of the ``connect_acq_I`` and
    ``connect_acq_Q`` pair a baseband module exposes. See
    ``qblox_instruments.qcodes_drivers.sequencer.Sequencer``.

    :param connection: Routing supplied for the sequencer.
    :param kind: Installed module kind.
    :param origin: Description of the sequencer, used to describe validation failures.
    :raises ValueError: If the routing names I and Q lanes an RF module does not expose.
    """

    complex_connections = sorted(
        entry.connection for entry in connection.connections if len(entry.port_ids) > 1
    )
    if complex_connections:
        raise ValueError(
            f"{origin} supplies connections {complex_connections!r}; {kind.value} is an "
            "RF module and accepts only one I/O port per connection"
        )
    component_outputs = sorted(
        f"out{entry.output_id}={entry.path.value}"
        for entry in connection.output_path_connections
        if entry.path is not SignalPath.iq
    )
    if component_outputs:
        raise ValueError(
            f"{origin} supplies output paths {component_outputs!r}; {kind.value} is an "
            f"RF module and drives an output with the combined "
            f"{SignalPath.iq.value!r} path only"
        )
    component_paths = _sorted_paths(
        {
            entry.path
            for entry in connection.acquisition_path_connections
            if entry.path is not SignalPath.iq
        }
        | {
            path
            for path in connection.disabled_acquisition_paths
            if path is not SignalPath.iq
        }
    )
    if component_paths:
        raise ValueError(
            f"{origin} supplies acquisition paths "
            f"{[path.value for path in component_paths]!r}; {kind.value} is an RF module "
            "and selects one input for its whole acquisition path"
        )


def _module_config(
    module_configuration: ReconciledModuleConfiguration,
    module_spec: ModuleSpec,
    channel_bindings: Iterable[QbloxChannelBinding],
    placements: Iterable[SequencerPlacement],
) -> ModuleConfigAttr:
    """Resolve the analogue configuration of one physical module.

    Only the lanes the allocated sequencers actually use are configured, so the attribute
    describes the module as this compilation drives it.

    :param module_configuration: Configuration assembled from every canonical port exposing
        the physical module.
    :param module_spec: Target description of the installed module kind.
    :param channel_bindings: Canonical channels routed through the module.
    :param placements: Allocated placements of those channels, in the same order.
    :returns: The resolved module configuration.
    :raises ValueError: If a supplied value conflicts with the canonical calibration.
    """

    module_view = module_configuration.module_view
    values = module_configuration.module_values
    placements = list(placements)
    connections = [placement.sequencer_configuration.connection for placement in placements]
    output_ids = sorted({output_id for c in connections for output_id in c.output_ids})
    input_ids = sorted({input_id for c in connections for input_id in c.input_ids})

    return ModuleConfigAttr(
        module_view.location.slot,
        module_view.location.instrument_id,
        module_view.kind,
        [
            OutputConfigAttr(
                output_id,
                pulse_shaping=_pulse_shaping_config(values, output_id),
                output_signal=_output_signal_config(values, output_id),
            )
            for output_id in output_ids
        ],
        [
            InputConfigAttr(
                input_id,
                input_signal=_input_signal_config(values, input_id),
                scope_acquire=_scope_acquire_config(values, input_id),
            )
            for input_id in input_ids
        ],
        _local_oscillator_configs(
            module_configuration, module_spec, channel_bindings, placements
        ),
    )


def _sequencer_config(
    channel_binding: QbloxChannelBinding,
    placement: SequencerPlacement,
    module_spec: ModuleSpec,
) -> SequencerConfigAttr:
    """Resolve the digital configuration and connections of one sequencer.

    :param channel_binding: Calibrated canonical channel driving the sequencer.
    :param placement: Allocated placement of the channel.
    :param module_spec: Target description of the installed module kind.
    :returns: The resolved sequencer configuration.
    :raises ValueError: If a supplied value conflicts with the canonical calibration.
    """

    values = placement.sequencer_configuration.values
    connection = placement.sequencer_configuration.connection
    origin = (
        f"Sequencer {placement.sequencer_index} of Qblox port {channel_binding.port_id!r}"
    )
    nco = _group(values, "nco")
    awg = _group(values, "awg")
    unweighted = _group(values, "square_weight_acq")
    thresholded = _group(values, "thresholded_acq")
    ttl = _group(values, "ttl_acq")

    frequency = _nco_frequency(channel_binding, nco, origin)
    return SequencerConfigAttr(
        port_id=channel_binding.port_id,
        carrier_frequency=channel_binding.carrier_frequency,
        connections=[
            ConnectionAttr(entry.direction, entry.port_ids)
            for entry in connection.connections
        ],
        output_path_connections=[
            OutputPathConnectionAttr(entry.output_id, entry.path)
            for entry in connection.output_path_connections
        ],
        acquisition_path_connections=[
            AcquisitionPathConnectionAttr(entry.input_id, entry.path)
            for entry in connection.acquisition_path_connections
        ],
        acquisition_enabled=connection.acquisition_enabled,
        disabled_outputs=sorted(connection.disabled_outputs),
        disabled_acquisition_paths=_sorted_paths(connection.disabled_acquisition_paths),
        acquisition_disabled=connection.acquisition_disabled,
        local_oscillator_id=channel_binding.oscillator_id,
        enable_sync=_boolean(values, "sync_en", origin),
        nco=NcoConfigAttr(
            frequency=frequency,
            phase_offs=_number(nco, "phase_offs", origin),
            prop_delay_comp=_integer(nco, "prop_delay_comp", origin),
            prop_delay_comp_en=_boolean(nco, "prop_delay_comp_en", origin),
        ),
        awg=_configured(
            AwgConfigAttr(
                gain_path0=_number(awg, "gain_path0", origin),
                gain_path1=_number(awg, "gain_path1", origin),
                offset_path0=_number(awg, "offset_path0", origin),
                offset_path1=_number(awg, "offset_path1", origin),
                mod_en=_boolean(awg, "mod_en", origin),
            )
        ),
        mixer=_mixer_config(channel_binding, values, module_spec, origin),
        marker_switch=_configured(
            MarkerOverrideConfigAttr(
                marker_ovr_en=_boolean(values, "marker_ovr_en", origin),
                marker_ovr_value=_integer(values, "marker_ovr_value", origin),
            )
        ),
        unweighted_acquire=_configured(
            UnweightedAcquireConfigAttr(_integer(unweighted, "integration_length", origin))
        ),
        acquire=_configured(
            AcquireConfigAttr(
                auto_bin_incr_en=_boolean(ttl, "auto_bin_incr_en", origin),
                demod_en_acq=_boolean(values, "demod_en_acq", origin),
            )
        ),
        thresholded_acquire=_configured(
            ThresholdedAcquireConfigAttr(
                rotation=_number(thresholded, "rotation", origin),
                threshold=_number(thresholded, "threshold", origin),
            )
        ),
    )


def _nco_frequency(
    channel_binding: QbloxChannelBinding,
    nco: Mapping[str, ConfigValue],
    origin: str,
) -> float:
    """Resolve the intermediate frequency the sequencer's NCO must synthesise.

    :param channel_binding: Calibrated canonical channel driving the sequencer.
    :param nco: Supplied NCO values of the sequencer.
    :param origin: Description of the sequencer, used to describe validation failures.
    :returns: The NCO frequency in Hz.
    :raises ValueError: If a supplied NCO frequency contradicts the calibration.
    """

    frequency = float(
        channel_binding.carrier_frequency - (channel_binding.oscillator_frequency or 0)
    )
    supplied_frequency = _number(nco, "freq", origin)
    if supplied_frequency is not None and not isclose(
        supplied_frequency, frequency, rel_tol=1e-12, abs_tol=1.0
    ):
        raise ValueError(
            f"{origin} supplies NCO frequency {supplied_frequency} Hz, but canonical channel "
            f"{channel_binding.channel_id!r} requires {frequency} Hz"
        )
    return frequency


def _mixer_config(
    channel_binding: QbloxChannelBinding,
    values: Mapping[str, ConfigValue],
    module_spec: ModuleSpec,
    origin: str,
) -> MixerCorrectionConfigAttr | None:
    """Resolve the mixer imbalance correction of one sequencer.

    The correction is owned by the canonical calibration of the channel, so a supplied
    value is only ever cross-checked against it.

    :param channel_binding: Calibrated canonical channel driving the sequencer.
    :param values: Supplied sequencer values.
    :param module_spec: Target description of the installed module kind.
    :param origin: Description of the sequencer, used to describe validation failures.
    :returns: The mixer correction, or ``None`` when the module has no mixer.
    :raises ValueError: If a supplied correction contradicts the calibration.
    """

    mixer = _group(values, "mixer")
    if not module_spec.supports_mixer_correction:
        if mixer:
            raise ValueError(
                f"{origin} supplies mixer correction, unsupported on "
                f"{module_spec.kind.value}"
            )
        return None
    phase_offset = degrees(channel_binding.phase_offset)
    for key, calibrated in (
        ("phase_offset", phase_offset),
        ("gain_ratio", channel_binding.imbalance),
    ):
        supplied_value = _number(mixer, key, origin)
        if supplied_value is not None and not isclose(
            supplied_value, calibrated, abs_tol=1e-9
        ):
            raise ValueError(
                f"{origin} supplies mixer {key} {supplied_value}, but canonical channel "
                f"{channel_binding.channel_id!r} is calibrated to {calibrated}"
            )
    return MixerCorrectionConfigAttr(
        phase_offset=phase_offset, gain_ratio=channel_binding.imbalance
    )


def _local_oscillator_configs(
    module_configuration: ReconciledModuleConfiguration,
    module_spec: ModuleSpec,
    channel_bindings: Iterable[QbloxChannelBinding],
    placements: Iterable[SequencerPlacement],
) -> list[LocalOscillatorConfigAttr]:
    """Resolve the local oscillators the module's allocated sequencers mix with.

    :param module_configuration: Configuration assembled from every canonical port exposing
        the physical module.
    :param module_spec: Target description of the installed module kind.
    :param channel_bindings: Canonical channels routed through the module.
    :param placements: Allocated placements of those channels, in the same order.
    :returns: The oscillator configurations, ordered by canonical identifier.
    :raises ValueError: If supplied oscillator values contradict the calibration.
    """

    frequencies = {
        oscillator.oscillator_id: oscillator.frequency
        for oscillator in module_configuration.module_view.oscillators
    }
    lanes: dict[str, set[str]] = {}
    for channel_binding, placement in zip(channel_bindings, placements, strict=True):
        if channel_binding.oscillator_id is None:
            continue
        lanes.setdefault(channel_binding.oscillator_id, set()).update(
            _oscillator_lane_names(placement.sequencer_configuration.connection)
        )

    supplied_values = _group(module_configuration.module_values, "lo")
    oscillators: list[LocalOscillatorConfigAttr] = []
    for oscillator_id, lane_names in sorted(lanes.items()):
        frequency = frequencies[oscillator_id]
        origin = f"Local oscillator {oscillator_id!r} on a {module_spec.kind.value} module"
        declared = _oscillator_value(supplied_values, lane_names, "freq", origin, _number)
        if declared is not None and not isclose(
            declared, frequency, rel_tol=1e-12, abs_tol=1.0
        ):
            raise ValueError(
                f"{origin} is supplied at {declared} Hz, but the canonical calibration "
                f"records {frequency} Hz"
            )
        oscillators.append(
            LocalOscillatorConfigAttr(
                oscillator_id,
                frequency,
                enable=_oscillator_value(
                    supplied_values, lane_names, "en", origin, _boolean
                ),
            )
        )
    return oscillators


def _oscillator_lane_names(connection: SequencerConnection) -> set[str]:
    """Return the module field prefixes naming the lanes one sequencer drives.

    Qblox names a module's oscillator fields after the lanes it feeds: ``out0`` on a
    control module, and ``out0_in0`` where one oscillator serves an output and its paired
    input.

    :param connection: Routing supplied for the sequencer.
    :returns: The field prefixes covering the sequencer's lanes.
    """

    output_ids = sorted(connection.output_ids)
    input_ids = sorted(connection.input_ids)
    if not input_ids:
        return {f"out{output_id}" for output_id in output_ids}
    if not output_ids:
        return {f"in{input_id}" for input_id in input_ids}
    return {
        f"out{output_id}_in{input_id}" for output_id in output_ids for input_id in input_ids
    }


def _oscillator_value(
    supplied_values: Mapping[str, ConfigValue],
    lane_names: Iterable[str],
    suffix: str,
    origin: str,
    read: Callable[[Mapping[str, ConfigValue], str, str], _ValueT | None],
) -> _ValueT | None:
    """Read one supplied oscillator field shared by every lane of an oscillator.

    :param supplied_values: Supplied module-wide ``lo`` values.
    :param lane_names: Field prefixes naming the lanes the oscillator feeds.
    :param suffix: Field suffix, ``en`` or ``freq``.
    :param origin: Description of the oscillator, used to describe validation failures.
    :param read: Typed accessor validating the value.
    :returns: The supplied value, or ``None`` when no lane supplies one.
    :raises ValueError: If the lanes of one oscillator supply different values.
    """

    found = {
        value
        for lane_name in lane_names
        if (value := read(supplied_values, f"{lane_name}_{suffix}", origin)) is not None
    }
    if len(found) > 1:
        raise ValueError(f"{origin} has conflicting {suffix!r} values {sorted(found)}")
    return found.pop() if found else None


def _validate_supplied_fields(
    values: Mapping[str, ConfigValue],
    represented: Mapping[str, re.Pattern[str] | None],
    origin: str,
) -> None:
    """Validate supplied configuration against the field schema of one fragment.

    The source syntax accepts more fields than the Q1 dialect models, and describes a
    configuration group as a mapping of leaf fields. Dropping an extra field, or treating
    a malformed group as an empty one, would silently change how the hardware is driven,
    so both are hard errors. Values the source never set were pruned during
    materialisation, so anything reaching here is a deliberate non-default setting.

    :param values: Supplied values of one module or sequencer.
    :param represented: Field schema of the fragment. A key mapped to ``None`` is a leaf
        the resolver projects whole; a key mapped to a pattern is a group whose leaf names
        must match that pattern.
    :param origin: Description of the fragment, used to describe validation failures.
    :raises ValueError: If a supplied configuration group is not a mapping, or any
        supplied field has no Q1 representation.
    """

    malformed: list[str] = []
    unrepresented: list[str] = []
    for key, value in values.items():
        if key not in represented:
            unrepresented.append(key)
            continue
        pattern = represented[key]
        if pattern is None:
            continue
        if not isinstance(value, Mapping):
            malformed.append(key)
            continue
        unrepresented.extend(
            f"{key}.{leaf}" for leaf in value if pattern.fullmatch(leaf) is None
        )
    if malformed:
        raise ValueError(
            f"{origin} supplies {sorted(malformed)!r} as a value, but a Qblox "
            "configuration group is a mapping of fields"
        )
    if unrepresented:
        raise ValueError(
            f"{origin} supplies {sorted(unrepresented)!r}, which the Q1 dialect cannot "
            "represent"
        )


def _pulse_shaping_config(
    values: Mapping[str, ConfigValue], output_id: int
) -> RealTimePredistortionConfigAttr | None:
    """Resolve the real-time predistortion filters of one physical output.

    Each filter fragment is keyed by output lane name, so ``fir.out0`` configures the
    finite-impulse-response filter of ``out0`` and ``exp2.out0`` its third
    exponential-overshoot filter.

    :param values: Reconciled module-wide values.
    :param output_id: Physical output to configure.
    :returns: The predistortion configuration, or ``None`` when the source supplies none.
    :raises ValueError: If a supplied filter value is not a filter setting string.
    """

    origin = f"Module output out{output_id} predistortion"
    lane = f"out{output_id}"
    return _configured(
        RealTimePredistortionConfigAttr(
            *(
                _string(_group(values, group), lane, origin)
                for group in _PREDISTORTION_GROUPS
            )
        )
    )


def _output_signal_config(
    values: Mapping[str, ConfigValue], output_id: int
) -> OutputSignalConfigAttr | None:
    """Resolve the signal conditioning of one physical output.

    :param values: Reconciled module-wide values.
    :param output_id: Physical output to configure.
    :returns: The output conditioning, or ``None`` when the source supplies none.
    :raises ValueError: If a supplied value has an unexpected type.
    """

    origin = f"Module output out{output_id}"
    offset = _group(values, "offset")
    return _configured(
        OutputSignalConfigAttr(
            attenuation=_number(_group(values, "attenuation"), f"out{output_id}", origin),
            offset=_number(offset, f"out{output_id}", origin),
            offset_path_0=_number(offset, f"out{output_id}_path0", origin),
            offset_path_1=_number(offset, f"out{output_id}_path1", origin),
        )
    )


def _input_signal_config(
    values: Mapping[str, ConfigValue], input_id: int
) -> InputSignalConfigAttr | None:
    """Resolve the signal conditioning of one physical input.

    :param values: Reconciled module-wide values.
    :param input_id: Physical input to configure.
    :returns: The input conditioning, or ``None`` when the source supplies none.
    :raises ValueError: If a supplied value has an unexpected type.
    """

    origin = f"Module input in{input_id}"
    offset = _group(values, "offset")
    return _configured(
        InputSignalConfigAttr(
            attenuation=_number(_group(values, "attenuation"), f"in{input_id}", origin),
            gain=_number(_group(values, "gain"), f"in{input_id}", origin),
            offset=_number(offset, f"in{input_id}", origin),
            offset_path_0=_number(offset, f"in{input_id}_path0", origin),
            offset_path_1=_number(offset, f"in{input_id}_path1", origin),
        )
    )


def _scope_acquire_config(
    values: Mapping[str, ConfigValue], input_id: int
) -> ScopeAcquireConfigAttr | None:
    """Resolve the trace acquisition configuration of one physical input.

    A module writes the acquisitions of a single selected sequencer into its scope memory.
    The selection is a physical sequencer index, and is carried through unchanged so that
    a module configuration shared by several bindings still names the one sequencer the
    source selected. The index itself is validated against the module's acquisition-capable
    sequencers before resolution.

    :param values: Reconciled module-wide values.
    :param input_id: Physical input to configure.
    :returns: The scope configuration, or ``None`` when the source supplies none.
    :raises ValueError: If a supplied value has an unexpected type.
    """

    origin = f"Module input in{input_id} scope acquisition"
    scope = _group(values, "scope_acq")
    return _configured(
        ScopeAcquireConfigAttr(
            sequencer_select=_integer(scope, "sequencer_select", origin),
            enable_average_mode=_boolean(scope, f"avg_mode_en_path{input_id}", origin),
        )
    )


def _configured(attribute: _AttributeT) -> _AttributeT | None:
    """Drop a configuration attribute none of whose parameters the source supplied.

    :param attribute: The constructed configuration attribute.
    :returns: The attribute, or ``None`` when every parameter is absent.
    """

    if all(isinstance(parameter, NoneAttr) for parameter in attribute.parameters):
        return None
    return attribute


def _sorted_paths(paths: Iterable[SignalPath]) -> list[SignalPath]:
    """Return signal paths in a deterministic order.

    :param paths: The signal paths to order.
    :returns: The paths ordered by their Qblox name.
    """

    return sorted(paths, key=lambda path: path.value)


def _group(values: Mapping[str, ConfigValue], key: str) -> Mapping[str, ConfigValue]:
    """Return a nested group of supplied values, empty when the source supplies none.

    Group shape is settled by :func:`_validate_supplied_fields` before resolution, so a
    key present here is always a mapping.

    :param values: Supplied values of a module or sequencer.
    :param key: Source field naming the group.
    :returns: The nested group.
    """

    nested = values.get(key)
    return nested if isinstance(nested, Mapping) else frozendict()


def _boolean(values: Mapping[str, ConfigValue], key: str, origin: str) -> bool | None:
    """Read a supplied boolean value.

    :param values: Supplied values containing the field.
    :param key: Source field to read.
    :param origin: Description of the owner, used to describe validation failures.
    :returns: The supplied value, or ``None`` when the source supplies none.
    :raises ValueError: If the supplied value is not a boolean.
    """

    value = values.get(key)
    if value is None or isinstance(value, bool):
        return value
    raise ValueError(f"{origin} supplies a non-boolean {key!r} value {value!r}")


def _string(values: Mapping[str, ConfigValue], key: str, origin: str) -> str | None:
    """Read a supplied string value.

    :param values: Supplied values containing the field.
    :param key: Source field to read.
    :param origin: Description of the owner, used to describe validation failures.
    :returns: The supplied value, or ``None`` when the source supplies none.
    :raises ValueError: If the supplied value is not a string.
    """

    value = values.get(key)
    if value is None or isinstance(value, str):
        return value
    raise ValueError(f"{origin} supplies a non-string {key!r} value {value!r}")


def _integer(values: Mapping[str, ConfigValue], key: str, origin: str) -> int | None:
    """Read a supplied integer value.

    :param values: Supplied values containing the field.
    :param key: Source field to read.
    :param origin: Description of the owner, used to describe validation failures.
    :returns: The supplied value, or ``None`` when the source supplies none.
    :raises ValueError: If the supplied value is not an integer.
    """

    value = values.get(key)
    if value is None or (isinstance(value, int) and not isinstance(value, bool)):
        return value
    raise ValueError(f"{origin} supplies a non-integer {key!r} value {value!r}")


def _number(values: Mapping[str, ConfigValue], key: str, origin: str) -> float | None:
    """Read a supplied real value.

    :param values: Supplied values containing the field.
    :param key: Source field to read.
    :param origin: Description of the owner, used to describe validation failures.
    :returns: The supplied value, or ``None`` when the source supplies none.
    :raises ValueError: If the supplied value is not a real number.
    """

    value = values.get(key)
    if value is None:
        return None
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    raise ValueError(f"{origin} supplies a non-numeric {key!r} value {value!r}")
