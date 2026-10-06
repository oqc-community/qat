# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Translate configured Q1 sequence IR into the shared Qblox runtime payload."""

from collections.abc import Iterable, Mapping

from qat.backend.qblox.config.specification import (
    AttConfig,
    AwgConfig,
    ConnectionConfig,
    ExpOvershoot0Config,
    ExpOvershoot1Config,
    ExpOvershoot2Config,
    ExpOvershoot3Config,
    FirConfig,
    GainConfig,
    LoConfig,
    MixerConfig,
    ModuleConfig,
    NcoConfig,
    OffsetConfig,
    ScopeAcqConfig,
    SequencerConfig,
    SquareWeightAcq,
    ThresholdedAcqConfig,
    TtlAcqConfig,
)
from qat.backend.qblox.execution import QbloxPackage
from qat.backend.qblox.ir import Sequence
from qat.experimental.dialect.q1_sequence.ir.attrs import (
    ModuleConfigAttr,
    SequencerConfigAttr,
)
from qat.experimental.dialect.q1_sequence.ir.ops import SequenceOp
from qat.experimental.dialect.q1_sequence.target import emit_config, emit_sequence
from qat.experimental.system_data.hardware.qblox.models import QbloxModuleKind
from qat.experimental.system_data.hardware.qblox.target import (
    DEFAULT_QBLOX_TARGET,
    AcquisitionConnectionMode,
)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected {name} to be a mapping")
    return value


def _set_connection_field(
    connection: ConnectionConfig,
    field: str,
    value: str | bool,
) -> None:
    previous = getattr(connection, field)
    if previous is not None and previous != value:
        raise ValueError(
            f"Conflicting Qblox connection values for {field!r}: "
            f"{previous!r} and {value!r}."
        )
    setattr(connection, field, value)


def _connection_config(
    config: Mapping[str, object],
    module_kind: QbloxModuleKind | None,
) -> ConnectionConfig:
    if module_kind is not None and DEFAULT_QBLOX_TARGET.module_spec(module_kind).is_rf:
        complex_connections = sorted(
            str(item["direction"]) + "_".join(str(port_id) for port_id in item["port_ids"])
            for item in config["connections"]
            if len(item["port_ids"]) > 1
        )
        if complex_connections:
            raise ValueError(
                f"{module_kind.value} accepts only one I/O port per connection; "
                f"got {complex_connections!r}."
            )
    connection = ConnectionConfig(
        bulk_value=[
            str(item["direction"]) + "_".join(str(port_id) for port_id in item["port_ids"])
            for item in config["connections"]
        ]
    )
    acquisition_mode = (
        AcquisitionConnectionMode.components
        if module_kind is None
        else DEFAULT_QBLOX_TARGET.module_spec(module_kind).acquisition_connection_mode
    )
    if acquisition_mode is AcquisitionConnectionMode.none and (
        config["acquisition_path_connections"]
        or config["disabled_acquisition_paths"]
        or config["acquisition_disabled"]
    ):
        raise ValueError(f"{module_kind.value} does not support acquisition connections.")
    for item in config["output_path_connections"]:
        if item["path"] is not None:
            _set_connection_field(connection, f"out{item['output_id']}", str(item["path"]))
    for item in config["boolean_output_connections"] or []:
        _set_connection_field(connection, f"out{item['output_id']}", bool(item["enabled"]))
    for output_id in config["disabled_outputs"]:
        _set_connection_field(connection, f"out{output_id}", "off")

    for item in config["acquisition_path_connections"]:
        input_name = f"in{item['input_id']}"
        path = item["path"]
        if acquisition_mode is AcquisitionConnectionMode.combined:
            if path != "IQ":
                raise ValueError(
                    f"{module_kind.value} supports the combined acquisition path only."
                )
            _set_connection_field(connection, "acq", input_name)
            continue
        if path in ("I", "IQ"):
            _set_connection_field(connection, "acq_I", input_name)
        if path in ("Q", "IQ"):
            _set_connection_field(connection, "acq_Q", input_name)
    for path in config["disabled_acquisition_paths"]:
        if acquisition_mode is AcquisitionConnectionMode.combined:
            if path != "IQ":
                raise ValueError(
                    f"{module_kind.value} supports the combined acquisition path only."
                )
            _set_connection_field(connection, "acq", "off")
            continue
        if path in ("I", "IQ"):
            _set_connection_field(connection, "acq_I", "off")
        if path in ("Q", "IQ"):
            _set_connection_field(connection, "acq_Q", "off")
    if config["acquisition_disabled"]:
        if acquisition_mode is AcquisitionConnectionMode.combined:
            _set_connection_field(connection, "acq", "off")
        else:
            _set_connection_field(connection, "acq_I", "off")
            _set_connection_field(connection, "acq_Q", "off")
    if config["combined_acquisition_alias"] is not None:
        _set_connection_field(connection, "acq", bool(config["combined_acquisition_alias"]))
    return connection


def translate_sequencer_config(
    config_attr: SequencerConfigAttr,
    module_kind: QbloxModuleKind | None = None,
) -> SequencerConfig:
    """Translate one verified Q1 sequencer configuration.

    :param config_attr: Fully resolved Q1 sequencer configuration.
    :param module_kind: Installed module kind, when translation is target-specific.
    :returns: Configuration consumed by the shared Qblox engine.
    """

    config = _mapping(emit_config(config_attr), "sequencer configuration")
    nco = _mapping(config["nco"], "NCO configuration") if config["nco"] else {}
    awg = _mapping(config["awg"], "AWG configuration") if config["awg"] else {}
    mixer = _mapping(config["mixer"], "mixer configuration") if config["mixer"] else {}
    marker = (
        _mapping(config["marker_switch"], "marker configuration")
        if config["marker_switch"]
        else {}
    )
    unweighted = (
        _mapping(config["unweighted_acquire"], "unweighted acquisition configuration")
        if config["unweighted_acquire"]
        else {}
    )
    acquire = (
        _mapping(config["acquire"], "acquisition configuration")
        if config["acquire"]
        else {}
    )
    thresholded = (
        _mapping(config["thresholded_acquire"], "threshold configuration")
        if config["thresholded_acquire"]
        else {}
    )

    return SequencerConfig(
        sync_en=config["enable_sync"],
        marker_ovr_en=marker.get("marker_ovr_en"),
        marker_ovr_value=marker.get("marker_ovr_value"),
        connection=_connection_config(config, module_kind),
        nco=NcoConfig(
            freq=nco.get("frequency"),
            phase_offs=nco.get("phase_offs"),
            prop_delay_comp=nco.get("prop_delay_comp"),
            prop_delay_comp_en=nco.get("prop_delay_comp_en"),
        ),
        awg=AwgConfig(**awg),
        mixer=MixerConfig(**mixer),
        demod_en_acq=acquire.get("demod_en_acq"),
        square_weight_acq=SquareWeightAcq(**unweighted),
        thresholded_acq=ThresholdedAcqConfig(**thresholded),
        ttl_acq=TtlAcqConfig(auto_bin_incr_en=acquire.get("auto_bin_incr_en")),
    )


def _set_field(
    values: dict[str, object],
    field: str,
    value: object,
    *,
    origin: str,
    supported_fields: Mapping[str, object],
) -> None:
    if value is None:
        return
    if field not in supported_fields:
        raise ValueError(f"{origin} cannot be represented by legacy Qblox field {field!r}.")
    previous = values.setdefault(field, value)
    if previous != value:
        raise ValueError(
            f"Conflicting values for legacy Qblox field {field!r}: "
            f"{previous!r} and {value!r} from {origin}."
        )


def _translate_output(
    output: Mapping[str, object],
    offsets: dict[str, object],
    attenuations: dict[str, object],
    filters: tuple[dict[str, object], ...],
) -> None:
    output_id = output["output_id"]
    signal = output["output_signal"]
    if signal:
        signal = _mapping(signal, f"output {output_id} signal configuration")
        _set_field(
            attenuations,
            f"out{output_id}",
            signal["attenuation"],
            origin=f"output {output_id}",
            supported_fields=AttConfig.model_fields,
        )
        for source, suffix in (
            ("offset", ""),
            ("offset_path_0", "_path0"),
            ("offset_path_1", "_path1"),
        ):
            _set_field(
                offsets,
                f"out{output_id}{suffix}",
                signal[source],
                origin=f"output {output_id}",
                supported_fields=OffsetConfig.model_fields,
            )

    pulse_shaping = output["pulse_shaping"]
    if pulse_shaping:
        pulse_shaping = _mapping(
            pulse_shaping, f"output {output_id} pulse-shaping configuration"
        )
        for target, target_type, source in zip(
            filters,
            (
                FirConfig,
                ExpOvershoot0Config,
                ExpOvershoot1Config,
                ExpOvershoot2Config,
                ExpOvershoot3Config,
            ),
            (
                "fir_out",
                "exp_overshoot_0_out",
                "exp_overshoot_1_out",
                "exp_overshoot_2_out",
                "exp_overshoot_3_out",
            ),
            strict=True,
        ):
            _set_field(
                target,
                f"out{output_id}",
                pulse_shaping[source],
                origin=f"output {output_id}",
                supported_fields=target_type.model_fields,
            )


def _translate_input(
    module_input: Mapping[str, object],
    module_kind: QbloxModuleKind,
    offsets: dict[str, object],
    attenuations: dict[str, object],
    gains: dict[str, object],
    scope: dict[str, object],
) -> None:
    input_id = module_input["input_id"]
    signal = module_input["input_signal"]
    if signal:
        signal = _mapping(signal, f"input {input_id} signal configuration")
        _set_field(
            attenuations,
            f"in{input_id}",
            signal["attenuation"],
            origin=f"input {input_id}",
            supported_fields=AttConfig.model_fields,
        )
        gain = signal["gain"]
        if isinstance(gain, float) and not gain.is_integer():
            raise ValueError(
                f"Input {input_id} gain {gain} cannot be represented by legacy "
                "integer Qblox configuration."
            )
        _set_field(
            gains,
            f"in{input_id}",
            gain,
            origin=f"input {input_id}",
            supported_fields=GainConfig.model_fields,
        )
        for source, suffix in (
            ("offset", ""),
            ("offset_path_0", "_path0"),
            ("offset_path_1", "_path1"),
        ):
            _set_field(
                offsets,
                f"in{input_id}{suffix}",
                signal[source],
                origin=f"input {input_id}",
                supported_fields=OffsetConfig.model_fields,
            )

    scope_acquire = module_input["scope_acquire"]
    if scope_acquire:
        scope_acquire = _mapping(
            scope_acquire, f"input {input_id} scope acquisition configuration"
        )
        _set_field(
            scope,
            "sequencer_select",
            scope_acquire["sequencer_select"],
            origin=f"input {input_id}",
            supported_fields=ScopeAcqConfig.model_fields,
        )
        scope_paths = (
            (2 * input_id, 2 * input_id + 1)
            if module_kind is QbloxModuleKind.qrc
            else (input_id,)
        )
        for path in scope_paths:
            _set_field(
                scope,
                f"avg_mode_en_path{path}",
                scope_acquire["enable_average_mode"],
                origin=f"input {input_id}",
                supported_fields=ScopeAcqConfig.model_fields,
            )
            if module_kind is QbloxModuleKind.qrc:
                _set_field(
                    scope,
                    f"trigger_mode_path{path}",
                    "sequencer",
                    origin=f"input {input_id}",
                    supported_fields=ScopeAcqConfig.model_fields,
                )


def _oscillator_lanes(kind: QbloxModuleKind, config: Mapping[str, object]) -> set[str]:
    output_ids: set[int] = set()
    input_ids: set[int] = set()
    for connection in config["connections"]:
        direction = connection["direction"]
        port_ids = set(connection["port_ids"])
        if direction != "in":
            output_ids.update(port_ids)
        if direction != "out":
            input_ids.update(port_ids)
    output_ids.update(
        connection["output_id"] for connection in config["output_path_connections"] or ()
    )
    output_ids.update(
        connection["output_id"] for connection in config["boolean_output_connections"] or ()
    )
    output_ids.update(config["disabled_outputs"] or ())
    input_ids.update(
        connection["input_id"]
        for connection in config["acquisition_path_connections"] or ()
    )
    if config["combined_acquisition_alias"] is not None:
        input_ids.add(0)

    if kind is QbloxModuleKind.qcm_rf:
        return {f"out{output_id}" for output_id in output_ids}
    if kind is QbloxModuleKind.qrm_rf:
        return {"out0_in0"} if output_ids or input_ids else set()
    if kind is QbloxModuleKind.qrc:
        lanes = {f"out{output_id}" for output_id in output_ids if output_id >= 2}
        if 0 in output_ids or 0 in input_ids:
            lanes.add("out0_in0")
        if 1 in output_ids or 1 in input_ids:
            lanes.add("out1_in1")
        return lanes
    return set()


def translate_module_config(
    config_attr: ModuleConfigAttr,
    sequencer_configs: Iterable[SequencerConfigAttr],
) -> ModuleConfig:
    """Translate one module configuration shared by its allocated sequencers.

    :param config_attr: Fully resolved Q1 module configuration.
    :param sequencer_configs: Configurations of every sequence allocated on the module.
    :returns: Configuration consumed by the shared Qblox engine.
    """

    config = _mapping(emit_config(config_attr), "module configuration")
    kind = QbloxModuleKind(config["kind"])
    offsets: dict[str, object] = {}
    attenuations: dict[str, object] = {}
    gains: dict[str, object] = {}
    scope: dict[str, object] = {}
    filters: tuple[dict[str, object], ...] = ({}, {}, {}, {}, {})
    for output in config["outputs"]:
        _translate_output(
            _mapping(output, "module output"),
            offsets,
            attenuations,
            filters,
        )
    for module_input in config["inputs"]:
        _translate_input(
            _mapping(module_input, "module input"),
            kind,
            offsets,
            attenuations,
            gains,
            scope,
        )

    oscillators = {
        oscillator["oscillator_id"]: oscillator
        for oscillator in (
            _mapping(item, "local oscillator") for item in config["local_oscillators"]
        )
    }
    oscillator_lanes: dict[str, set[str]] = {}
    for sequencer_attr in sequencer_configs:
        sequencer = _mapping(emit_config(sequencer_attr), "sequencer configuration")
        oscillator_id = sequencer["local_oscillator_id"]
        if oscillator_id is None:
            continue
        if oscillator_id not in oscillators:
            raise ValueError(
                f"Sequencer references missing local oscillator {oscillator_id!r}."
            )
        oscillator_lanes.setdefault(oscillator_id, set()).update(
            _oscillator_lanes(kind, sequencer)
        )

    lo: dict[str, object] = {}
    for oscillator_id, oscillator in oscillators.items():
        lanes = oscillator_lanes.get(oscillator_id, set())
        if not lanes:
            raise ValueError(
                f"Cannot map local oscillator {oscillator_id!r} to a legacy Qblox lane."
            )
        for lane in lanes:
            _set_field(
                lo,
                f"{lane}_freq",
                oscillator["frequency"],
                origin=f"local oscillator {oscillator_id!r}",
                supported_fields=LoConfig.model_fields,
            )
            enable_field = f"{lane}_en"
            if enable_field in LoConfig.model_fields:
                _set_field(
                    lo,
                    enable_field,
                    oscillator["enable"],
                    origin=f"local oscillator {oscillator_id!r}",
                    supported_fields=LoConfig.model_fields,
                )
            elif oscillator["enable"] is not None:
                raise ValueError(
                    f"Local oscillator {oscillator_id!r} enable state cannot be "
                    f"represented for legacy Qblox lane {lane!r}."
                )

    fir, exp0, exp1, exp2, exp3 = filters
    return ModuleConfig(
        offset=OffsetConfig(**offsets),
        attenuation=AttConfig(**attenuations),
        gain=GainConfig(**gains),
        scope_acq=ScopeAcqConfig(**scope),
        lo=LoConfig(**lo),
        fir=FirConfig(**fir),
        exp0=ExpOvershoot0Config(**exp0),
        exp1=ExpOvershoot1Config(**exp1),
        exp2=ExpOvershoot2Config(**exp2),
        exp3=ExpOvershoot3Config(**exp3),
    )


def translate_package(
    sequence_op: SequenceOp,
    module_config: ModuleConfig,
) -> QbloxPackage:
    """Translate one verified sequence to the shared runtime package.

    TODO(COMPILER-1448): Remove this adapter when Q1 configuration is consumed natively.

    :param sequence_op: Fully configured, allocated, and flat Q1 sequence.
    :param module_config: Shared translated configuration for its physical module.
    :returns: Package consumed by the shared Qblox engine.
    """

    sequence_op.verify()
    if (
        sequence_op.instrument_id is None
        or sequence_op.slot_idx is None
        or sequence_op.seq_idx is None
        or sequence_op.sequencer_config is None
        or sequence_op.module_config is None
    ):
        raise ValueError(
            "Qblox package translation requires a fully configured and allocated "
            "SequenceOp."
        )
    return QbloxPackage(
        pulse_channel_id=sequence_op.channel_id.data,
        physical_channel_id=sequence_op.port_id.data,
        instrument_id=sequence_op.instrument_id.data,
        seq_idx=sequence_op.seq_idx.data,
        seq_config=translate_sequencer_config(
            sequence_op.sequencer_config,
            sequence_op.module_config.kind.data,
        ),
        slot_idx=sequence_op.slot_idx.data,
        mod_config=module_config,
        sequence=Sequence(**emit_sequence(sequence_op)),
    )
