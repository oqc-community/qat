# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Runtime payload for target-emitted Q1 sequence data."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.metadata import version
from json import dumps
from math import isfinite
from typing import TypeAlias

from pydantic import ConfigDict, Field, field_validator, model_serializer, model_validator
from typing_extensions import TypeAliasType

from qat.executables import AbstractProgram
from qat.experimental.dialect.q1_sequence.ir.ops import SequenceOp
from qat.experimental.dialect.q1_sequence.target import emit_config, emit_sequence
from qat.utils.pydantic import NoExtraFieldsModel

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue = TypeAliasType(
    "JsonValue", "JsonScalar | list[JsonValue] | dict[str, JsonValue]"
)
JsonObject: TypeAlias = dict[str, JsonValue]
# TODO(COMPILER-1449): Stamp compatibility during target translation and enforce it at
# the execution boundary rather than during payload validation.
_QBLOX_DRIVER_VERSION = version("qblox-instruments")
_QBLOX_FIRMWARE_VERSION = "2.0.0"


def _validate_json_mapping(value: Mapping[object, object]) -> JsonObject:
    validated: JsonObject = {}
    for key, nested in value.items():
        if not isinstance(key, str):
            raise TypeError(
                f"Qblox payload mapping keys must be strings, got {type(key).__name__}"
            )
        validated[key] = _validate_json(nested)
    return validated


def _validate_json(value: object) -> JsonValue:
    """Return a JSON-compatible copy of *value* with finite numeric values."""

    if isinstance(value, Mapping):
        return _validate_json_mapping(value)
    if isinstance(value, list | tuple):
        return [_validate_json(nested) for nested in value]
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("Qblox payload floats must be finite")
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(f"Unsupported Qblox payload value {type(value).__name__}")


def _validate_json_object(value: object, name: str) -> JsonObject:
    if not isinstance(value, Mapping):
        raise TypeError(f"Qblox {name} must be a mapping")
    return _validate_json_mapping(value)


class QbloxPackage(NoExtraFieldsModel):
    """One target-emitted Qblox sequencer package.

    The shape mirrors :mod:`qat.backend.qblox.execution`; configuration and sequence
    dictionaries are emitted directly from the Q1 sequence IR.
    """

    pulse_channel_id: str = Field(min_length=1, strict=True)
    physical_channel_id: str = Field(min_length=1, strict=True)
    instrument_id: str = Field(min_length=1, strict=True)
    seq_idx: int = Field(ge=0, strict=True)
    seq_config: JsonObject
    slot_idx: int = Field(ge=1, strict=True)
    mod_config: JsonObject
    sequence: JsonObject

    # TODO(COMPILER-1447): Move SequenceOp translation into a target-side module
    # translator so runtime payload models remain independent of compiler IR.
    @classmethod
    def from_sequence(cls, sequence: SequenceOp) -> QbloxPackage:
        """Snapshot one fully configured and allocated sequence for runtime."""

        sequence.verify()
        if (
            sequence.instrument_id is None
            or sequence.slot_idx is None
            or sequence.seq_idx is None
            or sequence.sequencer_config is None
            or sequence.module_config is None
        ):
            raise ValueError(
                "Qblox package emission requires a fully configured and allocated "
                "SequenceOp"
            )
        return cls(
            pulse_channel_id=sequence.channel_id.data,
            physical_channel_id=sequence.port_id.data,
            instrument_id=sequence.instrument_id.data,
            seq_idx=sequence.seq_idx.data,
            seq_config=emit_config(sequence.sequencer_config),
            slot_idx=sequence.slot_idx.data,
            mod_config=emit_config(sequence.module_config),
            sequence=emit_sequence(sequence),
        )

    @field_validator("seq_config", "mod_config", "sequence", mode="before")
    @classmethod
    def _validate_emitted_mapping(cls, value: object, info) -> JsonObject:
        return _validate_json_object(value, info.field_name)

    @model_validator(mode="after")
    def _validate_emitted_identity(self) -> QbloxPackage:
        module_instrument_id = self.mod_config.get("instrument_id")
        if (
            not isinstance(module_instrument_id, str)
            or module_instrument_id != self.instrument_id
        ):
            raise ValueError(
                "Qblox package instrument_id does not match its module configuration"
            )
        module_slot_idx = self.mod_config.get("slot_idx")
        if (
            isinstance(module_slot_idx, bool)
            or not isinstance(module_slot_idx, int)
            or module_slot_idx != self.slot_idx
        ):
            raise ValueError(
                "Qblox package slot_idx does not match its module configuration"
            )
        port_id = self.seq_config.get("port_id")
        if port_id is not None and (
            not isinstance(port_id, str) or port_id != self.physical_channel_id
        ):
            raise ValueError(
                "Qblox package physical_channel_id does not match its sequencer "
                "configuration"
            )
        return self

    def to_dict(self) -> JsonObject:
        """Return the package as JSON-compatible data."""

        return self._serialize()

    @model_serializer(mode="plain")
    def _serialize(self) -> JsonObject:
        validated = type(self)(
            pulse_channel_id=self.pulse_channel_id,
            physical_channel_id=self.physical_channel_id,
            instrument_id=self.instrument_id,
            seq_idx=self.seq_idx,
            seq_config=self.seq_config,
            slot_idx=self.slot_idx,
            mod_config=self.mod_config,
            sequence=self.sequence,
        )
        return {
            "pulse_channel_id": validated.pulse_channel_id,
            "physical_channel_id": validated.physical_channel_id,
            "instrument_id": validated.instrument_id,
            "seq_idx": validated.seq_idx,
            "seq_config": validated.seq_config,
            "slot_idx": validated.slot_idx,
            "mod_config": validated.mod_config,
            "sequence": validated.sequence,
        }


# TODO(COMPILER-1417): Validate this runtime envelope and its emitted hardware
# configuration against the bSLAM program payload schema.
class QbloxProgram(AbstractProgram):
    """Complete target-emitted Qblox runtime program."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        ser_json_inf_nan="constants",
    )

    packages: dict[str, QbloxPackage] = Field(default_factory=dict)
    driver_version: str = _QBLOX_DRIVER_VERSION
    fw_version: str = _QBLOX_FIRMWARE_VERSION
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _consume_object_type(cls, value: object) -> object:
        if not isinstance(value, Mapping) or "object_type" not in value:
            return value
        fields = dict(value)
        object_type = fields.pop("object_type")
        expected = f"{cls.__module__}.{cls.__name__}"
        if object_type != expected:
            raise ValueError(
                f"Qblox program object_type {object_type!r} does not match {expected!r}"
            )
        return fields

    @field_validator("packages", mode="before")
    @classmethod
    def _validate_package_keys(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            raise TypeError("Qblox packages must be a mapping")
        for key in value:
            if type(key) is not str:
                raise TypeError("Qblox package mapping keys must be strings")
        return value

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: object) -> JsonObject:
        return _validate_json_object(value, "program metadata")

    @model_validator(mode="after")
    def _validate_program(self) -> QbloxProgram:
        if self.driver_version != _QBLOX_DRIVER_VERSION:
            raise ValueError(
                "Qblox driver_version must match the installed qblox-instruments "
                f"version {_QBLOX_DRIVER_VERSION}"
            )
        if self.fw_version != _QBLOX_FIRMWARE_VERSION:
            raise ValueError(f"Qblox fw_version must be {_QBLOX_FIRMWARE_VERSION}")
        allocations: set[tuple[str, int, int]] = set()
        module_configs: dict[tuple[str, int], JsonObject] = {}
        for pulse_channel_id, package in self.packages.items():
            if not pulse_channel_id:
                raise ValueError("Qblox package mapping keys must be non-empty")
            if pulse_channel_id != package.pulse_channel_id:
                raise ValueError(
                    f"Qblox package key {pulse_channel_id!r} does not match "
                    f"pulse_channel_id {package.pulse_channel_id!r}"
                )
            allocation = (package.instrument_id, package.slot_idx, package.seq_idx)
            if allocation in allocations:
                raise ValueError(
                    f"Qblox packages have duplicate physical allocation {allocation!r}"
                )
            allocations.add(allocation)
            module_location = (package.instrument_id, package.slot_idx)
            existing_config = module_configs.setdefault(module_location, package.mod_config)
            if dumps(existing_config, sort_keys=True, separators=(",", ":")) != dumps(
                package.mod_config, sort_keys=True, separators=(",", ":")
            ):
                raise ValueError(
                    "Qblox packages on module "
                    f"{module_location!r} have conflicting module configurations"
                )
        return self

    @property
    def acquire_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return acquisition result shapes once runtime result mapping is available."""

        return {}

    def to_dict(self) -> JsonObject:
        """Return a JSON-compatible representation."""

        return self._serialize()

    @model_serializer(mode="plain")
    def _serialize(self) -> JsonObject:
        packages = {
            pulse_channel_id: QbloxPackage(
                pulse_channel_id=package.pulse_channel_id,
                physical_channel_id=package.physical_channel_id,
                instrument_id=package.instrument_id,
                seq_idx=package.seq_idx,
                seq_config=package.seq_config,
                slot_idx=package.slot_idx,
                mod_config=package.mod_config,
                sequence=package.sequence,
            )
            for pulse_channel_id, package in self.packages.items()
        }
        validated = type(self)(
            packages=packages,
            driver_version=self.driver_version,
            fw_version=self.fw_version,
            metadata=self.metadata,
        )
        return {
            "packages": {
                pulse_channel_id: package.to_dict()
                for pulse_channel_id, package in sorted(validated.packages.items())
            },
            "driver_version": validated.driver_version,
            "fw_version": validated.fw_version,
            "metadata": validated.metadata,
            "object_type": validated.object_type,
        }

    def to_json(self) -> str:
        """Serialize the program deterministically."""

        return dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
