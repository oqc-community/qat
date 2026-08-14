# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Fluent builder for canonical system data.

Provides a chained ``with_*`` API that accepts primitive arguments and constructs the
corresponding canonical schema objects internally. The final frozen
:class:`~qat.experimental.system_data.canonical.schema.CanonicalSystemData` is
assembled on :meth:`~CanonicalSystemDataBuilder.build`.

Example::

    from qat.experimental.system_data.materialisers.builder import (
        CanonicalSystemDataBuilder,
    )

    canonical = (
        CanonicalSystemDataBuilder()
        .with_calibration_id("cal-001")
        .with_acquire_mode("integrator")
        .with_default_acquire_mode("integrator")
        .with_reset_method("passive")
        .with_default_reset_method("passive")
        .with_port("p0", sample_time=1000)
        .with_channel("ch0", port_id="p0", frequency=5_000_000_000)
        .with_qubit("q0", index=0)
        .with_metadata("source", "my-lab")
        .build()
    )

To produce a payload for the materialisation boundary use
:meth:`CanonicalSystemDataBuilder.build_payload`::

    from qat.experimental.system_data.materialisers import boundary

    result = boundary.materialise(
        source_payload=CanonicalSystemDataBuilder()
        .with_calibration_id("cal-001")
        .build_payload()
    )
"""

from __future__ import annotations

import dataclasses
import hashlib
import types as _types
import typing as _typing
from typing import Any, get_args, get_origin, get_type_hints

from qat.experimental.system_data.canonical.schema import (
    AcquireModeData,
    AttributeEntry,
    CanonicalSystemData,
    ChannelData,
    ExternalResourceData,
    ModeData,
    OperationData,
    OscillatorData,
    PortData,
    QubitCouplingData,
    QubitData,
    ReadoutProbabilityData,
    ResetData,
    TwoQubitGateFidelityData,
)
from qat.experimental.system_data.materialisers.model.validation import validate
from qat.experimental.system_data.materialisers.types import SourceType


def build_version_structure_hash() -> str:
    """Build a structural hash of :class:`CanonicalSystemData` for version mismatch
    detection.

    Recurses through all type hints on ``CanonicalSystemData`` and its children. Each field
    contributes a ``field_name[type_description]`` token; dataclass children are expanded
    in-place and generic wrappers are preserved around their recursed descriptions.

    :returns: String hash of the joined field-type description string.
    """
    seen: set[type] = set()

    def describe_type(tp) -> str:  # noqa: ANN001
        if tp is ...:
            return "..."

        origin = get_origin(tp)
        args = get_args(tp)

        if origin is not None:
            is_union = origin is _typing.Union or (
                hasattr(_types, "UnionType") and origin is _types.UnionType
            )
            if is_union:
                return "|".join(describe_type(a) for a in args)
            origin_name = (
                getattr(origin, "__name__", None)
                or getattr(origin, "_name", None)
                or str(origin)
            )
            return f"{origin_name}[{','.join(describe_type(a) for a in args)}]"

        if dataclasses.is_dataclass(tp) and isinstance(tp, type):
            if tp in seen:
                # Guard against recursive types
                return tp.__name__
            seen.add(tp)
            hints = get_type_hints(tp)
            field_strs = [
                f"{f.name}[{describe_type(hints.get(f.name, type(None)))}]"
                for f in dataclasses.fields(tp)
            ]
            seen.discard(tp)
            return f"{tp.__name__}({','.join(field_strs)})"

        if not isinstance(tp, type):
            # Literal values and other non-type annotations.
            return repr(tp)

        return tp.__name__

    hints = get_type_hints(CanonicalSystemData)
    field_strs = [
        f"{f.name}[{describe_type(hints.get(f.name, type(None)))}]"
        for f in dataclasses.fields(CanonicalSystemData)
    ]
    return hashlib.sha256(",".join(field_strs).encode("utf-8")).hexdigest()


version_structure_hash = build_version_structure_hash()


class CanonicalSystemDataBuilder:
    """Fluent builder for
    :class:`~qat.experimental.system_data.canonical.schema.CanonicalSystemData`.

    All ``with_*`` methods accept primitive arguments and build the corresponding
    canonical objects internally, applying sensible defaults for optional fields.
    Methods mutate the builder in place and return ``self``, enabling a chained
    call style.  The final frozen :class:`CanonicalSystemData` is only constructed
    on :meth:`build`.

    Example::

        canonical = (
            CanonicalSystemDataBuilder()
            .with_calibration_id("cal-001")
            .with_acquire_mode("integrator")
            .with_default_acquire_mode("integrator")
            .with_reset_method("passive")
            .with_default_reset_method("passive")
            .with_port("p0", sample_time=1000)
            .with_channel("ch0", port_id="p0", frequency=5_000_000_000)
            .with_qubit("q0", index=0)
            .with_metadata("source", "my-lab")
            .build()
        )
    """

    versioning_key = "_version"
    data_field = "model"

    def __init__(self) -> None:
        self._calibration_id: str = ""
        self._acquire_limit: int = -1
        self._acquire_modes: list[AcquireModeData] = []
        self._default_acquire_mode: str | None = None
        self._reset_methods: list[ResetData] = []
        self._default_reset_method: str | None = None
        self._oscillators: list[OscillatorData] = []
        self._ports: list[PortData] = []
        self._channels: list[ChannelData] = []
        self._qubits: list[QubitData] = []
        self._couplings: list[QubitCouplingData] = []
        self._external_resources: list[ExternalResourceData] = []
        self._metadata: list[AttributeEntry] = []

    def with_calibration_id(self, calibration_id: str) -> CanonicalSystemDataBuilder:
        """Set the calibration identifier.

        :param calibration_id: Calibration identifier string.
        :returns: This builder instance.
        """
        self._calibration_id = calibration_id
        return self

    def with_acquire_limit(self, acquire_limit: int) -> CanonicalSystemDataBuilder:
        """Set the maximum allowed acquisitions per execution batch.

        :param acquire_limit: Acquisition limit, or ``-1`` for unlimited.
        :returns: This builder instance.
        """
        self._acquire_limit = acquire_limit
        return self

    def with_acquire_mode(
        self,
        type: str,
        *,
        attributes: tuple[AttributeEntry, ...] = (),
    ) -> CanonicalSystemDataBuilder:
        """Append a supported acquisition mode descriptor.

        :param type: Acquisition mode type string, for example ``"integrator"`` or
            ``"scope"``.
        :param attributes: Optional additional mode metadata entries.
        :returns: This builder instance.
        """
        self._acquire_modes.append(AcquireModeData(type=type, attributes=attributes))
        return self

    def with_default_acquire_mode(self, mode: str) -> CanonicalSystemDataBuilder:
        """Set the default acquisition mode type.

        :param mode: Acquisition mode type string.  Should match an entry added via
            :meth:`with_acquire_mode`.
        :returns: This builder instance.
        """
        self._default_acquire_mode = mode
        return self

    def with_reset_method(
        self,
        type: str,
        *,
        attributes: tuple[AttributeEntry, ...] = (),
    ) -> CanonicalSystemDataBuilder:
        """Append a supported reset strategy descriptor.

        :param type: Reset strategy type string, for example ``"passive"`` or
            ``"active"``.
        :param attributes: Optional additional strategy metadata entries.
        :returns: This builder instance.
        """
        self._reset_methods.append(
            ResetData(type=type, operation_name=f"{type}_reset", attributes=attributes)
        )
        return self

    def with_default_reset_method(self, method: str) -> CanonicalSystemDataBuilder:
        """Set the default reset strategy type.

        :param method: Reset strategy type string.  Should match an entry added via
            :meth:`with_reset_method`.
        :returns: This builder instance.
        """
        self._default_reset_method = method
        return self

    def with_oscillator(
        self,
        id: str,
        frequency: int,
        *,
        external_resource_id: str | None = None,
    ) -> CanonicalSystemDataBuilder:
        """Append an oscillator configuration.

        :param id: Oscillator identifier.
        :param frequency: Oscillator frequency in Hz.
        :param external_resource_id: Optional linked external resource identifier.
        :returns: This builder instance.
        """
        self._oscillators.append(
            OscillatorData(
                id=id, frequency=frequency, external_resource_id=external_resource_id
            )
        )
        return self

    def with_port(
        self,
        id: str,
        sample_time: int,
        *,
        block_size: int = 1,
        min_blocks: int = 1,
        max_blocks: int = -1,
        acquire_allowed: bool = False,
        native_waveform_shapes: tuple[str, ...] = (),
        external_resource_id: str | None = None,
    ) -> CanonicalSystemDataBuilder:
        """Append a physical port descriptor.

        :param id: Port identifier.
        :param sample_time: Sample period in picoseconds.
        :param block_size: Hardware block granularity in samples.  Defaults to ``1``.
        :param min_blocks: Minimum blocks required per operation.  Defaults to ``1``.
        :param max_blocks: Maximum blocks allowed per operation, or ``-1`` for no
            maximum.  Defaults to ``-1``.
        :param acquire_allowed: Whether acquisition is permitted on this port.
            Defaults to ``False``.
        :param native_waveform_shapes: Natively supported waveform shape names.
        :param external_resource_id: Optional linked external resource identifier.
        :returns: This builder instance.
        """
        self._ports.append(
            PortData(
                id=id,
                sample_time=sample_time,
                block_size=block_size,
                min_blocks=min_blocks,
                max_blocks=max_blocks,
                acquire_allowed=acquire_allowed,
                native_waveform_shapes=native_waveform_shapes,
                external_resource_id=external_resource_id,
            )
        )
        return self

    def with_channel(
        self,
        id: str,
        port_id: str,
        frequency: int,
        *,
        oscillator_reference: str | None = None,
        scale: complex = 1.0 + 0.0j,
        imbalance: float = 1.0,
        phase_offset: float = 0.0,
    ) -> CanonicalSystemDataBuilder:
        """Append a logical channel calibration.

        :param id: Channel identifier.
        :param port_id: Referenced physical port identifier.
        :param frequency: Target channel frequency in Hz.
        :param oscillator_reference: Optional referenced oscillator identifier.
        :param scale: Complex scaling factor.  Defaults to ``1+0j``.
        :param imbalance: IQ gain imbalance factor.  Defaults to ``1.0``.
        :param phase_offset: IQ phase offset in radians.  Defaults to ``0.0``.
        :returns: This builder instance.
        """
        self._channels.append(
            ChannelData(
                id=id,
                port_id=port_id,
                frequency=frequency,
                oscillator_reference=oscillator_reference,
                scale=scale,
                imbalance=imbalance,
                phase_offset=phase_offset,
            )
        )
        return self

    def with_qubit(
        self,
        id: str,
        index: int,
        *,
        modes: tuple[ModeData, ...] = (),
        operations: tuple[OperationData, ...] = (),
        readout_probability: ReadoutProbabilityData | None = None,
    ) -> CanonicalSystemDataBuilder:
        """Append a qubit calibration record.

        :param id: Qubit identifier.
        :param index: Qubit index.
        :param modes: Modes supported by this qubit.  Defaults to an empty tuple.
        :param operations: Operation definitions available on this qubit. Defaults to an
            empty tuple.
        :param readout_probability: Optional readout confusion probabilities.
        :returns: This builder instance.
        """
        self._qubits.append(
            QubitData(
                id=id,
                index=index,
                modes=modes,
                operations=operations,
                readout_probability=readout_probability,
            )
        )
        return self

    def with_coupling(
        self,
        source_qubit_id: str,
        target_qubit_id: str,
        *,
        gate_fidelities: tuple[TwoQubitGateFidelityData, ...] = (),
    ) -> CanonicalSystemDataBuilder:
        """Append a directed coupling descriptor between two qubits.

        :param source_qubit_id: Source qubit identifier.
        :param target_qubit_id: Target qubit identifier.
        :param gate_fidelities: Per-gate fidelity entries for this directed pair. Defaults
            to an empty tuple.
        :returns: This builder instance.
        """
        self._couplings.append(
            QubitCouplingData(
                source_qubit_id=source_qubit_id,
                target_qubit_id=target_qubit_id,
                gate_fidelities=gate_fidelities,
            )
        )
        return self

    def with_external_resource(
        self,
        id: str,
        *,
        object_type: str | None = None,
        attributes: tuple[AttributeEntry, ...] = (),
    ) -> CanonicalSystemDataBuilder:
        """Append an external hardware resource descriptor.

        :param id: Resource identifier.
        :param object_type: Optional descriptive label from the source system.
        :param attributes: Additional unstructured metadata.
        :returns: This builder instance.
        """
        self._external_resources.append(
            ExternalResourceData(id=id, object_type=object_type, attributes=attributes)
        )
        return self

    def with_metadata(
        self,
        key_or_entry: str | AttributeEntry,
        value: Any = None,
    ) -> CanonicalSystemDataBuilder:
        """Append a metadata entry.

        Accepts either an :class:`~qat.experimental.system_data.canonical.schema.AttributeEntry`
        directly, or a ``(key, value)`` shorthand::

            builder.with_metadata(AttributeEntry(key="k", value="v"))
            # or equivalently:
            builder.with_metadata("k", "v")

        :param key_or_entry: An :class:`AttributeEntry` or a metadata key string.
        :param value: Metadata value, required when ``key_or_entry`` is a string.
        :returns: This builder instance.
        :raises TypeError: If ``key_or_entry`` is a string but no ``value`` is supplied,
            or if ``key_or_entry`` is an :class:`AttributeEntry` but ``value`` is also
            supplied.
        """
        if isinstance(key_or_entry, AttributeEntry):
            if value is not None:
                raise TypeError(
                    "with_metadata: do not pass a separate value when providing an "
                    "AttributeEntry instance."
                )
            self._metadata.append(key_or_entry)
        else:
            if value is None:
                raise TypeError(
                    "with_metadata: a value must be supplied when key_or_entry is a "
                    "string key."
                )
            self._metadata.append(AttributeEntry(key=key_or_entry, value=value))
        return self

    def build(self) -> CanonicalSystemData:
        """Construct and return the :class:`CanonicalSystemData` from accumulated state.

        This method is non-destructive: the builder's state is unchanged and
        :meth:`build` may be called again, producing an equal instance.  The constructed
        model is passed to :meth:`validate` before being returned.

        :returns: Frozen :class:`CanonicalSystemData` reflecting all accumulated calls.
        """
        model = CanonicalSystemData(
            calibration_id=self._calibration_id,
            acquire_limit=self._acquire_limit,
            acquire_modes=tuple(self._acquire_modes),
            default_acquire_mode=self._default_acquire_mode,
            reset_methods=tuple(self._reset_methods),
            default_reset_method=self._default_reset_method,
            oscillators=tuple(self._oscillators),
            ports=tuple(self._ports),
            channels=tuple(self._channels),
            qubits=tuple(self._qubits),
            couplings=tuple(self._couplings),
            external_resources=tuple(self._external_resources),
            metadata=tuple(self._metadata),
        )
        validate(model)
        return model

    def build_payload(self) -> dict[str, Any]:
        """Build a source payload mirroring :class:`CanonicalSystemData`'s field structure.

        Produces a shallow dict mapping each :class:`CanonicalSystemData` field name to its
        current value, plus the structural version key required by :func:`materialise_model`.

        :returns: Dict with one key per :class:`CanonicalSystemData` field plus the
            ``_version`` entry.
        """
        canonical = self.build()
        model = {f.name: getattr(canonical, f.name) for f in dataclasses.fields(canonical)}
        model[CanonicalSystemDataBuilder.versioning_key] = version_structure_hash
        results = {
            self.data_field: model,
            "metadata": {"source_type": SourceType.MODEL.value},
        }

        return results
