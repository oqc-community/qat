# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""PuRR-to-canonical materialisation orchestration for the experimental boundary.

This module keeps the public materialisation entrypoint and coordinates adaptation,
ingress validation, and canonical assembly via domain-specific builders.

Stage architecture
==================

PuRR materialisation is intentionally staged so source-boundary concerns remain
separate from compiler-owned enrichment and canonical assembly:

1. Source version compatibility check.
2. Source payload adaptation into boundary-normalised plain data.
3. Source ingress DTO validation and graph consistency validation.
4. Compiler-owned enrichment required for canonical assembly.
5. Canonical system data construction from validated/enriched ingress DTO.

This separation allows validation responsibility to move upstream over time
without changing canonical assembly responsibilities.
"""

import math
from typing import Any

from pydantic import ValidationError

from qat.experimental.system_data.canonical.schema import (
    AttributeEntry,
    CanonicalSystemData,
    ChannelData,
    OperationData,
    QubitCouplingData,
    QubitData,
    ResetData,
)
from qat.experimental.system_data.materialisers.errors import (
    SourceValidationError,
    UnsupportedSourceVersionError,
)
from qat.experimental.system_data.materialisers.operations.defaults import (
    DefaultOperationBuilder,
)
from qat.experimental.system_data.materialisers.operations.operation_builder import (
    AbstractOperationBuilder,
)
from qat.experimental.system_data.materialisers.purr.ingress.v0_1_0 import PurrIngressV010
from qat.experimental.system_data.materialisers.purr.materialisers.capabilities import (
    _build_acquire_limit,
    _build_acquire_modes,
    _build_reset_methods,
)
from qat.experimental.system_data.materialisers.purr.materialisers.couplings import (
    _build_couplings,
)
from qat.experimental.system_data.materialisers.purr.materialisers.external_resources import (
    ExternalResourceRegistry,
)
from qat.experimental.system_data.materialisers.purr.materialisers.qubits import (
    _build_qubits,
    _get_control_qubit_ids,
    _get_coupled_qubit_ids,
    _has_x_pi_waveform,
)
from qat.experimental.system_data.materialisers.purr.materialisers.signal_paths import (
    _build_channels,
    _build_oscillators,
    _build_ports,
)
from qat.experimental.system_data.materialisers.purr.validate import (
    validate_purr_ingress_graph,
)
from qat.model.target_data import TargetData

_SUPPORTED_PURR_SOURCE_VERSIONS = ("0.1.0",)

# Detected methods are ordered by this priority when selecting the default reset method if
# none is specified in the source payload. This order is also used to sort the supported
# reset methods list in the canonical system data.
_RESET_DEFAULT_ORDER = ("passive", "ddrop")


# QBlox hardware models expose readout as a single combined "macq" pulse channel that
# carries both the measure and acquire pulse channels.  We normalise the payload by
# splitting the channels (see ``_split_combined_readout_channels``).
_QBLOX_COMBINED_READOUT_RENAMES: dict[str, list[str]] = {
    "macq": ["measure", "acquire"],
}


def _is_qubit_device_payload(device_payload: dict[str, Any]) -> bool:
    """Return True when a device payload structurally represents a qubit.

    PuRR payload naming conventions (for example ``Q*`` IDs) are not relied on
    here because IDs are source-specific and may evolve. The qubit ``index``
    field is the stable structural discriminator in supported payloads.
    """

    if isinstance(device_payload.get("index"), int):
        return True

    return isinstance(device_payload.get("measure_device"), dict)


def _detect_supported_reset_methods(payload: dict[str, Any]) -> list[str]:
    """Collect supported reset method types from qubit payload records."""

    found_methods: set[str] = set()
    quantum_devices = payload.get("quantum_devices")
    if isinstance(quantum_devices, dict):
        for device_payload in quantum_devices.values():
            if not isinstance(device_payload, dict):
                continue
            if not _is_qubit_device_payload(device_payload):
                continue

            pulse_channels = device_payload.get("pulse_channels")
            if isinstance(pulse_channels, dict):
                if "reset" in pulse_channels or isinstance(
                    device_payload.get("ddrop_reset"), dict
                ):
                    found_methods.add("ddrop")

    return [method for method in _RESET_DEFAULT_ORDER if method in found_methods]


def _inject_supported_reset_methods(payload: dict[str, Any]) -> dict[str, Any]:
    """Inject top-level reset capability fields into the ingress payload."""

    updated = dict(payload)
    supported_reset_methods = _detect_supported_reset_methods(updated)

    passive_reset_time = updated.get("passive_reset_time")
    if isinstance(passive_reset_time, int | float) and passive_reset_time >= 0:
        if "passive" not in supported_reset_methods:
            supported_reset_methods.insert(0, "passive")

    existing_supported = updated.get("supported_reset_methods")
    if isinstance(existing_supported, list):
        for reset_type in existing_supported:
            if isinstance(reset_type, str) and reset_type not in supported_reset_methods:
                supported_reset_methods.append(reset_type)

    updated["supported_reset_methods"] = supported_reset_methods

    existing_default = updated.get("default_reset_method")
    if isinstance(existing_default, str) and existing_default in supported_reset_methods:
        updated["default_reset_method"] = existing_default
    else:
        updated["default_reset_method"] = (
            supported_reset_methods[0] if supported_reset_methods else None
        )

    return updated


def _inject_target_data_fields(
    adapted_payload: dict[str, Any],
    target_data: TargetData,
) -> dict[str, Any]:
    """Inject compiler-owned target data fields required by ingress DTO validation."""

    payload = dict(adapted_payload)
    payload["passive_reset_time"] = target_data.QUBIT_DATA.passive_reset_time

    physical_channels = payload.get("physical_channels")
    if isinstance(physical_channels, dict):
        updated_physical_channels = {}
        qubit_block_size = target_data.QUBIT_DATA.samples_per_clock_cycle
        resonator_block_size = target_data.RESONATOR_DATA.samples_per_clock_cycle
        qubit_min_blocks = max(
            1,
            math.ceil(
                target_data.QUBIT_DATA.pulse_duration_min
                / (
                    target_data.QUBIT_DATA.sample_time
                    * target_data.QUBIT_DATA.samples_per_clock_cycle
                )
            ),
        )
        resonator_min_blocks = max(
            1,
            math.ceil(
                target_data.RESONATOR_DATA.pulse_duration_min
                / (
                    target_data.RESONATOR_DATA.sample_time
                    * target_data.RESONATOR_DATA.samples_per_clock_cycle
                )
            ),
        )
        qubit_max_blocks = max(1, target_data.QUBIT_DATA.waveform_memory_size - 1)
        resonator_max_blocks = max(1, target_data.RESONATOR_DATA.waveform_memory_size - 1)

        for channel_id, channel_payload in physical_channels.items():
            if not isinstance(channel_payload, dict):
                updated_physical_channels[channel_id] = channel_payload
                continue

            updated_channel_payload = dict(channel_payload)
            acquire_allowed = bool(updated_channel_payload.get("acquire_allowed", False))
            if acquire_allowed:
                updated_channel_payload["block_size"] = resonator_block_size
                updated_channel_payload["min_blocks"] = resonator_min_blocks
                updated_channel_payload["max_blocks"] = resonator_max_blocks
            else:
                updated_channel_payload["block_size"] = qubit_block_size
                updated_channel_payload["min_blocks"] = qubit_min_blocks
                updated_channel_payload["max_blocks"] = qubit_max_blocks
            updated_physical_channels[channel_id] = updated_channel_payload

        payload["physical_channels"] = updated_physical_channels

    return payload


def _inject_native_waveform_shapes(
    adapted_payload: dict[str, Any],
    native_waveform_shapes: list[str],
) -> dict[str, Any]:
    """Inject compiler-owned native waveform shape fields required by ingress DTO
    validation."""

    payload = dict(adapted_payload)
    physical_channels = payload.get("physical_channels")
    if isinstance(physical_channels, dict):
        updated_physical_channels = {}
        for channel_id, channel_payload in physical_channels.items():
            if not isinstance(channel_payload, dict):
                updated_physical_channels[channel_id] = channel_payload
                continue

            updated_channel_payload = dict(channel_payload)
            updated_channel_payload.setdefault(
                "native_waveform_shapes", tuple(native_waveform_shapes)
            )
            updated_physical_channels[channel_id] = updated_channel_payload

        payload["physical_channels"] = updated_physical_channels

    return payload


def _split_combined_readout_channels(node: Any) -> Any:
    """Split QBlox combined ``macq`` readout channels into ``measure``/``acquire``.

    QBlox hardware models combines the ``acquire`` and ``measure`` pulse channels into
    a single ``macq`` readout channel, to match hardware requirements. To keep
    materialise consistent we need to split these channels to match the schemas
    data structure.

    The transform is applied recursively so it catches the resonator wherever PuRR reads
    it: both as a top-level ``quantum_devices`` entry and as an inline ``measure_device``
    on a qubit (which PuRR resolves in preference to the top-level copy).  Resonators
    without a combined ``macq`` channel are left untouched, and the input is not mutated.
    """
    if isinstance(node, list):
        return [_split_combined_readout_channels(item) for item in node]
    if not isinstance(node, dict):
        return node

    new_node = {key: _split_combined_readout_channels(value) for key, value in node.items()}

    pulse_channels = new_node.get("pulse_channels")
    if isinstance(pulse_channels, dict):
        for old_macq_key, new_keys in _QBLOX_COMBINED_READOUT_RENAMES.items():
            macq_view = pulse_channels.get(old_macq_key)
            if isinstance(macq_view, dict):
                new_channels = {
                    key: value
                    for key, value in pulse_channels.items()
                    if key != old_macq_key
                }
                # Both roles reference the same physical readout channel.
                for new_key in new_keys:
                    new_channels.setdefault(new_key, macq_view)
                new_node["pulse_channels"] = new_channels
    return new_node


class PurrMaterialiserV010:
    """Template-method materialiser for PuRR v0.1.0 source payloads.

    Subclass and override :meth:`prepare_ingress`, :meth:`build_qubits`,
    :meth:`build_channels`, :meth:`build_couplings`, or :meth:`assemble` to
    customise specific pipeline stages without duplicating the full flow.

    :meth:`materialise` orchestrates the standard validation and assembly
    stages. Subclasses are intended to customise the flow through the hooks
    documented above.
    """

    def __init__(
        self,
        *,
        target_data: TargetData | None = None,
        supported_acquire_modes: list[str] | None = None,
        native_waveform_shapes: list[str] | None = None,
        operation_builder_type: type[AbstractOperationBuilder] = DefaultOperationBuilder,
        extra_operations: tuple[OperationData, ...] = (),
    ) -> None:
        self._target_data = target_data if target_data is not None else TargetData()
        self._supported_acquire_modes = (
            supported_acquire_modes
            if supported_acquire_modes is not None
            else ["integrator", "raw", "scope"]
        )
        self._native_waveform_shapes = (
            native_waveform_shapes if native_waveform_shapes is not None else ["square"]
        )
        self._operation_builder_type = operation_builder_type
        self._extra_operations = extra_operations

    def prepare_ingress(
        self,
        *,
        adapted_payload: dict[str, Any],
        source_ingress_dto: PurrIngressV010,
    ) -> dict[str, Any]:
        """Enrich the adapted payload before canonical assembly.

        Called after boundary validation; the returned dict is re-validated before assembly.
        Override to inject hardware-specific fields alongside or instead of the standard
        compiler enrichment.

        :param adapted_payload: Boundary-normalised payload from the adapter.
        :param source_ingress_dto: Validated ingress DTO from the adapted payload.
        :returns: Enriched payload dict ready for the second model_validate pass.
        """
        payload = _inject_target_data_fields(adapted_payload, self._target_data)
        payload = _inject_supported_reset_methods(payload)
        payload = _inject_native_waveform_shapes(payload, self._native_waveform_shapes)
        payload.setdefault(
            "supported_acquire_modes",
            list(
                source_ingress_dto.supported_acquire_modes or self._supported_acquire_modes
            ),
        )
        # TODO: COMPILER-1441 - Move this function to a better location such as ``adapter.py``
        payload = _split_combined_readout_channels(payload)
        return payload

    def build_operation_builder(
        self,
        *,
        qubit_payload: dict[str, Any],
        reset_methods: tuple[ResetData, ...],
        default_reset_method: str | None,
        ddrop_delay_ps: int | None,
    ) -> AbstractOperationBuilder:
        """Build the operation builder for a single qubit.

        Override to supply a hardware-specific builder or to inject extra
        constructor arguments for a particular qubit.

        :param qubit_payload: Raw PuRR quantum-device payload for the qubit.
        :param reset_methods: Reset method objects from canonical assembly.
        :param default_reset_method: Default reset method identifier.
        :param ddrop_delay_ps: DDrop reset delay in picoseconds, or ``None``.
        :returns: A configured, ready-to-use operation builder instance.
        """
        return self._operation_builder_type(
            qubit_id=qubit_payload.get("id"),
            coupled_qubit_ids=_get_coupled_qubit_ids(qubit_payload),
            control_qubit_ids=_get_control_qubit_ids(qubit_payload),
            has_x_pi=_has_x_pi_waveform(qubit_payload),
            reset_methods=reset_methods,
            default_reset_method=default_reset_method,
            ddrop_delay_ps=ddrop_delay_ps,
        )

    def build_qubits(
        self,
        *,
        dto: PurrIngressV010,
        reset_methods: tuple[ResetData, ...],
        default_reset_method: str | None,
    ) -> tuple[QubitData, ...]:
        """Build per-qubit data from the enriched ingress DTO.

        :param dto: Enriched, fully-validated ingress DTO.
        :param reset_methods: Pre-built reset method objects from :meth:`assemble`.
        :param default_reset_method: Default reset method identifier.
        :returns: Tuple of qubit data for canonical assembly.
        """
        return _build_qubits(
            quantum_devices=dto.quantum_devices,
            error_mitigation=dto.error_mitigation,
            build_operation_builder=self.build_operation_builder,
            extra_operations=self._extra_operations,
            reset_methods=reset_methods,
            default_reset_method=default_reset_method,
        )

    def build_channels(self, *, dto: PurrIngressV010) -> tuple[ChannelData, ...]:
        """Build logical channel data from the enriched ingress DTO.

        :param dto: Enriched, fully-validated ingress DTO.
        :returns: Tuple of channel data for canonical assembly.
        """
        return _build_channels(
            quantum_devices=dto.quantum_devices,
            physical_channels=dto.physical_channels,
        )

    def build_couplings(self, *, dto: PurrIngressV010) -> tuple[QubitCouplingData, ...]:
        """Build qubit coupling data from the enriched ingress DTO.

        :param dto: Enriched, fully-validated ingress DTO.
        :returns: Tuple of coupling data for canonical assembly.
        """
        return _build_couplings(
            qubit_direction_couplings=dto.qubit_direction_couplings,
            quantum_devices=dto.quantum_devices,
        )

    def assemble(
        self,
        *,
        dto: PurrIngressV010,
        source_version: str,
    ) -> CanonicalSystemData:
        """Assemble canonical system data from the enriched ingress DTO.

        Calls :meth:`build_qubits`, :meth:`build_channels`, and
        :meth:`build_couplings`. Override to substitute a different output
        model or to add top-level fields (e.g. an extended hardware model).

        :param dto: Enriched, fully-validated ingress DTO.
        :param source_version: Source contract version, written to metadata.
        :returns: Assembled canonical system data.
        """
        external_resources = ExternalResourceRegistry()
        acquire_modes, default_acquire_mode = _build_acquire_modes(
            dto.supported_acquire_modes,
            dto.default_acquire_mode,
        )
        reset_methods, default_reset_method = _build_reset_methods(
            dto.supported_reset_methods,
            dto.default_reset_method,
            dto.passive_reset_time,
        )
        return CanonicalSystemData(
            calibration_id=dto.calibration_id,
            acquire_limit=_build_acquire_limit(dto.repeat_limit),
            acquire_modes=acquire_modes,
            default_acquire_mode=default_acquire_mode,
            reset_methods=reset_methods,
            default_reset_method=default_reset_method,
            oscillators=_build_oscillators(dto.basebands, external_resources),
            ports=_build_ports(dto.physical_channels, external_resources),
            channels=self.build_channels(dto=dto),
            qubits=self.build_qubits(
                dto=dto,
                reset_methods=reset_methods,
                default_reset_method=default_reset_method,
            ),
            couplings=self.build_couplings(dto=dto),
            external_resources=external_resources.to_tuple(),
            metadata=(
                AttributeEntry(key="materialiser_source_type", value="purr"),
                AttributeEntry(key="materialiser_source_version", value=source_version),
                AttributeEntry(
                    key="materialiser_status",
                    value="experimental_partial_mapping",
                ),
            ),
        )

    def materialise(
        self,
        *,
        adapted_payload: dict[str, Any],
        source_version: str,
        strict_version_check: bool = True,
    ) -> CanonicalSystemData:
        """Run the full PuRR materialisation pipeline.

        Orchestrates all pipeline stages. Boundary validation is non-bypassable;
        subclass customisation is via :meth:`prepare_ingress`, :meth:`build_qubits`,
        :meth:`build_channels`, :meth:`build_couplings`, and :meth:`assemble`.

        :param adapted_payload: Pre-adapted (boundary-normalised) PuRR payload.
            Callers are responsible for running the source-specific adapter before
            invoking this method; passing a raw jsonpickle payload will fail at
            ingress DTO validation with a confusing error.
        :param source_version: Source contract version.
        :param strict_version_check: When ``True`` (default), raises
            :class:`~qat.experimental.system_data.materialisers.errors.UnsupportedSourceVersionError`
            if ``source_version`` is not in :data:`_SUPPORTED_PURR_SOURCE_VERSIONS`.
            Set to ``False`` to attempt materialisation with an unrecognised version;
            DTO validation may still fail if the payload shape is incompatible.
        :returns: Materialised canonical system data.
        :raises UnsupportedSourceVersionError: If ``strict_version_check`` is ``True``
            and the source version is not supported.
        :raises SourceValidationError: If DTO or graph validation fails.
        """
        if strict_version_check and source_version not in _SUPPORTED_PURR_SOURCE_VERSIONS:
            raise UnsupportedSourceVersionError.for_version(
                source_type="purr",
                source_version=source_version,
                supported_versions=_SUPPORTED_PURR_SOURCE_VERSIONS,
            )

        try:
            source_ingress_dto = PurrIngressV010.model_validate(adapted_payload)
        except ValidationError as exc:
            raise SourceValidationError(
                "PuRR ingress DTO validation failed.",
                source_type="purr",
                source_version=source_version,
                details={"errors": exc.errors(include_url=False)},
                cause=exc,
            ) from exc

        validate_purr_ingress_graph(source_ingress_dto)

        enriched_payload = self.prepare_ingress(
            adapted_payload=adapted_payload,
            source_ingress_dto=source_ingress_dto,
        )

        try:
            enriched_dto = PurrIngressV010.model_validate(enriched_payload)
        except ValidationError as exc:
            raise SourceValidationError(
                "PuRR payload could not be prepared for materialisation.",
                source_type="purr",
                source_version=source_version,
                details={"errors": exc.errors(include_url=False)},
                cause=exc,
            ) from exc

        return self.assemble(dto=enriched_dto, source_version=source_version)
