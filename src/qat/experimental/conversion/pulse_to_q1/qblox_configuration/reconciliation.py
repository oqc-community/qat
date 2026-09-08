# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Reconcile configuration canonical ports supply for each physical Qblox module.

A physical module has exactly one analogue configuration, but a source describes that
configuration through the canonical ports exposing the module. This module reconciles those
fragments into one module-wide mapping, reporting the precise field two ports disagree on,
and keeps the supplied sequencer banks port-specific. This is configuration reconciliation,
not a hardware-connectivity topology representation.
"""

from __future__ import annotations

from collections.abc import Mapping

from frozendict import frozendict

from qat.experimental.conversion.pulse_to_q1.qblox_configuration.models import (
    ReconciledModuleConfiguration,
)
from qat.experimental.system_data.hardware.qblox.configuration import (
    ConfigValue,
    QbloxSuppliedConfiguration,
)
from qat.experimental.system_data.hardware.qblox.models import (
    QbloxModuleLocation,
    QbloxModuleView,
)
from qat.experimental.system_data.hardware.qblox.view import QbloxHardwareView


def reconcile_configurations(
    hardware_view: QbloxHardwareView,
    supplied_configurations_by_resource: Mapping[str, QbloxSuppliedConfiguration],
) -> frozendict[QbloxModuleLocation, ReconciledModuleConfiguration]:
    """Reconcile supplied configuration for every module of a Qblox hardware view.

    :param hardware_view: Derived Qblox projection of the canonical hardware.
    :param supplied_configurations_by_resource: Supplied configurations keyed by external-
        resource identifier.
    :returns: Reconciled configurations keyed by physical module location.
    :raises ValueError: If a port references configuration that cannot be resolved, or if
        two ports of one module supply conflicting module-wide values.
    """

    return frozendict(
        (
            module_location,
            _reconcile_module(module_view, supplied_configurations_by_resource),
        )
        for module_location, module_view in hardware_view.modules.items()
    )


def _reconcile_module(
    module_view: QbloxModuleView,
    supplied_configurations_by_resource: Mapping[str, QbloxSuppliedConfiguration],
) -> ReconciledModuleConfiguration:
    """Join the configuration supplied through one module's canonical ports.

    :param module_view: The hardware view of the physical module.
    :param supplied_configurations_by_resource: Supplied configurations keyed by external-
        resource identifier.
    :returns: The reconciled configuration of the physical module.
    :raises ValueError: If a reference cannot be resolved or fragments conflict.
    """

    supplied_by_port = {
        port.port_id: supplied_configurations_by_resource.get(port.resource_id)
        for port in module_view.ports
    }
    resolved_by_port = _resolve_references(module_view, supplied_by_port)

    module_values: dict[str, ConfigValue] = {}
    value_sources: dict[str, str] = {}
    for port_id, supplied_configuration in resolved_by_port.items():
        if supplied_configuration.module_configuration is None:
            continue
        _reconcile(
            module_values,
            supplied_configuration.module_configuration.values,
            value_sources=value_sources,
            path="",
            port_id=port_id,
            module_view=module_view,
        )

    return ReconciledModuleConfiguration(
        module_view=module_view,
        module_values=frozendict(module_values),
        sequencer_banks=frozendict(
            (
                port_id,
                frozendict(
                    {
                        sequencer.index: sequencer
                        for sequencer in supplied_configuration.sequencers
                    }
                ),
            )
            for port_id, supplied_configuration in resolved_by_port.items()
        ),
    )


def _resolve_references(
    module_view: QbloxModuleView,
    supplied_by_port: Mapping[str, QbloxSuppliedConfiguration | None],
) -> dict[str, QbloxSuppliedConfiguration]:
    """Resolve ports that reference configuration supplied through a sibling port.

    :param module_view: The hardware view of the physical module.
    :param supplied_by_port: Configuration each canonical port supplies, keyed by port
        identifier.
    :returns: The configuration of every port that supplies or references one.
    :raises ValueError: If a reference does not resolve to exactly one sibling fragment.
    """

    defined_by_port = {
        port_id: supplied_configuration
        for port_id, supplied_configuration in supplied_by_port.items()
        if supplied_configuration is not None and not supplied_configuration.is_reference
    }
    distinct_configurations: list[QbloxSuppliedConfiguration] = []
    for supplied_configuration in defined_by_port.values():
        if supplied_configuration not in distinct_configurations:
            distinct_configurations.append(supplied_configuration)

    resolved_by_port = dict(defined_by_port)
    for port_id, supplied_configuration in supplied_by_port.items():
        if supplied_configuration is None or not supplied_configuration.is_reference:
            continue
        if len(distinct_configurations) != 1:
            raise ValueError(
                f"Qblox port {port_id!r} on module {module_view.location!r} references "
                f"shared configuration, but its module has "
                f"{len(distinct_configurations)} distinct supplied configurations to "
                "resolve it against"
            )
        resolved_by_port[port_id] = distinct_configurations[0]
    return resolved_by_port


def _reconcile(
    merged_values: dict[str, ConfigValue],
    supplied_values: Mapping[str, ConfigValue],
    value_sources: dict[str, str],
    path: str,
    port_id: str,
    module_view: QbloxModuleView,
) -> None:
    """Merge one port's module fragment into the reconciled module-wide values.

    :param merged_values: Reconciled values merged so far, updated in place.
    :param supplied_values: Module values supplied through ``port_id``.
    :param value_sources: Port each already reconciled field came from, updated in place.
    :param path: Dotted path of ``supplied_values`` within the module configuration.
    :param port_id: Canonical port supplying ``supplied_values``.
    :param module_view: The hardware view of the physical module.
    :raises ValueError: If the supplied values disagree with an already reconciled field.
    """

    for key, value in supplied_values.items():
        field_path = f"{path}.{key}" if path else key
        existing = merged_values.get(key)
        if existing is not None and isinstance(existing, Mapping) != isinstance(
            value, Mapping
        ):
            raise ValueError(
                f"Qblox module {module_view.location!r} has conflicting "
                f"{field_path!r} values: one port supplies a group of fields and port "
                f"{port_id!r} does not"
            )
        if isinstance(value, Mapping):
            nested = dict(existing) if isinstance(existing, Mapping) else {}
            _reconcile(
                nested,
                value,
                value_sources=value_sources,
                path=field_path,
                port_id=port_id,
                module_view=module_view,
            )
            merged_values[key] = frozendict(nested)
        elif existing is None:
            merged_values[key] = value
            value_sources[field_path] = port_id
        elif existing != value:
            raise ValueError(
                f"Qblox module {module_view.location!r} has conflicting "
                f"{field_path!r} values: port {value_sources[field_path]!r} supplies "
                f"{existing!r} and port {port_id!r} supplies {value!r}"
            )
