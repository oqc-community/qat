# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Shared contracts and registry tooling for source-specific materialiser plugins.

Plugin authors should define two things in their source package:

1. A ``SourceAdditionalDataModel`` subclass describing any extra inputs needed in
    addition to ``source_payload``.
2. A class implementing ``SourceMaterialiserPlugin`` with:
    - ``source_type`` and ``source_version`` identifiers,
    - ``resolve_type_and_version`` detector for payload-based source resolution,
    - ``additional_data_model`` pointing to the schema class,
    - ``verify_integrity`` for source trust/integrity checks,
    - ``materialise`` to build and return ``CanonicalSystemData``.

Plugins are registered by calling ``register_materialiser_plugin`` from this module,
typically as an import-side effect in the plugin module itself.
"""

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from qat.experimental.system_data.canonical.schema import CanonicalSystemData
from qat.experimental.system_data.materialisers.types import SourceType


class SourceAdditionalDataModel(BaseModel):
    """Base schema for source-specific additional-data payloads.

    Subclass this model in each source package to define the typed
    ``source_additional_data`` contract expected by that plugin.

    Design notes:
    - ``extra='forbid'`` rejects unknown keys at the boundary.
    - ``arbitrary_types_allowed=True`` permits rich compiler-owned types (for
      example ``TargetData``) in plugin models.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class SourceMaterialiserPlugin(Protocol):
    """Structural interface for source/version materialiser plugins.

    Minimal implementation pattern:

    .. code-block:: python

        class MySourceAdditionalData(SourceAdditionalDataModel):
            helper: MyHelperType

        class MySourcePlugin:
            source_type = SourceType.MY_SOURCE
            source_version = "1.0.0"
            additional_data_model = MySourceAdditionalData

            def resolve_type_and_version(
                self,
                source_payload: dict[str, Any],
            ) -> tuple[SourceType, str] | None:
                ...

            def verify_integrity(self, source_payload: dict[str, Any]) -> None:
                ...

            def materialise(
                self,
                *,
                source_payload: dict[str, Any],
                source_version: str,
                additional_data: MySourceAdditionalData,
            ) -> CanonicalSystemData:
                ...

    Register the plugin via ``register_materialiser_plugin`` in this module.
    """

    source_type: SourceType
    source_version: str
    additional_data_model: type[SourceAdditionalDataModel]

    def resolve_type_and_version(
        self,
        source_payload: dict[str, Any],
    ) -> tuple[SourceType, str] | None:
        """Return this plugin descriptor when payload matches, else ``None``.

        The detector may use payload metadata and/or structural pattern checks, but it
        should stay lightweight and side-effect free. Avoid full payload validation or
        expensive normalisation in this method.
        """

    def verify_integrity(self, source_payload: dict[str, Any]) -> None:
        """Verify source payload integrity for this source/version.

        Raise a ``MaterialisationError`` subclass for structured failures. Any
        unexpected exception may be wrapped by the boundary as a
        ``SourceIntegrityError``.
        """

    def materialise(
        self,
        *,
        source_payload: dict[str, Any],
        source_version: str,
        additional_data: SourceAdditionalDataModel,
    ) -> CanonicalSystemData:
        """Materialise canonical data for this source/version plugin.

        ``additional_data`` is already validated by
        ``additional_data_model.model_validate`` before this method is called.
        """


_PluginKey = tuple[SourceType, str]
_PLUGIN_REGISTRY: dict[_PluginKey, SourceMaterialiserPlugin] = {}


def _plugin_identity(plugin: SourceMaterialiserPlugin) -> tuple[str, str, str, str]:
    """Return a stable identity tuple for duplicate registration detection."""

    plugin_type = type(plugin)
    model_type = plugin.additional_data_model
    return (
        plugin_type.__module__,
        plugin_type.__qualname__,
        model_type.__module__,
        model_type.__qualname__,
    )


def register_materialiser_plugin(
    *,
    plugin: SourceMaterialiserPlugin,
    replace: bool = False,
) -> None:
    """Register a materialiser plugin for source type/version dispatch."""

    if not isinstance(plugin.source_type, SourceType):
        raise ValueError("plugin.source_type must be a SourceType value.")
    if not isinstance(plugin.source_version, str) or not plugin.source_version:
        raise ValueError("plugin.source_version must be a non-empty string.")
    if not isinstance(plugin.additional_data_model, type) or not issubclass(
        plugin.additional_data_model, SourceAdditionalDataModel
    ):
        raise ValueError(
            "plugin.additional_data_model must subclass SourceAdditionalDataModel."
        )

    key = (plugin.source_type, plugin.source_version)
    if key in _PLUGIN_REGISTRY and not replace:
        existing_plugin = _PLUGIN_REGISTRY[key]
        # Re-importing the same plugin module should be a no-op.
        if _plugin_identity(existing_plugin) == _plugin_identity(plugin):
            return
        raise ValueError(
            "materialiser plugin already registered for source/version; "
            "set replace=True to overwrite."
        )
    _PLUGIN_REGISTRY[key] = plugin


def get_materialiser_plugin(
    *,
    source_type: SourceType | str,
    source_version: str,
) -> SourceMaterialiserPlugin | None:
    """Return the registered plugin for ``source_type``/``source_version``."""

    try:
        resolved_source = SourceType(source_type)
    except ValueError:
        return None

    return _PLUGIN_REGISTRY.get((resolved_source, source_version))


def get_registered_source_versions(source_type: SourceType) -> tuple[str, ...]:
    """Return sorted registered versions for a source type."""

    return tuple(
        sorted(version for source, version in _PLUGIN_REGISTRY if source == source_type)
    )


def get_registered_materialiser_plugins(
    source_type: SourceType | None = None,
) -> tuple[SourceMaterialiserPlugin, ...]:
    """Return registered plugins in deterministic registry-key order.

    When ``source_type`` is provided, only plugins for that source are returned.
    """

    return tuple(
        plugin
        for (registered_source, _), plugin in sorted(_PLUGIN_REGISTRY.items())
        if source_type is None or registered_source == source_type
    )
