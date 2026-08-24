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

External packages may use a plain string for ``source_type`` instead of a
``SourceType`` enum value.  This lets third-party plugins register under their own
source identity without requiring changes to the built-in enum.

Plugins that extend or specialise an existing plugin may declare a ``supersedes``
class attribute — a tuple of ``(source_type, source_version)`` descriptors.  When
payload detection matches multiple plugins, any descriptor listed in the winning
plugin's ``supersedes`` set is filtered out before the ambiguity check.  This
allows a more-specific plugin to take precedence over a generic one without
modifying the generic plugin.

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
            source_type = "my-source"  # SourceType enum value or plain string
            source_version = "1.0.0"
            additional_data_model = MySourceAdditionalData
            # Optional: declare precedence over a less-specific plugin.
            supersedes = ((SourceType.PURR, "0.1.0"),)

            def resolve_type_and_version(
                self,
                source_payload: dict[str, Any],
            ) -> tuple[SourceType | str, str] | None:
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

    source_type: SourceType | str
    source_version: str
    additional_data_model: type[SourceAdditionalDataModel]

    def resolve_type_and_version(
        self,
        source_payload: dict[str, Any],
    ) -> tuple[SourceType | str, str] | None:
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


_PluginKey = tuple[SourceType | str, str]
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


def _is_valid_source_type(source_type: object) -> bool:
    """Return whether a source type is an enum or a non-empty string."""
    return isinstance(source_type, SourceType | str) and bool(source_type)


def _is_reserved_source_type(source_type: SourceType | str) -> bool:
    """Return whether a source type belongs to the built-in enum namespace."""
    return isinstance(source_type, str) and any(
        source_type == member.value for member in SourceType
    )


def _is_valid_source_version(source_version: object) -> bool:
    """Return whether a source version is a non-empty string."""
    return isinstance(source_version, str) and bool(source_version)


def _is_valid_source_descriptor(descriptor: object) -> bool:
    """Return whether a source descriptor is a valid type/version tuple."""
    return (
        isinstance(descriptor, tuple)
        and len(descriptor) == 2
        and _is_valid_source_type(descriptor[0])
        and _is_valid_source_version(descriptor[1])
    )


def register_materialiser_plugin(
    *,
    plugin: SourceMaterialiserPlugin,
    replace: bool = False,
) -> None:
    """Register a materialiser plugin for source type/version dispatch."""

    if not _is_valid_source_type(plugin.source_type):
        raise ValueError(
            "plugin.source_type must be a SourceType value or a non-empty string."
        )
    if (
        isinstance(plugin.source_type, str)
        and not isinstance(plugin.source_type, SourceType)
        and _is_reserved_source_type(plugin.source_type)
    ):
        raise ValueError(
            "plugin.source_type string conflicts with a built-in SourceType value."
        )
    if not _is_valid_source_version(plugin.source_version):
        raise ValueError("plugin.source_version must be a non-empty string.")
    if not isinstance(plugin.additional_data_model, type) or not issubclass(
        plugin.additional_data_model, SourceAdditionalDataModel
    ):
        raise ValueError(
            "plugin.additional_data_model must subclass SourceAdditionalDataModel."
        )
    supersedes = getattr(plugin, "supersedes", ())
    if not isinstance(supersedes, tuple):
        raise ValueError("plugin.supersedes must be a tuple of source descriptors.")
    for entry in supersedes:
        if not _is_valid_source_descriptor(entry):
            raise ValueError(
                "plugin.supersedes entries must each be a (SourceType | str, str) 2-tuple."
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


def _source_type_matches(registered: SourceType | str, requested: SourceType | str) -> bool:
    """Return True when ``registered`` and ``requested`` refer to the same source type.

    Accepts either form so callers using a string get the same result as those using an
    enum.
    """
    if registered == requested:
        return True
    # Coerce string to SourceType for backward compatibility with enum-keyed plugins.
    if isinstance(requested, str):
        try:
            return registered == SourceType(requested)
        except ValueError:
            pass
    return False


def get_materialiser_plugin(
    *,
    source_type: SourceType | str,
    source_version: str,
) -> SourceMaterialiserPlugin | None:
    """Return the registered plugin for ``source_type``/``source_version``."""

    plugin = _PLUGIN_REGISTRY.get((source_type, source_version))
    if plugin is not None:
        return plugin
    if isinstance(source_type, str):
        try:
            return _PLUGIN_REGISTRY.get((SourceType(source_type), source_version))
        except ValueError:
            pass
    return None


def get_registered_source_versions(source_type: SourceType | str) -> tuple[str, ...]:
    """Return sorted registered versions for a source type."""

    return tuple(
        sorted(
            version
            for source, version in _PLUGIN_REGISTRY
            if _source_type_matches(source, source_type)
        )
    )


def get_registered_materialiser_plugins(
    source_type: SourceType | str | None = None,
) -> tuple[SourceMaterialiserPlugin, ...]:
    """Return registered plugins in deterministic registry-key order.

    When ``source_type`` is provided, only plugins for that source are returned.
    """

    def _sort_key(item: tuple[_PluginKey, Any]) -> tuple[str, str]:
        (st, version), _ = item
        return (st.value if isinstance(st, SourceType) else st, version)

    return tuple(
        plugin
        for (registered_source, _), plugin in sorted(
            _PLUGIN_REGISTRY.items(), key=_sort_key
        )
        if source_type is None or _source_type_matches(registered_source, source_type)
    )
