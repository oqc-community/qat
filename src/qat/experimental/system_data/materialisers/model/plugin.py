# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Default materialiser plugin for boundary dispatch.

This module self-registers on import via the shared registry API in
``qat.experimental.system_data.materialisers.plugins``.

Model payloads embed a ``_version`` key produced by
:meth:`~qat.experimental.system_data.materialisers.builder.CanonicalSystemDataBuilder.build_payload`
to allow unambiguous detection at the boundary.
"""

from __future__ import annotations

from typing import Any

from qat.experimental.system_data.canonical.schema import CanonicalSystemData
from qat.experimental.system_data.materialisers.builder import (
    CanonicalSystemDataBuilder,
    version_structure_hash,
)
from qat.experimental.system_data.materialisers.plugins import (
    SourceAdditionalDataModel,
    SourceMaterialiserPlugin,
    register_materialiser_plugin,
)
from qat.experimental.system_data.materialisers.types import SourceType


class DefaultAdditionalData(SourceAdditionalDataModel):
    """Additional-data contract for default v0.1.0 materialisation.

    Default payloads require no extra inputs beyond the source payload itself, so this model
    carries no fields.
    """


class DefaultPlugin(SourceMaterialiserPlugin):
    """Boundary plugin for default v0.1.0 source payloads."""

    source_type = SourceType.MODEL
    source_version = "0.1.0"
    additional_data_model = DefaultAdditionalData

    def resolve_type_and_version(
        self,
        source_payload: dict[str, Any],
    ) -> tuple[SourceType, str] | None:
        """Detect a default source payload.

        A payload is identified as default v0.1.0 when its ``model`` sub-dict contains
        a ``_version`` key matching the current structural hash (``version_structure_hash``).

        :param source_payload: Raw source payload dict.
        :returns: ``(SourceType.MODEL, "0.1.0")`` if matched, otherwise ``None``.
        """
        if not isinstance(source_payload, dict):
            return None

        model = source_payload.get(CanonicalSystemDataBuilder.data_field)
        if not isinstance(model, dict):
            return None

        if model.get(CanonicalSystemDataBuilder.versioning_key) != version_structure_hash:
            return None

        return self.source_type, self.source_version

    def verify_integrity(self, source_payload: dict[str, Any]):
        """No additional integrity checks required for default payloads."""
        ...

    def materialise(
        self,
        *,
        source_payload: dict[str, Any],
        source_version: str,
        additional_data: DefaultAdditionalData,
    ) -> CanonicalSystemData:
        """Materialise canonical data from a default source payload.

        :param source_payload: Dict with a ``model`` key containing the
            ``CanonicalSystemData`` field values and a ``_version`` structural hash
            entry, as produced by :meth:`CanonicalSystemDataBuilder.build_payload`.
        :returns: Reconstructed ``CanonicalSystemData``.
        """
        from qat.experimental.system_data.materialisers.model.materialise import (
            materialise_model,
        )

        return materialise_model(source_payload)


register_materialiser_plugin(plugin=DefaultPlugin())
