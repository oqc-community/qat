# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Default materialiser for canonical system data.

Reconstitutes a :class:`~qat.experimental.system_data.canonical.schema.CanonicalSystemData`
from a payload produced by
:meth:`~qat.experimental.system_data.materialisers.builder.CanonicalSystemDataBuilder.build_payload`.
Version integrity is verified via the structural hash embedded in the payload.
"""

from __future__ import annotations

from typing import Any

from qat.experimental.system_data.canonical.schema import CanonicalSystemData
from qat.experimental.system_data.materialisers.builder import (
    CanonicalSystemDataBuilder,
    version_structure_hash,
)
from qat.experimental.system_data.materialisers.errors import (
    SourceValidationError,
    UnsupportedSourceVersionError,
)
from qat.experimental.system_data.materialisers.model.validation import validate


def materialise_model(source_payload: dict[str, Any]) -> CanonicalSystemData:
    """Materialise canonical system data from a model source payload.

    The model materialiser is a zero-transform materialiser: it reconstructs a
    :class:`CanonicalSystemData` directly from a payload whose keys and values
    mirror the dataclass fields.  Use :meth:`CanonicalSystemDataBuilder.build_payload`
    to produce a conforming payload.

    :param source_payload: Dict with a ``model`` key containing the
        :class:`CanonicalSystemData` field values and a ``_version`` structural hash
        entry, as produced by :meth:`~CanonicalSystemDataBuilder.build_payload`.
    :returns: A validated :class:`CanonicalSystemData` constructed from
        ``source_payload``.
    :raises UnsupportedSourceVersionError: If the ``_version`` key is absent or does
        not match the current structural hash.
    :raises SourceValidationError: If ``source_payload`` contains keys incompatible
        with the :class:`CanonicalSystemData` constructor.
    """
    # Just drop wrappers and look at the model itself.
    source_payload = source_payload[CanonicalSystemDataBuilder.data_field]
    version = source_payload.pop(CanonicalSystemDataBuilder.versioning_key, None)
    if version is None:
        raise UnsupportedSourceVersionError(
            f"Materialisation requires the versioning hash field "
            f"'{CanonicalSystemDataBuilder.versioning_key}' to be present in the payload."
        )

    if version != version_structure_hash:
        raise UnsupportedSourceVersionError(
            f"Can't materialise: payload was produced with a different model version. "
            f"Incoming: {version}, current: {version_structure_hash}."
        )

    try:
        result = CanonicalSystemData(**source_payload)
    except TypeError as exc:
        raise SourceValidationError(
            f"Model source payload could not construct a CanonicalSystemData instance: {exc}"
        ) from exc
    validate(result)
    return result
