# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""PuRR materialiser plugins for boundary dispatch.

Plugins in this module self-register on import via the shared registry API in
``qat.experimental.system_data.materialisers.plugins``.
"""

from typing import Any

from qat.experimental.system_data.canonical.schema import CanonicalSystemData
from qat.experimental.system_data.materialisers.plugins import (
    SourceAdditionalDataModel,
    register_materialiser_plugin,
)
from qat.experimental.system_data.materialisers.types import SourceType
from qat.model.target_data import TargetData


class PurrV010AdditionalData(SourceAdditionalDataModel):
    """Additional-data contract for PuRR v0.1.0 materialisation."""

    target_data: TargetData | None = None
    supported_acquire_modes: list[str] | None = None
    native_waveform_shapes: list[str] | None = None
    decoder_extra_reduce_target_types: list[str] | None = None
    decoder_extra_reduce_target_suffixes: list[str] | None = None


class PurrV010Plugin:
    """Boundary plugin for PuRR v0.1.0 source payloads."""

    source_type = SourceType.PURR
    source_version = "0.1.0"
    additional_data_model = PurrV010AdditionalData

    @staticmethod
    def _payload_view_for_detection(source_payload: dict[str, Any]) -> dict[str, Any]:
        """Return payload mapping used for lightweight structural detection.

        Historic calibration dumps may wrap the effective model fields in
        ``py/state``; normalised payloads place the same keys at top level.
        """

        py_state = source_payload.get("py/state")
        if isinstance(py_state, dict):
            return py_state
        return source_payload

    def resolve_type_and_version(
        self,
        source_payload: dict[str, Any],
    ) -> tuple[SourceType, str] | None:
        """Resolve PuRR source descriptor from structural payload keys."""

        detection_payload = self._payload_view_for_detection(source_payload)

        required_keys = (
            "quantum_devices",
            "pulse_channels",
            "physical_channels",
            "basebands",
        )
        if all(key in detection_payload for key in required_keys):
            return self.source_type, self.source_version
        return None

    def verify_integrity(self, source_payload: dict) -> None:
        # Placeholder trust verifier until source integrity artefacts are available.
        _ = source_payload

    def materialise(
        self,
        *,
        source_payload: dict,
        source_version: str,
        additional_data: PurrV010AdditionalData,
    ) -> CanonicalSystemData:
        from qat.experimental.system_data.materialisers.purr.materialise import (
            materialise_purr_v0_1_0,
        )

        return materialise_purr_v0_1_0(
            source_payload=source_payload,
            source_version=source_version,
            target_data=additional_data.target_data,
            supported_acquire_modes=additional_data.supported_acquire_modes,
            native_waveform_shapes=additional_data.native_waveform_shapes,
            decoder_extra_reduce_target_types=(
                additional_data.decoder_extra_reduce_target_types
            ),
            decoder_extra_reduce_target_suffixes=(
                additional_data.decoder_extra_reduce_target_suffixes
            ),
        )


register_materialiser_plugin(plugin=PurrV010Plugin())
