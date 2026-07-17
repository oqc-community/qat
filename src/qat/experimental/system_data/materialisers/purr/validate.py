# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""PuRR ingress validation orchestration entrypoint.

This module provides the compiler-side validation boundary for parsed PuRR ingress
DTOs by orchestrating domain validators from the ``validators`` subpackage.
"""

# NOTE: This module currently hosts the compiler-side PuRR trust boundary.
# TODO: Move these validations server-side and replace them with thin trust validation
# that only checks source verification, signature integrity, and hash/SHA validation.

from qat.experimental.system_data.materialisers.purr.ingress.v0_1_0 import PurrIngressV010
from qat.experimental.system_data.materialisers.purr.validators.common import (
    _collect_qubit_indices,
)
from qat.experimental.system_data.materialisers.purr.validators.couplings import (
    _validate_coupling_indices,
    _validate_cr_crc_auxiliary_targets,
    _validate_cr_crc_channel_mapping_keys,
    _validate_cr_crc_counterparts,
    _validate_cr_crc_matches_coupling_graph,
    _warn_missing_coupling_quality,
)
from qat.experimental.system_data.materialisers.purr.validators.postprocess import (
    _validate_mean_z_map_args,
    _validate_post_process_method,
    _validate_readout_mitigation,
    _warn_extra_mitigation_entries,
)
from qat.experimental.system_data.materialisers.purr.validators.signal_paths import (
    _validate_basebands,
    _validate_passive_reset_time,
    _validate_physical_channels,
    _validate_pulse_channel_frequencies,
    _validate_pulse_channel_references,
    _validate_pulse_channel_scales,
    _validate_repeat_limit,
    _validate_top_level_collections,
    _warn_sample_time_consistency,
)
from qat.experimental.system_data.materialisers.purr.validators.waveforms import (
    _validate_acquire_sync_field,
    _validate_measure_acquire_payloads,
    _validate_waveform_numeric_fields,
    _validate_waveform_payloads,
)


def validate_purr_ingress_graph(dto: PurrIngressV010) -> None:
    """Validate cross-entity relationships for a parsed PuRR ingress DTO.

    :param dto: Validated ingress DTO. This orchestration applies structural,
        channel/baseband, waveform/acquire, post-processing, coupling/CRCRC, and warning-
        level validations before canonical materialisation proceeds.
    """
    _validate_top_level_collections(
        quantum_devices=dto.quantum_devices,
        physical_channels=dto.physical_channels,
        basebands=dto.basebands,
    )
    _validate_repeat_limit(dto.repeat_limit)
    _validate_passive_reset_time(dto.passive_reset_time)

    _validate_physical_channels(dto.physical_channels)
    _validate_basebands(dto.basebands)
    _validate_pulse_channel_references(
        pulse_channels=dto.pulse_channels,
        physical_channel_ids=frozenset(dto.physical_channels.keys()),
    )
    _validate_cr_crc_channel_mapping_keys(dto)
    _validate_cr_crc_counterparts(dto)
    _validate_pulse_channel_frequencies(dto)
    _validate_waveform_payloads(dto)
    _validate_measure_acquire_payloads(dto)
    _validate_mean_z_map_args(dto)
    _validate_post_process_method(dto)
    _validate_pulse_channel_scales(dto)
    _validate_waveform_numeric_fields(dto)
    _validate_acquire_sync_field(dto)
    qubit_indices = _collect_qubit_indices(dto)

    _validate_coupling_indices(dto.qubit_direction_couplings, qubit_indices)
    _validate_cr_crc_matches_coupling_graph(dto)
    _validate_cr_crc_auxiliary_targets(dto)

    _validate_readout_mitigation(dto.error_mitigation, qubit_indices)

    # Warning-level checks: log non-fatal issues that may indicate payload incompleteness
    # or unintended heterogeneity, but do not block canonicalisation.
    _warn_extra_mitigation_entries(dto.error_mitigation, qubit_indices)
    _warn_sample_time_consistency(dto)
    _warn_missing_coupling_quality(dto.qubit_direction_couplings)
