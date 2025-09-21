# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


@dataclass
class ConnectionConfig:
    bulk_value: List[str] = field(default_factory=list)
    out0: Optional[str] = None  # Enum: {‘I’, ‘Q’, ‘off’}
    out1: Optional[str] = None  # Enum: {‘I’, ‘Q’, ‘off’}
    out2: Optional[str] = None  # Enum: {‘I’, ‘Q’, ‘off’}
    out3: Optional[str] = None  # Enum: {‘I’, ‘Q’, ‘off’}
    acq: Optional[str] = None  # Enum: {False, ‘in0’, True, ‘off’}
    acq_I: Optional[str] = None  # Enum: {‘in0’, ‘in1’, ‘off’}
    acq_Q: Optional[str] = None  # Enum: {‘in0’, ‘in1’, ‘off’}


@dataclass
class NcoConfig:
    freq: Optional[float] = None
    phase_offs: Optional[float] = None
    prop_delay_comp: Optional[int] = None
    prop_delay_comp_en: Optional[bool] = None


@dataclass
class AwgConfig:
    cont_mode_en_path0: Optional[bool] = None
    cont_mode_en_path1: Optional[bool] = None
    cont_mode_waveform_idx_path0: Optional[int] = None
    cont_mode_waveform_idx_path1: Optional[int] = None

    upsample_rate_path0: Optional[int] = None
    upsample_rate_path1: Optional[int] = None

    gain_path0: Optional[float] = None
    gain_path1: Optional[float] = None

    offset_path0: Optional[float] = None
    offset_path1: Optional[float] = None

    mod_en: Optional[bool] = None


@dataclass
class MixerConfig:
    phase_offset: Optional[float] = None
    gain_ratio: Optional[float] = None


@dataclass
class SquareWeightAcq:
    integration_length: Optional[int] = None


@dataclass
class ThresholdedAcqConfig:
    rotation: Optional[float] = None
    threshold: Optional[float] = None
    marker_en: Optional[bool] = None
    marker_address: Optional[int] = None
    marker_invert: Optional[bool] = None
    trigger_en: Optional[bool] = None
    trigger_address: Optional[int] = None
    trigger_invert: Optional[bool] = None


@dataclass
class TtlAcqConfig:
    auto_bin_incr_en: Optional[bool] = None
    threshold: Optional[float] = None
    input_select: Optional[int] = None


@dataclass
class SequencerConfig:
    sync_en: Optional[bool] = None
    marker_ovr_en: Optional[bool] = None
    marker_ovr_value: Optional[int] = None

    trigger_count_thresholds: Dict[int, float] = field(
        default_factory=dict
    )  # key in [0,15]
    trigger_threshold_inverts: Dict[int, bool] = field(
        default_factory=dict
    )  # key in [0,15]

    connection: ConnectionConfig = field(default_factory=lambda: ConnectionConfig())
    nco: NcoConfig = field(default_factory=lambda: NcoConfig())
    awg: AwgConfig = field(default_factory=lambda: AwgConfig())
    mixer: MixerConfig = field(default_factory=lambda: MixerConfig())

    demod_en_acq: Optional[bool] = None
    square_weight_acq: SquareWeightAcq = field(default_factory=lambda: SquareWeightAcq())
    thresholded_acq: ThresholdedAcqConfig = field(
        default_factory=lambda: ThresholdedAcqConfig()
    )
    ttl_acq: TtlAcqConfig = field(default_factory=lambda: TtlAcqConfig())


@dataclass
class OffsetConfig:
    out0: Optional[float] = None
    out1: Optional[float] = None
    out2: Optional[float] = None
    out3: Optional[float] = None

    in1: Optional[float] = None
    in0: Optional[float] = None

    out0_path0: Optional[float] = None
    out0_path1: Optional[float] = None
    out1_path0: Optional[float] = None
    out1_path1: Optional[float] = None

    in0_path0: Optional[float] = None
    in0_path1: Optional[float] = None


@dataclass
class LoConfig:
    out0_en: Optional[bool] = None
    out0_freq: Optional[float] = None
    out1_en: Optional[bool] = None
    out1_freq: Optional[float] = None

    out0_in0_en: Optional[bool] = None
    out0_in0_freq: Optional[float] = None


@dataclass
class AttConfig:
    out0: Optional[int] = None
    out1: Optional[int] = None
    in0: Optional[int] = None


@dataclass
class GainConfig:
    in0: Optional[int] = None
    in1: Optional[int] = None


@dataclass
class ScopeAcqConfig:
    sequencer_select: Optional[int] = None
    trigger_mode_path0: Optional[str] = None  # Enum: {‘sequencer’, ‘level’}
    trigger_mode_path1: Optional[str] = None  # Enum: {‘sequencer’, ‘level’}
    trigger_level_path0: Optional[float] = None
    trigger_level_path1: Optional[float] = None
    avg_mode_en_path0: Optional[bool] = None
    avg_mode_en_path1: Optional[bool] = None


@dataclass
class ModuleConfig:
    marker_inverts: Dict[int, bool] = field(default_factory=dict)

    offset: OffsetConfig = field(default_factory=lambda: OffsetConfig())
    lo: LoConfig = field(default_factory=lambda: LoConfig())
    attenuation: AttConfig = field(default_factory=lambda: AttConfig())
    gain: GainConfig = field(default_factory=lambda: GainConfig())
    scope_acq: ScopeAcqConfig = field(default_factory=lambda: ScopeAcqConfig())


@dataclass
class QbloxConfig:
    slot_idx: Optional[int] = None
    module: ModuleConfig = field(default_factory=lambda: ModuleConfig())
    sequencers: Dict[int, SequencerConfig] = field(default_factory=dict)
