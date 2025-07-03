# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd

from dataclasses import dataclass, field
from typing import Dict, List

from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


@dataclass
class ConnectionConfig:
    bulk_value: List[str] = field(default_factory=list)
    out0: str = None  # Enum: {‘I’, ‘Q’, ‘off’}
    out1: str = None  # Enum: {‘I’, ‘Q’, ‘off’}
    out2: str = None  # Enum: {‘I’, ‘Q’, ‘off’}
    out3: str = None  # Enum: {‘I’, ‘Q’, ‘off’}
    acq: str = None  # Enum: {False, ‘in0’, True, ‘off’}
    acq_I: str = None  # Enum: {‘in0’, ‘in1’, ‘off’}
    acq_Q: str = None  # Enum: {‘in0’, ‘in1’, ‘off’}


@dataclass
class NcoConfig:
    freq: float = None
    phase_offs: float = None
    prop_delay_comp: int = None
    prop_delay_comp_en: bool = None


@dataclass
class AwgConfig:
    cont_mode_en_path0: bool = None
    cont_mode_en_path1: bool = None
    cont_mode_waveform_idx_path0: int = None
    cont_mode_waveform_idx_path1: int = None

    upsample_rate_path0: int = None
    upsample_rate_path1: int = None

    gain_path0: float = None
    gain_path1: float = None

    offset_path0: float = None
    offset_path1: float = None

    mod_en: bool = None


@dataclass
class MixerConfig:
    phase_offset: float = None
    gain_ratio: float = None


@dataclass
class SquareWeightAcq:
    integration_length: int = None


@dataclass
class ThresholdedAcqConfig:
    rotation: int = None
    threshold: float = None
    marker_en: bool = None
    marker_address: int = None
    marker_invert: bool = None
    trigger_en: bool = None
    trigger_address: int = None
    trigger_invert: bool = None


@dataclass
class TtlAcqConfig:
    auto_bin_incr_en: bool = None
    threshold: float = None
    input_select: int = None


@dataclass
class SequencerConfig:
    sync_en: bool = None
    marker_ovr_en: bool = None
    marker_ovr_value: int = None

    trigger_count_thresholds: int = field(default_factory=dict)  # key in [0,15]
    trigger_threshold_inverts: bool = field(default_factory=dict)  # key in [0,15]

    connection: ConnectionConfig = field(default_factory=lambda: ConnectionConfig())
    nco: NcoConfig = field(default_factory=lambda: NcoConfig())
    awg: AwgConfig = field(default_factory=lambda: AwgConfig())
    mixer: MixerConfig = field(default_factory=lambda: MixerConfig())

    demod_en_acq: bool = None
    square_weight_acq: SquareWeightAcq = field(default_factory=lambda: SquareWeightAcq())
    thresholded_acq: ThresholdedAcqConfig = field(
        default_factory=lambda: ThresholdedAcqConfig()
    )
    ttl_acq: TtlAcqConfig = field(default_factory=lambda: TtlAcqConfig())


@dataclass
class OffsetConfig:
    out0: float = None
    out1: float = None
    out2: float = None
    out3: float = None

    in1: float = None
    in0: float = None

    out0_path0: float = None
    out0_path1: float = None
    out1_path0: float = None
    out1_path1: float = None

    in0_path0: float = None
    in0_path1: float = None


@dataclass
class LoConfig:
    out0_en: bool = None
    out0_freq: float = None
    out1_en: bool = None
    out1_freq: float = None

    out0_in0_en: bool = None
    out0_in0_freq: float = None


@dataclass
class AttConfig:
    out0: int = None
    out1: int = None
    in0: int = None


@dataclass
class GainConfig:
    in0: int = None
    in1: int = None


@dataclass
class ScopeAcqConfig:
    sequencer_select: int = None
    trigger_mode_path0: str = None  # Enum: {‘sequencer’, ‘level’}
    trigger_mode_path1: str = None  # Enum: {‘sequencer’, ‘level’}
    trigger_level_path0: float = None
    trigger_level_path1: float = None
    avg_mode_en_path0: bool = None
    avg_mode_en_path1: bool = None


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
    slot_idx: int = None
    module: ModuleConfig = field(default_factory=lambda: ModuleConfig())
    sequencers: Dict[int, SequencerConfig] = field(default_factory=dict)
