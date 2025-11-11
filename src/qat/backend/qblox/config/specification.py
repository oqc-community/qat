# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd

from dataclasses import dataclass, field

from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


@dataclass
class ConnectionConfig:
    bulk_value: list[str] = field(default_factory=list)
    out0: str | None = None  # Enum: {‘I’, ‘Q’, ‘off’}
    out1: str | None = None  # Enum: {‘I’, ‘Q’, ‘off’}
    out2: str | None = None  # Enum: {‘I’, ‘Q’, ‘off’}
    out3: str | None = None  # Enum: {‘I’, ‘Q’, ‘off’}
    acq: str | None = None  # Enum: {False, ‘in0’, True, ‘off’}
    acq_I: str | None = None  # Enum: {‘in0’, ‘in1’, ‘off’}
    acq_Q: str | None = None  # Enum: {‘in0’, ‘in1’, ‘off’}


@dataclass
class NcoConfig:
    freq: float | None = None
    phase_offs: float | None = None
    prop_delay_comp: int | None = None
    prop_delay_comp_en: bool | None = None


@dataclass
class AwgConfig:
    cont_mode_en_path0: bool | None = None
    cont_mode_en_path1: bool | None = None
    cont_mode_waveform_idx_path0: int | None = None
    cont_mode_waveform_idx_path1: int | None = None

    upsample_rate_path0: int | None = None
    upsample_rate_path1: int | None = None

    gain_path0: float | None = None
    gain_path1: float | None = None

    offset_path0: float | None = None
    offset_path1: float | None = None

    mod_en: bool | None = None


@dataclass
class MixerConfig:
    phase_offset: float | None = None
    gain_ratio: float | None = None


@dataclass
class SquareWeightAcq:
    integration_length: int | None = None


@dataclass
class ThresholdedAcqConfig:
    rotation: float | None = None
    threshold: float | None = None
    marker_en: bool | None = None
    marker_address: int | None = None
    marker_invert: bool | None = None
    trigger_en: bool | None = None
    trigger_address: int | None = None
    trigger_invert: bool | None = None


@dataclass
class TtlAcqConfig:
    auto_bin_incr_en: bool | None = None
    threshold: float | None = None
    input_select: int | None = None


@dataclass
class SequencerConfig:
    sync_en: bool | None = None
    marker_ovr_en: bool | None = None
    marker_ovr_value: int | None = None

    trigger_count_thresholds: dict[int, float] = field(
        default_factory=dict
    )  # key in [0,15]
    trigger_threshold_inverts: dict[int, bool] = field(
        default_factory=dict
    )  # key in [0,15]

    connection: ConnectionConfig = field(default_factory=lambda: ConnectionConfig())
    nco: NcoConfig = field(default_factory=lambda: NcoConfig())
    awg: AwgConfig = field(default_factory=lambda: AwgConfig())
    mixer: MixerConfig = field(default_factory=lambda: MixerConfig())

    demod_en_acq: bool | None = None
    square_weight_acq: SquareWeightAcq = field(default_factory=lambda: SquareWeightAcq())
    thresholded_acq: ThresholdedAcqConfig = field(
        default_factory=lambda: ThresholdedAcqConfig()
    )
    ttl_acq: TtlAcqConfig = field(default_factory=lambda: TtlAcqConfig())


@dataclass
class OffsetConfig:
    out0: float | None = None
    out1: float | None = None
    out2: float | None = None
    out3: float | None = None

    in1: float | None = None
    in0: float | None = None

    out0_path0: float | None = None
    out0_path1: float | None = None
    out1_path0: float | None = None
    out1_path1: float | None = None

    in0_path0: float | None = None
    in0_path1: float | None = None


@dataclass
class LoConfig:
    out0_en: bool | None = None
    out0_freq: float | None = None
    out1_en: bool | None = None
    out1_freq: float | None = None

    out0_in0_en: bool | None = None
    out0_in0_freq: float | None = None


@dataclass
class AttConfig:
    out0: int | None = None
    out1: int | None = None
    in0: int | None = None


@dataclass
class GainConfig:
    in0: int | None = None
    in1: int | None = None


@dataclass
class ScopeAcqConfig:
    sequencer_select: int | None = None
    trigger_mode_path0: str | None = None  # Enum: {‘sequencer’, ‘level’}
    trigger_mode_path1: str | None = None  # Enum: {‘sequencer’, ‘level’}
    trigger_level_path0: float | None = None
    trigger_level_path1: float | None = None
    avg_mode_en_path0: bool | None = None
    avg_mode_en_path1: bool | None = None


@dataclass
class ModuleConfig:
    marker_inverts: dict[int, bool] = field(default_factory=dict)

    offset: OffsetConfig = field(default_factory=lambda: OffsetConfig())
    lo: LoConfig = field(default_factory=lambda: LoConfig())
    attenuation: AttConfig = field(default_factory=lambda: AttConfig())
    gain: GainConfig = field(default_factory=lambda: GainConfig())
    scope_acq: ScopeAcqConfig = field(default_factory=lambda: ScopeAcqConfig())


@dataclass
class QbloxConfig:
    slot_idx: int | None = None
    module: ModuleConfig = field(default_factory=lambda: ModuleConfig())
    sequencers: dict[int, SequencerConfig] = field(default_factory=dict)
