from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

from qblox_instruments.qcodes_drivers.module import Module
from qblox_instruments.qcodes_drivers.sequencer import Sequencer

from qat.purr.compiler.devices import PulseChannel


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

    connection: ConnectionConfig = ConnectionConfig()
    nco: NcoConfig = NcoConfig()
    awg: AwgConfig = AwgConfig()
    mixer: MixerConfig = MixerConfig()

    demod_en_acq: bool = None
    square_weight_acq: SquareWeightAcq = SquareWeightAcq()
    thresholded_acq: ThresholdedAcqConfig = ThresholdedAcqConfig()
    ttl_acq: TtlAcqConfig = TtlAcqConfig()


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

    offset: OffsetConfig = OffsetConfig()
    lo: LoConfig = LoConfig()
    attenuation: AttConfig = AttConfig()
    gain: GainConfig = GainConfig()
    scope_acq: ScopeAcqConfig = ScopeAcqConfig()


@dataclass
class QbloxConfig:
    slot_idx: int = None
    module: ModuleConfig = ModuleConfig()
    sequencers: Dict[int, SequencerConfig] = field(default_factory=dict)


class QbloxConfigHelper(ABC):
    def __init__(self, target: PulseChannel):
        self.target = target

    def configure(self, module: Module, sequencer: Sequencer):
        config: QbloxConfig = self.target.physical_channel.config
        self.clear(module, sequencer)
        self.configure_module(module, config.module)
        self.configure_sequencer(sequencer, config.sequencers[sequencer.seq_idx])

    def clear(self, module: Module, sequencer: Sequencer):
        module.disconnect_outputs()

    @abstractmethod
    def configure_module(self, module: Module, config: ModuleConfig):
        pass

    @abstractmethod
    def configure_sequencer(self, sequencer: Sequencer, config: SequencerConfig):
        pass

    def configure_connection(self, sequencer: Sequencer, config: SequencerConfig):
        if config.connection:
            sequencer.connect_sequencer(*config.connection.bulk_value)

    def configure_nco(self, sequencer: Sequencer, config: SequencerConfig):
        physical_channel = self.target.physical_channel
        bb_if_frequency = physical_channel.baseband_if_frequency
        sequencer.nco_freq(bb_if_frequency)
        if config.nco.prop_delay_comp_en:
            sequencer.nco_prop_delay_comp_en(config.nco.prop_delay_comp_en)
        if config.nco.prop_delay_comp:
            sequencer.nco_prop_delay_comp(config.nco.prop_delay_comp)
        if config.nco.phase_offs:
            sequencer.nco_phase_offs(config.nco.phase_offs)

    def configure_awg(self, sequencer: Sequencer, config: SequencerConfig):
        if config.awg.mod_en:
            sequencer.mod_en_awg(config.awg.mod_en)
        if config.awg.gain_path0:
            sequencer.gain_awg_path0(config.awg.gain_path0)
        if config.awg.gain_path1:
            sequencer.gain_awg_path1(config.awg.gain_path1)
        if config.awg.offset_path0:
            sequencer.offset_awg_path0(config.awg.offset_path0)
        if config.awg.offset_path1:
            sequencer.offset_awg_path1(config.awg.offset_path1)

    def configure_mixer(self, sequencer: Sequencer, config: SequencerConfig):
        if config.mixer.phase_offset:
            sequencer.mixer_corr_phase_offset_degree(config.mixer.phase_offset)
        if config.mixer.gain_ratio:
            sequencer.mixer_corr_gain_ratio(config.mixer.gain_ratio)


class QcmConfigHelper(QbloxConfigHelper):
    pass


class QcmRfConfigHelper(QcmConfigHelper):
    def configure_module(self, module, config):
        self.configure_lo(module, config)
        self.configure_attenuation(module, config)
        self.configure_offset(module, config)

    def configure_sequencer(self, sequencer, config):
        if config.sync_en:
            sequencer.sync_en(config.sync_en)
        self.configure_connection(sequencer, config)
        self.configure_nco(sequencer, config)
        self.configure_awg(sequencer, config)
        self.configure_mixer(sequencer, config)

    def configure_lo(self, module, config):
        physical_channel = self.target.physical_channel
        bb_if_frequency = physical_channel.baseband_if_frequency
        bb_frequency = self.target.frequency - bb_if_frequency  # TMP
        if config.lo.out0_en:
            module.out0_lo_en(config.lo.out0_en)  # Switch the LO 0 on
            module.out0_lo_freq(bb_frequency)
        if config.lo.out1_en:
            module.out1_lo_en(config.lo.out1_en)  # Switch the LO 1 on
            module.out1_lo_freq(bb_frequency)

    def configure_attenuation(self, module, config):
        if config.attenuation.out0:
            module.out0_att(config.attenuation.out0)
        if config.attenuation.out1:
            module.out1_att(config.attenuation.out1)

    def configure_offset(self, module, config):
        if config.offset.out0_path0:
            module.out0_offset_path0(config.offset.out0_path0)
        if config.offset.out0_path1:
            module.out0_offset_path1(config.offset.out0_path1)
        if config.offset.out1_path0:
            module.out1_offset_path0(config.offset.out1_path0)
        if config.offset.out1_path1:
            module.out1_offset_path1(config.offset.out1_path1)


class QrmConfigHelper(QbloxConfigHelper):
    def configure(self, module, sequencer):
        super().configure(module, sequencer)

        # TODO - make these settings dynamic
        module.scope_acq_sequencer_select(sequencer.seq_idx)
        module.scope_acq_trigger_mode_path0("sequencer")
        module.scope_acq_trigger_mode_path1("sequencer")

    def clear(self, module: Module, sequencer: Sequencer):
        super().clear(module, sequencer)
        module.disconnect_inputs()

    def configure_scope_acq(self, module, config):
        scope_acq = config.scope_acq
        if scope_acq.avg_mode_en_path0:
            module.scope_acq_avg_mode_en_path0(scope_acq.avg_mode_en_path0)
        if scope_acq.avg_mode_en_path1:
            module.scope_acq_avg_mode_en_path1(scope_acq.avg_mode_en_path1)


class QrmRfConfigHelper(QrmConfigHelper):
    def configure_module(self, module, config):
        self.configure_lo(module, config)
        self.configure_attenuation(module, config)
        self.configure_offset(module, config)
        self.configure_scope_acq(module, config)

    def configure_sequencer(self, sequencer, config):
        if config.sync_en:
            sequencer.sync_en(config.sync_en)
        self.configure_connection(sequencer, config)
        self.configure_nco(sequencer, config)
        self.configure_awg(sequencer, config)
        self.configure_mixer(sequencer, config)
        self.configure_acq(sequencer, config)

    def configure_lo(self, module, config):
        physical_channel = self.target.physical_channel
        bb_if_frequency = physical_channel.baseband_if_frequency
        bb_frequency = self.target.frequency - bb_if_frequency  # TMP
        if config.lo.out0_in0_en:
            module.out0_in0_lo_en(config.lo.out0_in0_en)  # Switch the LO on
            module.out0_in0_lo_freq(bb_frequency)

    def configure_attenuation(self, module, config):
        if config.attenuation.out0:
            module.out0_att(config.attenuation.out0)
        if config.attenuation.in0:
            module.in0_att(config.attenuation.in0)

    def configure_offset(self, module, config):
        if config.offset.out0_path0:
            module.out0_offset_path0(config.offset.out0_path0)
        if config.offset.out0_path1:
            module.out0_offset_path1(config.offset.out0_path1)

    def configure_acq(self, sequencer, config):
        # Square weight integration
        if config.demod_en_acq:
            sequencer.demod_en_acq(config.demod_en_acq)
        if config.square_weight_acq.integration_length:
            sequencer.integration_length_acq(config.square_weight_acq.integration_length)
