# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd

from dataclasses import dataclass, field

from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


@dataclass
class ConnectionConfig:
    """Configuration for the sequencer's connection to the analogue input/output paths.

    :param bulk_value: A list of strings in the format <direction><channel> or
        <direction><I-channel>_<Q-channel>: <direction> must be 'in' to make a connection
        between an input and the acquisition path, 'out' to make a connection from the
        waveform generator to an output, or 'io' to do both. <channel> must be integer
        channel indices. If only one channel is specified,the sequencer operates in real
        mode; if two channels are specified, it operates in complex mode.
    :param out0: Component config of a sequencer's connection to output 0, if any. Possible
        values are 'I', 'Q', 'IQ', or 'off'
    :param out1: Component config of a sequencer's connection to output 1, if any. Possible
        values are 'I', 'Q', 'IQ', or 'off'
    :param out2: Component config of a sequencer's connection to output 2, if any. Possible
        values are 'I', 'Q', 'IQ', or 'off'
    :param out3: Component config of a sequencer's connection to output 3, if any. Possible
        values are 'I', 'Q', 'IQ', or 'off'
    :param out4: Component config of a sequencer's connection to output 4, if any. Possible
        values are 'I', 'Q', 'IQ', or 'off'
    :param out5: Component config of a sequencer's connection to output 5, if any. Possible
        values are 'I', 'Q', 'IQ', or 'off'
    :param acq_I: Input config for the 'I' input of the acquisition path of this sequencer
        is connected to, if any. Possible values are 'in0', 'in1', or 'off'
    :param acq_Q: Input config for the 'Q' input of the acquisition path of this sequencer
        is connected to, if any. Possible values are 'in0', 'in1', or 'off'
    """

    bulk_value: list[str] = field(default_factory=list)
    out0: str | None = None
    out1: str | None = None
    out2: str | None = None
    out3: str | None = None
    out4: str | None = None
    out5: str | None = None

    acq_I: str | None = None
    acq_Q: str | None = None


@dataclass
class NcoConfig:
    """Configuration components related to the sequencer's NCO.

    :param freq: NCO frequency in Hz.
    :param phase_offs: Phase offset of the NCO in degrees with a resolution of 3.6e-7
        degrees.
    :param prop_delay_comp: Delay that compensates the NCO phase to the input path with
        respect to the instrument’s combined output and input propagation delay. This delays
        the frequency update as well.
    :param prop_delay_comp_en: Flag to enable/disable compensation of propagation delay.
    """

    freq: float | None = None
    phase_offs: float | None = None
    prop_delay_comp: int | None = None
    prop_delay_comp_en: bool | None = None


@dataclass
class AwgConfig:
    """Configuration components related to the sequencer's AWG.

    :param cont_mode_en_path0: Flag to enable/disable continuous waveform mode enable path 0 (I).
    :param cont_mode_en_path1: Flag to enable/disable continuous waveform mode enable path 1 (Q).
    :param cont_mode_waveform_idx_path0: Waveform index to play continuously on AWG path 0
                                         (if enabled, see :param:`cont_mode_en_path0`)
    :param cont_mode_waveform_idx_path1: Waveform index to play continuously on AWG path 1
                                         (if enabled, see :param:`cont_mode_en_path1`)
    :param upsample_rate_path0: Upsample rate for AWG path 0.
    :param upsample_rate_path1: Upsample rate for AWG path 1.
    :param gain_path0: Gain for AWG path 0.
    :param gain_path1: Gain for AWG path 1.
    :param offset_path0: Offset for AWG path 0.
    :param offset_path1: Offset for AWG path 1.
    :param mod_en: Flag to enable/disable modulation for AWG.
    """

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
    """Configuration related to the sequencer's mixer correction component.

    :param phase_offset: Mixer phase imbalance correction for AWG.
    :param gain_ratio: Mixer gain imbalance correction for AWG.
    """

    phase_offset: float | None = None
    gain_ratio: float | None = None


@dataclass
class SquareWeightAcq:
    """Configuration components for non-weighed acquisition.

    :param integration_length: Integration length in number of samples for non-weighed
        acquisitions on paths 0 and 1. Must be a multiple of 4. Default value is 1024.
    """

    integration_length: int | None = None


@dataclass
class ThresholdedAcqConfig:
    """Configuration components for thresholded acquisition.

    :param rotation:  Phase rotation (in degrees) for the integration result.
    :param threshold: Threshold for discretizing the phase-rotated result
                      (see :param:`rotation`). Discretization is done by comparing
                      the threshold to the rotated integration result of path 0.
                      This comparison is applied before normalization (i.e. division)
                      of the rotated value with the integration length and therefore
                      the threshold needs to be compensated (i.e. multiplied) with
                      this length for the discretization to function properly.
    :param marker_en: Flag to enable/disable mapping of thresholded acquisition result
                      to markers.
    :param marker_address: Marker mask which maps the thresholded acquisition result
                      to the markers (M1 to M4).
    :param marker_invert: Inversion of the thresholded acquisition result before
                      it is masked onto the markers.
    :param trigger_en: Flag to enable/disable mapping of the thresholded acquisition
                       result to trigger network.
    :param trigger_address: Trigger address to which the thresholded acquisition result
                            is mapped to the trigger network (T1 to T15)
    :param trigger_invert: Inversion of the thresholded acquisition result before it is
                            mapped to the trigger network.
    """

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
    """Configuration components for Transistor-Transistor-Logic acquisition.

    :param auto_bin_incr_en: Flag to enable/disable whether the bin index is automatically
        incremented when acquiring multiple triggers. Disabling the TTL trigger acquisition
        path resets the bin index.
    :param threshold: Threshold value with which to compare the input ADC values of the
        selected input path.
    :param input_select: The input used to compare against the threshold value in the TTL
        trigger acquisition path.
    """

    auto_bin_incr_en: bool | None = None
    threshold: float | None = None
    input_select: int | None = None


@dataclass
class SequencerConfig:
    """Configuration specification for sequencer (the digital side of the RF chain).

    :param sync_en: Flag to enable/disable party-line synchronization. If enabled,
                    the sequencer is "registered" in the SYNC protocol to coordinate
                    the timeline with other sequencers using the `wait_sync` instruction
    :param marker_ovr_en: Flag to enable/disable marker overriding feature. It has priority
                          and will overwrite `set_mrk` instruction.
    :param marker_ovr_value: Marker override value. Its binary representation codifies
                             On/Off flags for marker channels. It has priority and will
                             overwrite `set_mrk` instruction.
    :param trigger_count_thresholds: Threshold map for counters on trigger addresses 0-15.
                                     Thresholding condition used: greater than or equal.
    :param trigger_threshold_inverts: Comparison result inversion for trigger
                                      addresses 0-15.
    :param connection: Sequencer connection config, see :class:`ConnectionConfig`.
    :param nco: NCO config, see :class:`NcoConfig`.
    :param awg: AWG config, see :class:`AwgConfig`.
    :param mixer: Mixer config, see :class:`MixerConfig`.
    :param demod_en_acq: Flag to enable/disable demodulation on the acquisition path.
    :param square_weight_acq: Unweighed acquisition config, see :class:`SquareWeightAcq`.
    :param thresholded_acq: Thresholded acquisition config, see :class:`ThresholdedAcqConfig`.
    :param ttl_acq: TTL acquisition config, see :class:`TtlAcqConfig`.
    """

    sync_en: bool | None = None
    marker_ovr_en: bool | None = None
    marker_ovr_value: int | None = None

    trigger_count_thresholds: dict[int, float] = field(default_factory=dict)
    trigger_threshold_inverts: dict[int, bool] = field(default_factory=dict)

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
    """Configuration components to apply on the input/output the signal. They are DC voltage
    levels can be used to shift the baseline the waveforms or to calibrate out hardware
    imperfections such as mixer leakage.

    :param out0: Offset (in V) for output 0 (I) in QCM/QRM.
    :param out1: Offset (in V) for output 1 (Q) in QCM/QRM.
    :param out2: Offset (in V) for output 2 (I) in QCM.
    :param out3: Offset (in V) for output 3 (Q) in QCM.
    :param in0: Offset (in V) for input 0 (I) in QRM.
    :param in1: Offset (in V) for input 1 (Q) in QRM.
    :param out0_path0: Offset (in mV) for output 0 path 0 (I) in QCM-RF/QRM-RF.
    :param out0_path1: Offset (in mV) for output 0 path 1 (Q) in QCM-RF/QRM-RF.
    :param out1_path0: Offset (in mV) for output 1 path 0 (I) in QCM-RF.
    :param out1_path1: Offset (in mV) for output 1 path 1 (Q) in QCM-RF.
    :param in0_path0: Offset (in V) for input 0 path (I) in QRM-RF.
    :param in0_path1: Offset (in V) for input 1 path (Q) in QRM-RF.
    """

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
class FirConfig:
    """Configuration of the Finite Impulse Response filter. Possible values for the
    outputs/markers are 'bypassed' where the filter is disabled, or 'delay_comp' where the
    filter is bypassed and the output is delayed as if it were applied.

    :param out0: Configuration for the FIR filter for output 0.
    :param out1: Configuration for the FIR filter for output 1.
    :param out2: Configuration for the FIR filter for output 2.
    :param out3: Configuration for the FIR filter for output 3.
    :param out4: Configuration for the FIR filter for output 4.
    :param out5: Configuration for the FIR filter for output 5.
    :param marker0: Configuration for the FIR filter for marker 0.
    """

    out0: str | None = None
    out1: str | None = None
    out2: str | None = None
    out3: str | None = None
    out4: str | None = None
    out5: str | None = None

    marker0: str | None = None


@dataclass
class ExpOvershoot0Config:
    """Configuration of exponential overshoot filter 0. Possible values for the
    outputs/markers are 'bypassed' where the filter is disabled, or 'delay_comp' where the
    filter is bypassed and the output is delayed as if it were applied.

    :param out0: Configuration of exponential overshoot filter 0 for output 0.
    :param out1: Configuration of exponential overshoot filter 0 for output 1.
    :param out2: Configuration of exponential overshoot filter 0 for output 2.
    :param out3: Configuration of exponential overshoot filter 0 for output 3.
    :param out4: Configuration of exponential overshoot filter 0 for output 4.
    :param out5: Configuration of exponential overshoot filter 0 for output 5.
    :param marker0: Configuration of exponential overshoot filter 0 for marker 0.
    """

    out0: str | None = None
    out1: str | None = None
    out2: str | None = None
    out3: str | None = None
    out4: str | None = None
    out5: str | None = None

    marker0: str | None = None


@dataclass
class ExpOvershoot1Config:
    """Configuration of exponential overshoot filter 1. Possible values for the
    outputs/markers are 'bypassed' where the filter is disabled, or 'delay_comp' where the
    filter is bypassed and the output is delayed as if it were applied.

    :param out0: Configuration of exponential overshoot filter 1 for output 0.
    :param out1: Configuration of exponential overshoot filter 1 for output 1.
    :param out2: Configuration of exponential overshoot filter 1 for output 2.
    :param out3: Configuration of exponential overshoot filter 1 for output 3.
    :param out4: Configuration of exponential overshoot filter 1 for output 4.
    :param out5: Configuration of exponential overshoot filter 1 for output 5.
    :param marker0: Configuration of exponential overshoot filter 1 for marker 0.
    """

    out0: str | None = None
    out1: str | None = None
    out2: str | None = None
    out3: str | None = None
    out4: str | None = None
    out5: str | None = None

    marker0: str | None = None


@dataclass
class ExpOvershoot2Config:
    """Configuration of exponential overshoot filter 2. Possible values for the
    outputs/markers are 'bypassed' where the filter is disabled, or 'delay_comp' where the
    filter is bypassed and the output is delayed as if it were applied.

    :param out0: Configuration of exponential overshoot filter 2 for output 0.
    :param out1: Configuration of exponential overshoot filter 2 for output 1.
    :param out2: Configuration of exponential overshoot filter 2 for output 2.
    :param out3: Configuration of exponential overshoot filter 2 for output 3.
    :param out4: Configuration of exponential overshoot filter 2 for output 4.
    :param out5: Configuration of exponential overshoot filter 2 for output 5.
    :param marker0: Configuration of exponential overshoot filter 2 for marker 0.
    """

    out0: str | None = None
    out1: str | None = None
    out2: str | None = None
    out3: str | None = None
    out4: str | None = None
    out5: str | None = None

    marker0: str | None = None


@dataclass
class ExpOvershoot3Config:
    """Configuration of exponential overshoot filter 3. Possible values for the
    outputs/markers are 'bypassed' where the filter is disabled, or 'delay_comp' where the
    filter is bypassed and the output is delayed as if it were applied.

    :param out0: Configuration of exponential overshoot filter 3 for output 0.
    :param out1: Configuration of exponential overshoot filter 3 for output 1.
    :param out2: Configuration of exponential overshoot filter 3 for output 2.
    :param out3: Configuration of exponential overshoot filter 3 for output 3.
    :param out4: Configuration of exponential overshoot filter 3 for output 4.
    :param out5: Configuration of exponential overshoot filter 3 for output 5.
    :param marker0: Configuration of exponential overshoot filter 3 for marker 0.
    """

    out0: str | None = None
    out1: str | None = None
    out2: str | None = None
    out3: str | None = None
    out4: str | None = None
    out5: str | None = None

    marker0: str | None = None


@dataclass
class LoConfig:
    """Configuration for the local oscillator in QCM-RF, QRM-RF, and QRC.

    :param out0_en: Flag to enable/diable the LO on output 0. Relevant in QCM-RF.
    :param out0_freq: Frequency (in Hz) for the LO attached to output 0. Relevant in QCM-RF.
    :param out1_en: Flag to enable/diable the LO on output 1.Relevant in QCM-RF.
    :param out1_freq: Frequency (in Hz) for the LO attached to output 1. Relevant in QCM-RF.
    :param out2_freq: Frequency (in Hz) for the LO attached to output 2. Relevant in QRC.
    :param out3_freq: Frequency (in Hz) for the LO attached to output 3. Relevant in QRC.
    :param out4_freq: Frequency (in Hz) for the LO attached to output 4. Relevant in QRC.
    :param out5_freq: Frequency (in Hz) for the LO attached to output 5. Relevant in QRC.
    :param out0_in0_en: Flag to enable/diable the LO common to output 0 and input 0.
        Relevant in QRM-RF.
    :param out0_in0_freq: Frequency (in Hz) for the LO common to output 0 and input 0.
        Relevant in QCM-RF/QRC.
    :param out1_in1_freq: Frequency (in Hz) for the LO common to output 1 and input 1.
        Relevant in QRC.
    """

    out0_en: bool | None = None
    out0_freq: float | None = None
    out1_en: bool | None = None
    out1_freq: float | None = None

    out2_freq: float | None = None
    out3_freq: float | None = None
    out4_freq: float | None = None
    out5_freq: float | None = None

    out0_in0_en: bool | None = None
    out0_in0_freq: float | None = None

    out1_in1_freq: float | None = None


@dataclass
class AttConfig:
    """Configuration for output/input attenuation.

    :param out0: Attenuation (in dB) for output 0.
    :param out1: Attenuation (in dB) for output 1.
    :param out2: Attenuation (in dB) for output 2.
    :param out3: Attenuation (in dB) for output 3.
    :param out4: Attenuation (in dB) for output 4.
    :param out5: Attenuation (in dB) for output 5.
    :param in0: Attenuation (in dB) for input 0.
    :param in1: Attenuation (in dB) for input 1.
    """

    out0: float | None = None
    out1: float | None = None
    out2: float | None = None
    out3: float | None = None
    out4: float | None = None
    out5: float | None = None

    in0: float | None = None
    in1: float | None = None


@dataclass
class GainConfig:
    """Configuration for input gain relevant in the QRM.

    :param in0: Gain (in dB) for input 0.
    :param in1: Gain (in dB) for input 1.
    """

    in0: int | None = None
    in1: int | None = None


@dataclass
class ScopeAcqConfig:
    """Scope acquisition configuration relevant in QRM/QRM-RF/QRC. Possible values For the
    trigger mode are 'sequencer' to trigger by sequencer, 'level' to trigger by input level.

    :param sequencer_select: Sequencer that specifies which sequencer triggers the scope
        acquisition when using sequencer trigger mode. It is a sequencer id, or a list of
        sequencer ids for each scope IQ pair.
    :param trigger_mode_path0: Trigger mode for input path 0.
    :param trigger_mode_path1: Trigger mode for input path 1.
    :param trigger_mode_path2: Trigger mode for input path 2.
    :param trigger_mode_path3: Trigger mode for input path 3.
    :param trigger_level_path0: Trigger level when using input level trigger mode for input
        path 0.
    :param trigger_level_path1: Trigger level when using input level trigger mode for input
        path 1.
    :param trigger_level_path2: Trigger level when using input level trigger mode for input
        path 2.
    :param trigger_level_path3: Trigger level when using input level trigger mode for input
        path 3.
    :param avg_mode_en_path0: Flag to enable/disable scope acquisition averaging mode for
        input path 0.
    :param avg_mode_en_path1: Flag to enable/disable scope acquisition averaging mode for
        input path 1.
    :param avg_mode_en_path2: Flag to enable/disable scope acquisition averaging mode for
        input path 2.
    :param avg_mode_en_path3: Flag to enable/disable scope acquisition averaging mode for
        input path 3.
    """

    sequencer_select: int | None = None

    trigger_mode_path0: str | None = None  # Enum: {‘sequencer’, ‘level’}
    trigger_mode_path1: str | None = None  # Enum: {‘sequencer’, ‘level’}
    trigger_mode_path2: str | None = None  # Enum: {‘sequencer’, ‘level’}
    trigger_mode_path3: str | None = None  # Enum: {‘sequencer’, ‘level’}

    trigger_level_path0: float | None = None
    trigger_level_path1: float | None = None
    trigger_level_path2: float | None = None
    trigger_level_path3: float | None = None

    avg_mode_en_path0: bool | None = None
    avg_mode_en_path1: bool | None = None
    avg_mode_en_path2: bool | None = None
    avg_mode_en_path3: bool | None = None


@dataclass
class ModuleConfig:
    """Configuration specification for module (the analogue side of the RF chain).

    :param marker_inverts: Dictionary mapping marker indices to a flag whether
                           to enable/disable marker inversion.
    :param offset: Offset configuration, see :class:`OffsetConfig`.
    :param lo: Local Oscillator configuration, see :class:`LoConfig`.
    :param attenuation: Attenuation configuration, see :class:`AttenuationConfig`.
    :param gain: Gain configuration, see :class:`GainConfig`.
    :param scope_acq: Scope acquisition configuration, see :class:`ScopeAcqConfig`.
    :param fir: FIR filter configuration, see :class:`FirConfig`.
    :param exp0: Exponential overshoot 0 configuration, see :class:`ExpOvershoot0Config`.
    :param exp1: Exponential overshoot 1 configuration, see :class:`ExpOvershoot1Config`.
    :param exp2: Exponential overshoot 2 configuration, see :class:`ExpOvershot2Config`.
    :param exp3: Exponential overshoot 3 configuration, see :class:`ExpOvershoot3Config`.
    """

    marker_inverts: dict[int, bool] = field(default_factory=dict)

    offset: OffsetConfig = field(default_factory=lambda: OffsetConfig())
    lo: LoConfig = field(default_factory=lambda: LoConfig())
    attenuation: AttConfig = field(default_factory=lambda: AttConfig())
    gain: GainConfig = field(default_factory=lambda: GainConfig())
    scope_acq: ScopeAcqConfig = field(default_factory=lambda: ScopeAcqConfig())
    fir: FirConfig = field(default_factory=lambda: FirConfig())
    exp0: ExpOvershoot0Config = field(default_factory=lambda: ExpOvershoot0Config())
    exp1: ExpOvershoot1Config = field(default_factory=lambda: ExpOvershoot1Config())
    exp2: ExpOvershoot2Config = field(default_factory=lambda: ExpOvershoot2Config())
    exp3: ExpOvershoot3Config = field(default_factory=lambda: ExpOvershoot3Config())


@dataclass
class QbloxConfig:
    """Object grouping configuration of the analogue side and the digital side of the RF
    chain. For a given output/input analogue path, :param:`module` describes the necessary
    QCodes configuration to set it up completely and :param:`sequencers` is a collection of
    sequencer indices that are allowed down/up the said output/input channel. Such mechanism
    allows us to restrict where the code generator is allowed to pick the next available
    sequencer.

    :param slot_idx: The slot index on the Qblox chassis (a.k.a. Cluster) where
                     the module is installed.
    :param module: Module configuration, see :class:`ModuleConfig`.
    :param sequencers: Statically mapped sequencers and their configuration,
                       see :class:`SequencerConfig`.
    """

    slot_idx: int | None = None
    module: ModuleConfig = field(default_factory=lambda: ModuleConfig())
    sequencers: dict[int, SequencerConfig] = field(default_factory=dict)
