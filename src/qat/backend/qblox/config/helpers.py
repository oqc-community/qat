# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd

from abc import ABC, abstractmethod

from qblox_instruments import Module, Sequencer

from qat.backend.qblox.config.specification import (
    MixerConfig,
    ModuleConfig,
    OffsetConfig,
    SequencerConfig,
)
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class QbloxConfigHelper(ABC):
    def __init__(
        self,
        module_config: ModuleConfig = None,
        sequencer_config: SequencerConfig = None,
    ):
        self.sequencer_config = sequencer_config or SequencerConfig()
        self.module_config = module_config or SequencerConfig()

    def configure(self, module: Module, sequencer: Sequencer):
        self.configure_module(module)
        self.configure_sequencer(sequencer)

    @abstractmethod
    def configure_module(self, module: Module):
        pass

    @abstractmethod
    def configure_sequencer(self, sequencer: Sequencer):
        pass

    def configure_connection(self, sequencer: Sequencer):
        if self.sequencer_config.connection.bulk_value:
            sequencer.connect_sequencer(*self.sequencer_config.connection.bulk_value)

    def configure_nco(self, sequencer: Sequencer):
        if self.sequencer_config.nco.freq:
            sequencer.nco_freq(self.sequencer_config.nco.freq)
        if self.sequencer_config.nco.prop_delay_comp_en:
            sequencer.nco_prop_delay_comp_en(self.sequencer_config.nco.prop_delay_comp_en)
        if self.sequencer_config.nco.prop_delay_comp:
            sequencer.nco_prop_delay_comp(self.sequencer_config.nco.prop_delay_comp)
        if self.sequencer_config.nco.phase_offs:
            sequencer.nco_phase_offs(self.sequencer_config.nco.phase_offs)

    def configure_awg(self, sequencer: Sequencer):
        if self.sequencer_config.awg.mod_en:
            sequencer.mod_en_awg(self.sequencer_config.awg.mod_en)
        if self.sequencer_config.awg.gain_path0:
            sequencer.gain_awg_path0(self.sequencer_config.awg.gain_path0)
        if self.sequencer_config.awg.gain_path1:
            sequencer.gain_awg_path1(self.sequencer_config.awg.gain_path1)
        if self.sequencer_config.awg.offset_path0:
            sequencer.offset_awg_path0(self.sequencer_config.awg.offset_path0)
        if self.sequencer_config.awg.offset_path1:
            sequencer.offset_awg_path1(self.sequencer_config.awg.offset_path1)

        if self.sequencer_config.awg.cont_mode_en_path0:
            sequencer.cont_mode_en_awg_path0(self.sequencer_config.awg.cont_mode_en_path0)
        if self.sequencer_config.awg.cont_mode_en_path1:
            sequencer.cont_mode_en_awg_path1(self.sequencer_config.awg.cont_mode_en_path1)
        if self.sequencer_config.awg.cont_mode_waveform_idx_path0:
            sequencer.cont_mode_waveform_idx_awg_path0(
                self.sequencer_config.awg.cont_mode_waveform_idx_path0
            )
        if self.sequencer_config.awg.cont_mode_waveform_idx_path1:
            sequencer.cont_mode_waveform_idx_awg_path1(
                self.sequencer_config.awg.cont_mode_waveform_idx_path1
            )

        if self.sequencer_config.awg.cont_mode_en_path0:
            sequencer.cont_mode_en_awg_path0(self.sequencer_config.awg.cont_mode_en_path0)
        if self.sequencer_config.awg.cont_mode_en_path1:
            sequencer.cont_mode_en_awg_path1(self.sequencer_config.awg.cont_mode_en_path1)
        if self.sequencer_config.awg.cont_mode_waveform_idx_path0:
            sequencer.cont_mode_waveform_idx_awg_path0(
                self.sequencer_config.awg.cont_mode_waveform_idx_path0
            )
        if self.sequencer_config.awg.cont_mode_waveform_idx_path1:
            sequencer.cont_mode_waveform_idx_awg_path1(
                self.sequencer_config.awg.cont_mode_waveform_idx_path1
            )

    def configure_mixer(self, sequencer: Sequencer):
        if self.sequencer_config.mixer.phase_offset:
            sequencer.mixer_corr_phase_offset_degree(
                self.sequencer_config.mixer.phase_offset
            )
        if self.sequencer_config.mixer.gain_ratio:
            sequencer.mixer_corr_gain_ratio(self.sequencer_config.mixer.gain_ratio)

    def calibrate_mixer(
        self, module: Module, sequencer: Sequencer = None, connection: str = None
    ):
        """
        - Compensates for the LO leakage for input(s)/output(s) depending on module
        - Suppresses undesired sideband on sequencer

        :param module: A Module instance representing a QCM-RF or a QRM-RF card.
        :param sequencer: A Sequencer instance representing the sequencer.
            When None, sideband cal will run on all sequencers belonging to module `module`
        :param connection: An optional string indicating muxing of Sequencer to analog channel
            Defaults to "out0"
        """

        sequencers = [sequencer] if sequencer else module.sequencers

        if connection:
            log.info(
                f"Provided argument `connection` is {connection}. Will overwrite sequencer's connections"
            )
            module.disconnect_outputs()
            for sequencer in sequencers:
                sequencer.connect_sequencer(connection)
        else:
            log.info(
                "Argument `connection` not provided. Assuming `connection` already specified"
            )

        log.info(f"Calibrating LO leakage on module {module}")
        offset_config = self.calibrate_lo_leakage(module)
        mixer_configs = {}
        for sequencer in sequencers:
            mixer_configs[sequencer.seq_idx] = self.calibrate_sideband(
                sequencer, connection
            )

        if connection:
            log.info(
                f"Mixer calibration finished. Resetting sequencer connections for module {module.slot_idx}"
            )
            module.disconnect_outputs()

        return offset_config, mixer_configs

    @abstractmethod
    def calibrate_lo_leakage(self, module: Module):
        pass

    def calibrate_sideband(self, sequencer: Sequencer, connection: str = None):
        log.info(f"Calibrating sidebands on sequencer {sequencer}")
        sequencer.sideband_cal()
        sequencer.arm_sequencer()
        sequencer.start_sequencer()

        return MixerConfig(
            phase_offset=sequencer.mixer_corr_phase_offset_degree(),
            gain_ratio=sequencer.mixer_corr_gain_ratio(),
        )


class QcmConfigHelper(QbloxConfigHelper):
    pass


class QcmRfConfigHelper(QcmConfigHelper):
    def configure_module(self, module: Module):
        self.configure_lo(module)
        self.configure_attenuation(module)
        self.configure_offset(module)

    def configure_sequencer(self, sequencer: Sequencer):
        self.configure_connection(sequencer)
        self.configure_nco(sequencer)
        self.configure_awg(sequencer)
        self.configure_mixer(sequencer)

    def configure_lo(self, module: Module):
        if self.module_config.lo.out0_en:
            module.out0_lo_en(self.module_config.lo.out0_en)  # Switch the LO 0 on
        if self.module_config.lo.out0_freq:
            module.out0_lo_freq(self.module_config.lo.out0_freq)
        if self.module_config.lo.out1_en:
            module.out1_lo_en(self.module_config.lo.out1_en)  # Switch the LO 1 on
        if self.module_config.lo.out1_freq:
            module.out1_lo_freq(self.module_config.lo.out1_freq)

    def configure_attenuation(self, module: Module):
        if self.module_config.attenuation.out0:
            module.out0_att(self.module_config.attenuation.out0)
        if self.module_config.attenuation.out1:
            module.out1_att(self.module_config.attenuation.out1)

    def configure_offset(self, module: Module):
        if self.module_config.offset.out0_path0:
            module.out0_offset_path0(self.module_config.offset.out0_path0)
        if self.module_config.offset.out0_path1:
            module.out0_offset_path1(self.module_config.offset.out0_path1)
        if self.module_config.offset.out1_path0:
            module.out1_offset_path0(self.module_config.offset.out1_path0)
        if self.module_config.offset.out1_path1:
            module.out1_offset_path1(self.module_config.offset.out1_path1)

    def calibrate_lo_leakage(self, module: Module):
        module.out0_lo_cal()
        module.out1_lo_cal()

        return OffsetConfig(
            out0_path0=module.out0_offset_path0(),
            out0_path1=module.out0_offset_path1(),
            out1_path0=module.out1_offset_path0(),
            out1_path1=module.out1_offset_path1(),
        )


class QrmConfigHelper(QbloxConfigHelper):
    def configure_scope_acq(self, module: Module):
        scope_acq = self.module_config.scope_acq
        if scope_acq.sequencer_select:
            module.scope_acq_sequencer_select(scope_acq.sequencer_select)
        if scope_acq.trigger_mode_path0:
            module.scope_acq_trigger_mode_path0(scope_acq.trigger_mode_path0)
        if scope_acq.avg_mode_en_path0:
            module.scope_acq_avg_mode_en_path0(scope_acq.avg_mode_en_path0)
        if scope_acq.trigger_mode_path1:
            module.scope_acq_trigger_mode_path1(scope_acq.trigger_mode_path1)
        if scope_acq.avg_mode_en_path1:
            module.scope_acq_avg_mode_en_path1(scope_acq.avg_mode_en_path1)


class QrmRfConfigHelper(QrmConfigHelper):
    def configure_module(self, module: Module):
        self.configure_lo(module)
        self.configure_attenuation(module)
        self.configure_offset(module)
        self.configure_scope_acq(module)

    def configure_sequencer(self, sequencer: Sequencer):
        self.configure_connection(sequencer)
        self.configure_nco(sequencer)
        self.configure_awg(sequencer)
        self.configure_mixer(sequencer)
        self.configure_acq(sequencer)

    def configure_lo(self, module: Module):
        if self.module_config.lo.out0_in0_en:
            module.out0_in0_lo_en(self.module_config.lo.out0_in0_en)  # Switch the LO on
        if self.module_config.lo.out0_in0_freq:
            module.out0_in0_lo_freq(self.module_config.lo.out0_in0_freq)

    def configure_attenuation(self, module: Module):
        if self.module_config.attenuation.out0:
            module.out0_att(self.module_config.attenuation.out0)
        if self.module_config.attenuation.in0:
            module.in0_att(self.module_config.attenuation.in0)

    def configure_offset(self, module: Module):
        if self.module_config.offset.out0_path0:
            module.out0_offset_path0(self.module_config.offset.out0_path0)
        if self.module_config.offset.out0_path1:
            module.out0_offset_path1(self.module_config.offset.out0_path1)
        if self.module_config.offset.in0_path0:
            module.in0_offset_path0(self.module_config.offset.in0_path0)
        if self.module_config.offset.in0_path1:
            module.in0_offset_path1(self.module_config.offset.in0_path1)

    def configure_acq(self, sequencer: Sequencer):
        # Square weight integration
        if self.sequencer_config.demod_en_acq:
            sequencer.demod_en_acq(self.sequencer_config.demod_en_acq)
        if self.sequencer_config.square_weight_acq.integration_length:
            sequencer.integration_length_acq(
                self.sequencer_config.square_weight_acq.integration_length
            )
        if self.sequencer_config.thresholded_acq.rotation:
            sequencer.thresholded_acq_rotation(
                self.sequencer_config.thresholded_acq.rotation
            )
        if self.sequencer_config.thresholded_acq.threshold:
            sequencer.thresholded_acq_threshold(
                self.sequencer_config.thresholded_acq.threshold
            )
        if self.sequencer_config.thresholded_acq.trigger_en:
            sequencer.thresholded_acq_trigger_en(
                self.sequencer_config.thresholded_acq.trigger_en
            )
        if self.sequencer_config.thresholded_acq.trigger_address:
            sequencer.thresholded_acq_trigger_address(
                self.sequencer_config.thresholded_acq.trigger_address
            )
        if self.sequencer_config.thresholded_acq.trigger_invert:
            sequencer.thresholded_acq_trigger_invert(
                self.sequencer_config.thresholded_acq.trigger_invert
            )

    def calibrate_lo_leakage(self, module: Module):
        module.out0_in0_lo_cal()

        return OffsetConfig(
            out0_path0=module.out0_offset_path0(),
            out0_path1=module.out0_offset_path1(),
            in0_path0=module.in0_offset_path0(),
            in0_path1=module.in0_offset_path1(),
        )
