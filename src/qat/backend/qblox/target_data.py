# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
from warnings import warn

from pydantic import NegativeFloat, NegativeInt, PositiveFloat, PositiveInt

from qat.model.target_data import TargetData
from qat.utils.pydantic import NoExtraFieldsFrozenModel


class Q1asmDescription(NoExtraFieldsFrozenModel):
    """
    Constants related to Q1asm.

    :param min_gain: Minimum gain in Q1ASM programs.
    :param max_gain: Maximum gain in Q1ASM programs.
    :param min_offset: Minimum offset in Q1ASM programs.
    :param max_offset: Maximum offset in Q1ASM programs.
    :param max_wait_time: Max size of wait instruction immediate operands in Q1ASM programs.
                          Max value allowed by assembler is 2**16-1, but this is the largest
                          that is a multiple of 4 ns.
    :param register_size: Size of registers in Q1ASM programs.
    :param loop_unroll_threshold: Size above which loops have tolerable overhead.
    """

    min_gain: NegativeInt = -pow(2, 15)
    max_gain: PositiveInt = pow(2, 15) - 1
    min_offset: NegativeInt = -pow(2, 15)
    max_offset: PositiveInt = pow(2, 15) - 1
    max_wait_time: PositiveInt = pow(2, 16) - 4
    register_size: PositiveInt = pow(2, 32) - 1
    loop_unroll_threshold: PositiveInt = 4

    @classmethod
    def default(cls):
        warn(
            f"`{cls.__name__}.default()` is deprecated; use `{cls.__name__}()` instead. "
            "This will be removed in v4.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls()


class SequencerDescription(NoExtraFieldsFrozenModel):
    """
    Constants related to Qblox Sequencer.

    :param grid_time: Clock period of the sequencers. All time intervals used must be multiples of this value.
    :param nco_min_freq: Minimum NCO frequency.
    :param nco_max_freq: Maximum NCO frequency.
    :param nco_max_phase_steps: Maximum NCO phase steps.
    :param nco_phase_steps_per_deg: The number of steps per degree for NCO phase instructions arguments.
    :param nco_freq_steps_per_hz: The number of steps per Hz for the NCO set_freq instruction.
    :param nco_freq_limit_steps: The maximum and minimum frequency expressed in steps for the NCO set_freq instruction.
                                 For the minimum we multiply by -1.
    :param number_of_registers: Number of registers available in the Qblox sequencers.
    :param min_acq_integration_length: Minimum integration lengths.
    :param max_acq_integration_length: Maximum integration lengths.
    :param min_acq_threshold: Minimum thresholds for the thresholded acquisition.
    :param max_acq_threshold: Maximum thresholds for the thresholded acquisition.
    :param max_sample_size_waveforms: Maximal amount of samples in the waveforms to be uploaded to a sequencer.
    """

    grid_time: PositiveInt = 4
    nco_min_freq: NegativeInt = -500e6
    nco_max_freq: PositiveInt = 500e6
    nco_max_phase_steps: PositiveInt = int(1e9)
    nco_phase_steps_per_deg: PositiveFloat = 1e9 / 360
    nco_freq_steps_per_hz: PositiveInt = 4
    nco_freq_limit_steps: PositiveInt = 2e9
    number_of_registers: PositiveInt = 64
    min_acq_integration_length: PositiveInt = 4
    max_acq_integration_length: PositiveInt = (1 << 24) - 4
    min_acq_threshold: NegativeInt = -((1 << 24) - 4)
    max_acq_threshold: PositiveInt = (1 << 24) - 4
    max_sample_size_waveforms: PositiveInt = 16384
    max_num_instructions: PositiveInt = 12288

    @classmethod
    def default(cls):
        warn(
            f"`{cls.__name__}.default()` is deprecated; use `{cls.__name__}()` instead. "
            "This will be removed in v4.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls()


class ControlSequencerDescription(SequencerDescription):
    max_num_instructions: PositiveInt = 16384


class ReadoutSequencerDescription(SequencerDescription):
    max_num_instructions: PositiveInt = 12288


class ModuleDescription(NoExtraFieldsFrozenModel):
    """
    Common constants related to Qblox Modules.

    :param number_of_sequencers: Number of sequencers.

    """

    number_of_sequencers: PositiveInt = 6

    @classmethod
    def default(cls):
        warn(
            f"`{cls.__name__}.default()` is deprecated; use `{cls.__name__}()` instead. "
            "This will be removed in v4.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls()


class QcmDescription(ModuleDescription):
    """
    Constants related to Qblox QCM Module.

    :param min_qcm_offset_v: Minimum offset for QCM.
    :param max_qcm_offset_v: Maximum offset for QCM.
    """

    min_qcm_offset_v: NegativeFloat = -2.5
    max_qcm_offset_v: PositiveFloat = 2.5


class QcmRfDescription(QcmDescription):
    """
    Constants related to Qblox QCM-RF Module.

    :param min_qcm_rf_offset_mv: Minimum offset for QCM-RF.
    :param max_qcm_rf_offset_mv: Maximum offset for QCM-RF.
    """

    min_qcm_rf_offset_mv: NegativeInt = -84
    max_qcm_rf_offset_mv: PositiveInt = 73


class QrmDescription(ModuleDescription):
    """
    Constants related to Qblox QRM Module.

    :param min_qrm_offset_v: Minimum offset for QRM.
    :param max_qrm_offset_v: Maximum offset for QRM.
    :param min_sample_size_scope_acquisitions: Minimum amount of scope trace acquisition datapoints returned.
    :param max_sample_size_scope_acquisitions: Maximal amount of scope trace acquisition datapoints returned.
    :param max_binned_acquisitions: Each QRM(-RF) module has a maximum of 3M memory bins. This memory can be
                                    dynamically allocated by each of the 6 sequencers. For example, all 6 sequencers
                                    can evenly allocate 500K bins simultaneously or a single sequencers can allocate
                                    the whole 3M and leaves nothing for other sequencers.
    """

    min_qrm_offset_v: NegativeFloat = -0.09
    max_qrm_offset_v: PositiveFloat = 0.09
    min_sample_size_scope_acquisitions: PositiveInt = 4
    max_sample_size_scope_acquisitions: PositiveInt = 16384
    max_binned_acquisitions: PositiveInt = 3_000_000


class QrmRfDescription(QrmDescription):
    """
    Constants related to Qblox QRM-RF Module.

    :param min_qrm_rf_offset_v: Minimum offset for QRM-RF.
    :param max_qrm_rf_offset_v: Maximum offset for QRM-RF.
    """

    min_qrm_rf_offset_v: NegativeFloat = -0.09
    max_qrm_rf_offset_v: PositiveFloat = 0.09


class QbloxTargetData(TargetData):
    """
    Taxonomy of constants for Qblox electronics.

    :param Q1ASM_DATA: Q1asm related constants.
    :param CONTROL_SEQUENCER_DATA: Control sequencer related constants.
    :param READOUT_SEQUENCER_DATA: Readout sequencer related constants.
    :param QCM_DATA: Constants related to the QCM module.
    :param QCM_RF_DATA: Constants related to the QCM-RF module.
    :param QRM_DATA: Constants related to the QRM module.
    :param QRM_RF_DATA: Constants related to the QRM module.
    """

    Q1ASM_DATA: Q1asmDescription = Q1asmDescription()
    CONTROL_SEQUENCER_DATA: ControlSequencerDescription = ControlSequencerDescription()
    READOUT_SEQUENCER_DATA: ReadoutSequencerDescription = ReadoutSequencerDescription()
    QCM_DATA: QcmDescription = QcmDescription()
    QCM_RF_DATA: QcmRfDescription = QcmRfDescription()
    QRM_DATA: QrmDescription = QrmDescription()
    QRM_RF_DATA: QrmRfDescription = QrmRfDescription()
