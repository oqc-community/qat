# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd

import uuid
from dataclasses import dataclass

import numpy as np
import pytest
from qblox_instruments import ClusterType

from qat.purr.backends.qblox.codegen import QbloxEmitter, calculate_duration
from qat.purr.backends.qblox.config import (
    ModuleConfig,
    QcmConfigHelper,
    QcmRfConfigHelper,
    QrmConfigHelper,
    QrmRfConfigHelper,
    SequencerConfig,
)
from qat.purr.backends.qblox.constants import Constants
from qat.purr.compiler.devices import PulseShapeType
from qat.purr.compiler.emitter import InstructionEmitter
from qat.purr.compiler.instructions import Acquire
from qat.purr.compiler.runtime import get_builder

from tests.unit.backend.qblox.conftest import random_resource


class TestQbloxConfigMixin:
    @staticmethod
    def extract_lo_and_nco_freqs(target):
        if target.fixed_if:  # NCO freq constant
            nco_freq = target.baseband_if_frequency
            lo_freq = target.frequency - nco_freq
        else:  # LO freq constant
            lo_freq = target.baseband_frequency
            nco_freq = target.frequency - lo_freq

        return lo_freq, nco_freq

    @staticmethod
    def setup_qcm_mixer_config(module_config: ModuleConfig, i_offset, q_offset):
        module_config.offset.out0 = i_offset
        module_config.offset.out1 = q_offset
        module_config.offset.out2 = i_offset
        module_config.offset.out3 = q_offset

    @staticmethod
    def setup_qcm_rf_mixer_config(module_config: ModuleConfig, i_offset, q_offset):
        module_config.offset.out0_path0 = i_offset
        module_config.offset.out0_path1 = q_offset
        module_config.offset.out1_path0 = i_offset
        module_config.offset.out1_path1 = q_offset

    @staticmethod
    def setup_qrm_mixer_config(module_config: ModuleConfig, i_offset, q_offset):
        module_config.offset.out0 = i_offset
        module_config.offset.out1 = q_offset
        module_config.offset.in0 = i_offset
        module_config.offset.in1 = q_offset

    @staticmethod
    def setup_qrm_rf_mixer_config(module_config: ModuleConfig, i_offset, q_offset):
        module_config.offset.out0_path0 = i_offset
        module_config.offset.out0_path1 = q_offset
        module_config.offset.in0_path0 = i_offset
        module_config.offset.in0_path1 = q_offset

    @staticmethod
    def setup_sequencer_mixer_config(seq_config: SequencerConfig, phase_offset, gain_ratio):
        seq_config.mixer.gain_ratio = gain_ratio
        seq_config.mixer.phase_offset = phase_offset


@pytest.mark.parametrize("model", [None], indirect=True)
class TestSequencerConfig(TestQbloxConfigMixin):
    def test_lo_and_nco_freq(self, model):
        width = 100e-9
        amp = 0.5
        qubit = model.get_qubit(0)
        drive_channel = qubit.get_drive_channel()
        measure_channel = qubit.get_measure_channel()

        builder = (
            get_builder(model)
            .pulse(drive_channel, PulseShapeType.SQUARE, width=width, amp=amp)
            .measure_mean_z(qubit)
        )
        qat_file = InstructionEmitter().emit(builder.instructions, model)
        packages = QbloxEmitter(qat_file.repeat).emit(qat_file.instructions)

        assert len(packages) == 2

        # Drive
        drive_pkg = next((pkg for pkg in packages if pkg.target == drive_channel))
        lo_freq, nco_freq = self.extract_lo_and_nco_freqs(drive_channel)
        module, sequencer = model.control_hardware.allocate_resources(drive_pkg)
        model.control_hardware.configure(drive_pkg, module, sequencer)
        assert sequencer.nco_freq() == nco_freq
        if module.out0_lo_en():
            assert module.out0_lo_freq() == lo_freq
        if module.out1_lo_en():
            assert module.out1_lo_freq() == lo_freq

        # Measurement
        acquire = next(
            (inst for inst in qat_file.instructions if isinstance(inst, Acquire))
        )
        measure_pkg = next((pkg for pkg in packages if pkg.target == measure_channel))
        module, sequencer = model.control_hardware.allocate_resources(measure_pkg)
        model.control_hardware.configure(measure_pkg, module, sequencer)
        hwm_seq_config = measure_channel.physical_channel.config.sequencers[
            sequencer.seq_idx
        ]
        pkg_seq_config = measure_pkg.sequencer_config

        assert pkg_seq_config.square_weight_acq.integration_length == calculate_duration(
            acquire
        )
        assert (
            pkg_seq_config.square_weight_acq.integration_length
            == hwm_seq_config.square_weight_acq.integration_length
        )
        assert (
            sequencer.integration_length_acq()
            == pkg_seq_config.square_weight_acq.integration_length
        )

        lo_freq, nco_freq = self.extract_lo_and_nco_freqs(measure_channel)
        assert sequencer.nco_freq() == nco_freq
        if module.out0_in0_lo_en():
            assert module.out0_in0_lo_freq() == lo_freq


@dataclass
class MixerTestValues:
    num_points = 2  # Low value for testing to reduce the size of the cartesian products

    # Module values
    qcm_i_offsets = np.linspace(-2.5, 2.5, num_points)  # I offsets (Volt)
    qcm_q_offsets = np.linspace(-2.5, 2.5, num_points)  # Q offsets (Volt)
    qcm_rf_i_offsets = np.linspace(-84, 73, num_points)  # I offsets (mVolt)
    qcm_rf_q_offsets = np.linspace(-84, 73, num_points)  # Q offsets (mVolt)
    qrm_i_offsets = np.linspace(-0.09, 0.09, num_points)  # I offsets (Volt)
    qrm_q_offsets = np.linspace(-0.09, 0.09, num_points)  # Q offsets (Volt)
    qrm_rf_i_offsets = np.linspace(-0.09, 0.09, num_points)  # I offsets (Volt)
    qrm_rf_q_offsets = np.linspace(-0.09, 0.09, num_points)  # Q offsets (Volt)

    # Sequencer values
    phase_offsets = np.linspace(-45, 45, num_points)  # Phase offsets (Degree)
    gain_ratios = np.linspace(0.5, 2, num_points)  # Gain ratios


class TestMixerConfig(TestQbloxConfigMixin):
    mixer_values = MixerTestValues()

    @pytest.mark.skip("Needs config implementation for QCM")
    @pytest.mark.parametrize("phase_offset", mixer_values.phase_offsets)
    @pytest.mark.parametrize("gain_ratio", mixer_values.gain_ratios)
    @pytest.mark.parametrize("i_offset", mixer_values.qcm_i_offsets)
    @pytest.mark.parametrize("q_offset", mixer_values.qcm_q_offsets)
    def test_qcm_mixer_config(self, request, phase_offset, gain_ratio, i_offset, q_offset):
        module_config = ModuleConfig()
        sequencer_config = SequencerConfig()

        # The qcodes package generates a warning if the name of the cluster contains dashes.
        name = f"{request.node.originalname}_{uuid.uuid4()}".replace("-", "_")
        module, sequencer = random_resource(ClusterType.CLUSTER_QCM, name)

        self.setup_qcm_mixer_config(module_config, i_offset, q_offset)
        self.setup_sequencer_mixer_config(sequencer_config, phase_offset, gain_ratio)
        QcmConfigHelper(module_config, sequencer_config).configure(module, sequencer)

        assert module.out0_offset() == module_config.offset.out0
        assert module.out1_offset() == module_config.offset.out1
        assert module.out2_offset() == module_config.offset.out2
        assert module.out3_offset() == module_config.offset.out3

    @pytest.mark.parametrize("phase_offset", mixer_values.phase_offsets)
    @pytest.mark.parametrize("gain_ratio", mixer_values.gain_ratios)
    @pytest.mark.parametrize("i_offset", mixer_values.qcm_rf_i_offsets)
    @pytest.mark.parametrize("q_offset", mixer_values.qcm_rf_q_offsets)
    def test_qcm_rf_mixer_config(
        self, request, phase_offset, gain_ratio, i_offset, q_offset
    ):
        module_config = ModuleConfig()
        sequencer_config = SequencerConfig()

        # The qcodes package generates a warning if the name of the cluster contains dashes.
        name = f"{request.node.originalname}_{uuid.uuid4()}".replace("-", "_")
        module, sequencer = random_resource(ClusterType.CLUSTER_QCM_RF, name)

        self.setup_qcm_rf_mixer_config(module_config, i_offset, q_offset)
        self.setup_sequencer_mixer_config(sequencer_config, phase_offset, gain_ratio)
        QcmRfConfigHelper(module_config, sequencer_config).configure(module, sequencer)

        assert module.out0_offset_path0() == module_config.offset.out0_path0
        assert module.out0_offset_path1() == module_config.offset.out0_path1
        assert module.out1_offset_path0() == module_config.offset.out1_path0
        assert module.out1_offset_path1() == module_config.offset.out1_path1

    @pytest.mark.skip("Needs config implementation for QRM")
    @pytest.mark.parametrize("phase_offset", mixer_values.phase_offsets)
    @pytest.mark.parametrize("gain_ratio", mixer_values.gain_ratios)
    @pytest.mark.parametrize("i_offset", mixer_values.qrm_i_offsets)
    @pytest.mark.parametrize("q_offset", mixer_values.qrm_q_offsets)
    def test_qrm_mixer_config(self, request, phase_offset, gain_ratio, i_offset, q_offset):
        module_config = ModuleConfig()
        sequencer_config = SequencerConfig()

        # The qcodes package generates a warning if the name of the cluster contains dashes.
        name = f"{request.node.originalname}_{uuid.uuid4()}".replace("-", "_")
        module, sequencer = random_resource(ClusterType.CLUSTER_QRM, name)

        self.setup_qrm_mixer_config(module_config, i_offset, q_offset)
        self.setup_sequencer_mixer_config(sequencer_config, phase_offset, gain_ratio)
        QrmConfigHelper(module_config, sequencer_config).configure(module, sequencer)

        assert module.out0_offset() == module_config.offset.out0
        assert module.out1_offset() == module_config.offset.out1
        assert module.in0_offset() == module_config.offset.in0
        assert module.in1_offset() == module_config.offset.in1

    @pytest.mark.parametrize("phase_offset", mixer_values.phase_offsets)
    @pytest.mark.parametrize("gain_ratio", mixer_values.gain_ratios)
    @pytest.mark.parametrize("i_offset", mixer_values.qrm_rf_i_offsets)
    @pytest.mark.parametrize("q_offset", mixer_values.qrm_rf_q_offsets)
    def test_qrm_rf_mixer_config(
        self, request, phase_offset, gain_ratio, i_offset, q_offset
    ):
        module_config = ModuleConfig()
        sequencer_config = SequencerConfig()

        # The qcodes package generates a warning if the name of the cluster contains dashes.
        name = f"{request.node.originalname}_{uuid.uuid4()}".replace("-", "_")
        module, sequencer = random_resource(ClusterType.CLUSTER_QRM_RF, name)

        self.setup_qrm_rf_mixer_config(module_config, i_offset, q_offset)
        self.setup_sequencer_mixer_config(sequencer_config, phase_offset, gain_ratio)
        QrmRfConfigHelper(module_config, sequencer_config).configure(module, sequencer)

        assert module.out0_offset_path0() == module_config.offset.out0_path0
        assert module.out0_offset_path1() == module_config.offset.out0_path1

        # TODO - potential bug with the dummy structures
        with pytest.raises(AssertionError):
            assert module.in0_offset_path0() == module_config.offset.in0_path0

        with pytest.raises(AssertionError):
            assert module.in0_offset_path1() == module_config.offset.in0_path1


@dataclass
class AcqTestValues:
    num_points = 5  # Low value for testing to reduce the size of the cartesian products

    int_lengths = np.random.choice(
        np.arange(
            Constants.MIN_ACQ_INTEGRATION_LENGTH, Constants.MAX_ACQ_INTEGRATION_LENGTH, 4
        ),
        size=num_points,
    )
    rotations = np.random.choice(np.arange(0, 360), size=num_points).astype(float)
    thresholds = np.random.choice(
        np.arange(Constants.MIN_ACQ_THRESHOLD, Constants.MAX_ACQ_THRESHOLD), size=num_points
    ).astype(float)


class TestAcqConfig(TestQbloxConfigMixin):
    acq_values = AcqTestValues()

    @pytest.mark.parametrize("int_length", acq_values.int_lengths)
    def test_qrm_rf_square_acq(self, request, int_length):
        module_config = ModuleConfig()
        sequencer_config = SequencerConfig()

        # The qcodes package generates a warning if the name of the cluster contains dashes.
        name = f"{request.node.originalname}_{uuid.uuid4()}".replace("-", "_")
        module, sequencer = random_resource(ClusterType.CLUSTER_QRM_RF, name)

        sequencer_config.square_weight_acq.integration_length = int_length

        QrmRfConfigHelper(module_config, sequencer_config).configure_acq(sequencer)

        assert int_length == sequencer_config.square_weight_acq.integration_length
        assert (
            sequencer_config.square_weight_acq.integration_length
            == sequencer.integration_length_acq()
        )

    @pytest.mark.parametrize("rotation", acq_values.rotations)
    @pytest.mark.parametrize("threshold", acq_values.thresholds)
    def test_qrm_rf_thresholded_acq(self, request, rotation, threshold):
        module_config = ModuleConfig()
        sequencer_config = SequencerConfig()

        # The qcodes package generates a warning if the name of the cluster contains dashes.
        name = f"{request.node.originalname}_{uuid.uuid4()}".replace("-", "_")
        module, sequencer = random_resource(ClusterType.CLUSTER_QRM_RF, name)

        sequencer_config.thresholded_acq.rotation = rotation
        sequencer_config.thresholded_acq.threshold = threshold

        QrmRfConfigHelper(module_config, sequencer_config).configure_acq(sequencer)

        assert sequencer_config.thresholded_acq.rotation == rotation
        assert np.isclose(
            sequencer.thresholded_acq_rotation(), sequencer_config.thresholded_acq.rotation
        )

        assert sequencer_config.thresholded_acq.threshold == threshold
        assert np.isclose(
            sequencer.thresholded_acq_threshold(),
            sequencer_config.thresholded_acq.threshold,
        )
