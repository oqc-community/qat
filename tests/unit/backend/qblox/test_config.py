# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd

from dataclasses import dataclass

import numpy as np
import pytest
from qblox_instruments import ClusterType

from qat.backend.qblox.config.helpers import (
    QcmConfigHelper,
    QcmRfConfigHelper,
    QrcConfigHelper,
    QrmConfigHelper,
    QrmRfConfigHelper,
)
from qat.backend.qblox.config.specification import ModuleConfig, SequencerConfig
from qat.purr.backends.qblox.constants import Constants

rng = np.random.default_rng(42)  # Seed for reproducibility


class TestQbloxConfigMixin:
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

    @staticmethod
    def setup_qrc_attenuation_config(module_config: ModuleConfig, attenuation):
        module_config.attenuation.out0 = attenuation
        module_config.attenuation.out1 = attenuation
        module_config.attenuation.out2 = attenuation
        module_config.attenuation.out3 = attenuation
        module_config.attenuation.out4 = attenuation
        module_config.attenuation.out5 = attenuation

        module_config.attenuation.in0 = attenuation
        module_config.attenuation.in1 = attenuation

    @staticmethod
    def setup_qrc_latency_config(module_config: ModuleConfig, latency):
        module_config.latency.out0 = latency
        module_config.latency.out1 = latency
        module_config.latency.out2 = latency
        module_config.latency.out3 = latency
        module_config.latency.out4 = latency
        module_config.latency.out5 = latency

    @staticmethod
    def setup_qrc_lo_config(module_config: ModuleConfig, lo_freq):
        module_config.lo.out0_in0_freq = lo_freq
        module_config.lo.out1_in1_freq = lo_freq

        module_config.lo.out2_freq = lo_freq
        module_config.lo.out3_freq = lo_freq
        module_config.lo.out4_freq = lo_freq
        module_config.lo.out5_freq = lo_freq


@dataclass
class ConfigTestValues:
    max_num_points = 50


@dataclass
class MixerTestValues(ConfigTestValues):
    num_points = 2

    # Module values
    qcm_i_offsets = rng.choice(np.linspace(-2.5, 2.5), size=num_points)  # fmt: skip # I offsets (Volt)
    qcm_q_offsets = rng.choice(np.linspace(-2.5, 2.5), size=num_points)  # fmt: skip # Q offsets (Volt)
    qcm_rf_i_offsets = rng.choice(np.linspace(-84, 73), size=num_points)  # fmt: skip # I offsets (mVolt)
    qcm_rf_q_offsets = rng.choice(np.linspace(-84, 73), size=num_points)  # fmt: skip # Q offsets (mVolt)
    qrm_i_offsets = rng.choice(np.linspace(-0.09, 0.09), size=num_points)  # fmt: skip # I offsets (Volt)
    qrm_q_offsets = rng.choice(np.linspace(-0.09, 0.09), size=num_points)  # fmt: skip # Q offsets (Volt)
    qrm_rf_i_offsets = rng.choice(np.linspace(-0.09, 0.09), size=num_points)  # fmt: skip # I offsets (Volt)
    qrm_rf_q_offsets = rng.choice(np.linspace(-0.09, 0.09), size=num_points)  # fmt: skip # Q offsets (Volt)

    # Sequencer values
    phase_offsets = rng.choice(np.linspace(-45, 45), size=num_points)  # fmt: skip # Phase offsets (Degree)
    gain_ratios = rng.choice(np.linspace(0.5, 2), size=num_points)  # fmt: skip # Gain ratios


@dataclass
class AcqTestValues:
    num_points = 5

    int_lengths = rng.choice(
        np.arange(
            Constants.MIN_ACQ_INTEGRATION_LENGTH, Constants.MAX_ACQ_INTEGRATION_LENGTH, 4
        ),
        size=num_points,
    )
    rotations = rng.choice(np.arange(0, 360), size=num_points).astype(float)
    thresholds = rng.choice(
        np.arange(Constants.MIN_ACQ_THRESHOLD, Constants.MAX_ACQ_THRESHOLD), size=num_points
    ).astype(float)


@dataclass
class AttenuationTestValues:
    """
    Attenuation values (dB)
    """

    num_points = 5

    qcm_rf_attenuations = rng.choice(np.arange(0, 60 + 2, 2), size=num_points)
    qrm_rf_in_attenuations = rng.choice(np.arange(0, 30 + 2, 2), size=num_points)
    qrm_rf_out_attenuations = rng.choice(np.arange(0, 60 + 2, 2), size=num_points)
    qrc_attenuations = rng.choice(np.arange(0, 31.5 + 0.5, 0.5), size=num_points)


@dataclass
class LatencyTestValues:
    """
    Latency values (s)
    """

    num_points = 3

    latencies = rng.choice(np.linspace(0, 11), size=num_points)


@dataclass
class LoTestValues:
    """
    LO frequencies (Hz)
    """

    num_points = 3

    qcm_rf_lo_freqs = rng.choice(np.linspace(2e9, 18e9), size=num_points)
    qrm_rf_lo_freqs = rng.choice(np.linspace(2e9, 18e9), size=num_points)
    qrc_lo_freqs = rng.choice(np.arange(500e6, 10.1e9 + 100e6, 100e6), size=num_points)


@pytest.mark.parametrize("qblox_resource", [None], indirect=True)
class TestMixerConfig(TestQbloxConfigMixin):
    test_values = MixerTestValues()

    @pytest.mark.skip("Needs config implementation for QCM")
    @pytest.mark.parametrize("phase_offset", test_values.phase_offsets)
    @pytest.mark.parametrize("gain_ratio", test_values.gain_ratios)
    @pytest.mark.parametrize("i_offset", test_values.qcm_i_offsets)
    @pytest.mark.parametrize("q_offset", test_values.qcm_q_offsets)
    def test_qcm_mixer_config(
        self, qblox_resource, phase_offset, gain_ratio, i_offset, q_offset
    ):
        module_config = ModuleConfig()
        sequencer_config = SequencerConfig()
        module, sequencer = qblox_resource(ClusterType.CLUSTER_QCM)

        self.setup_qcm_mixer_config(module_config, i_offset, q_offset)
        self.setup_sequencer_mixer_config(sequencer_config, phase_offset, gain_ratio)
        QcmConfigHelper(module_config, sequencer_config).configure(module, sequencer)

        assert module.out0_offset() == module_config.offset.out0
        assert module.out1_offset() == module_config.offset.out1
        assert module.out2_offset() == module_config.offset.out2
        assert module.out3_offset() == module_config.offset.out3

    @pytest.mark.parametrize("phase_offset", test_values.phase_offsets)
    @pytest.mark.parametrize("gain_ratio", test_values.gain_ratios)
    @pytest.mark.parametrize("i_offset", test_values.qcm_rf_i_offsets)
    @pytest.mark.parametrize("q_offset", test_values.qcm_rf_q_offsets)
    def test_qcm_rf_mixer_config(
        self, qblox_resource, phase_offset, gain_ratio, i_offset, q_offset
    ):
        module_config = ModuleConfig()
        sequencer_config = SequencerConfig()
        module, sequencer = qblox_resource(ClusterType.CLUSTER_QCM_RF)

        self.setup_qcm_rf_mixer_config(module_config, i_offset, q_offset)
        self.setup_sequencer_mixer_config(sequencer_config, phase_offset, gain_ratio)
        QcmRfConfigHelper(module_config, sequencer_config).configure(module, sequencer)

        assert module.out0_offset_path0() == module_config.offset.out0_path0
        assert module.out0_offset_path1() == module_config.offset.out0_path1
        assert module.out1_offset_path0() == module_config.offset.out1_path0
        assert module.out1_offset_path1() == module_config.offset.out1_path1

    @pytest.mark.skip("Needs config implementation for QRM")
    @pytest.mark.parametrize("phase_offset", test_values.phase_offsets)
    @pytest.mark.parametrize("gain_ratio", test_values.gain_ratios)
    @pytest.mark.parametrize("i_offset", test_values.qrm_i_offsets)
    @pytest.mark.parametrize("q_offset", test_values.qrm_q_offsets)
    def test_qrm_mixer_config(
        self, qblox_resource, phase_offset, gain_ratio, i_offset, q_offset
    ):
        module_config = ModuleConfig()
        sequencer_config = SequencerConfig()
        module, sequencer = qblox_resource(ClusterType.CLUSTER_QRM)

        self.setup_qrm_mixer_config(module_config, i_offset, q_offset)
        self.setup_sequencer_mixer_config(sequencer_config, phase_offset, gain_ratio)
        QrmConfigHelper(module_config, sequencer_config).configure(module, sequencer)

        assert module.out0_offset() == module_config.offset.out0
        assert module.out1_offset() == module_config.offset.out1
        assert module.in0_offset() == module_config.offset.in0
        assert module.in1_offset() == module_config.offset.in1

    @pytest.mark.parametrize("phase_offset", test_values.phase_offsets)
    @pytest.mark.parametrize("gain_ratio", test_values.gain_ratios)
    @pytest.mark.parametrize("i_offset", test_values.qrm_rf_i_offsets)
    @pytest.mark.parametrize("q_offset", test_values.qrm_rf_q_offsets)
    def test_qrm_rf_mixer_config(
        self, qblox_resource, phase_offset, gain_ratio, i_offset, q_offset
    ):
        module_config = ModuleConfig()
        sequencer_config = SequencerConfig()
        module, sequencer = qblox_resource(ClusterType.CLUSTER_QRM_RF)

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


@pytest.mark.parametrize("qblox_resource", [None], indirect=True)
class TestAcqConfig(TestQbloxConfigMixin):
    test_values = AcqTestValues()

    @pytest.mark.parametrize("int_length", test_values.int_lengths)
    def test_qrm_rf_square_acq(self, qblox_resource, int_length):
        module_config = ModuleConfig()
        sequencer_config = SequencerConfig()
        module, sequencer = qblox_resource(ClusterType.CLUSTER_QRM_RF)

        sequencer_config.square_weight_acq.integration_length = int_length

        QrmRfConfigHelper(module_config, sequencer_config).configure_acq(sequencer)

        assert int_length == sequencer_config.square_weight_acq.integration_length
        assert (
            sequencer_config.square_weight_acq.integration_length
            == sequencer.integration_length_acq()
        )

    @pytest.mark.parametrize("rotation", test_values.rotations)
    @pytest.mark.parametrize("threshold", test_values.thresholds)
    def test_qrm_rf_thresholded_acq(self, qblox_resource, rotation, threshold):
        module_config = ModuleConfig()
        sequencer_config = SequencerConfig()
        module, sequencer = qblox_resource(ClusterType.CLUSTER_QRM_RF)

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


@pytest.mark.parametrize("qblox_resource", [None], indirect=True)
class TestAttenuationConfig(TestQbloxConfigMixin):
    test_values = AttenuationTestValues()

    @pytest.mark.parametrize("attenuation_values", test_values.qrc_attenuations)
    def test_qrc_attenuation_config(self, qblox_resource, attenuation_values):
        module_config = ModuleConfig()
        sequencer_config = SequencerConfig()
        module, sequencer = qblox_resource(ClusterType.CLUSTER_QRC)

        self.setup_qrc_attenuation_config(module_config, attenuation_values)
        QrcConfigHelper(module_config, sequencer_config).configure(module, sequencer)

        assert module.out0_att() == module_config.attenuation.out0
        assert module.out1_att() == module_config.attenuation.out1
        assert module.out2_att() == module_config.attenuation.out2
        assert module.out3_att() == module_config.attenuation.out3
        assert module.out4_att() == module_config.attenuation.out4
        assert module.out5_att() == module_config.attenuation.out5

        with pytest.raises(AssertionError):
            # TODO - QBlox bug: Dummy cluster fails to update input attenuation values: COMPILER-1052
            assert module.in0_att() == module_config.attenuation.in0
            assert module.in1_att() == module_config.attenuation.in1


@pytest.mark.parametrize("qblox_resource", [None], indirect=True)
class TestLatencyConfig(TestQbloxConfigMixin):
    test_values = LatencyTestValues()

    @pytest.mark.parametrize("latency", test_values.latencies)
    def test_qrc_latency_config(self, qblox_resource, latency):
        module_config = ModuleConfig()
        sequencer_config = SequencerConfig()
        module, sequencer = qblox_resource(ClusterType.CLUSTER_QRC)

        self.setup_qrc_latency_config(module_config, latency)

        with pytest.raises(NotImplementedError):
            # TODO - latency commands fail on both live and dummy cluster: COMPILER-1054
            QrcConfigHelper(module_config, sequencer_config).configure(module, sequencer)

        with pytest.raises(AssertionError):
            # TODO - latency commands fail on both live and dummy cluster: COMPILER-1054
            assert module.out0_latency() == module_config.latency.out0
            assert module.out1_latency() == module_config.latency.out1
            assert module.out2_latency() == module_config.latency.out2
            assert module.out3_latency() == module_config.latency.out3
            assert module.out4_latency() == module_config.latency.out4
            assert module.out5_latency() == module_config.latency.out5


@pytest.mark.parametrize("qblox_resource", [None], indirect=True)
class TestLoConfig(TestQbloxConfigMixin):
    test_values = LoTestValues()

    @pytest.mark.parametrize("lo_frequency", test_values.qrc_lo_freqs)
    def test_qrc_lo_config(self, qblox_resource, lo_frequency):
        module_config = ModuleConfig()
        sequencer_config = SequencerConfig()
        module, sequencer = qblox_resource(ClusterType.CLUSTER_QRC)

        self.setup_qrc_lo_config(module_config, lo_frequency)

        QrcConfigHelper(module_config, sequencer_config).configure(module, sequencer)

        assert module.out0_in0_lo_freq() == module_config.lo.out0_in0_freq
        assert module.out1_in1_lo_freq() == module_config.lo.out1_in1_freq

        assert module.out2_lo_freq() == module_config.lo.out2_freq
        assert module.out3_lo_freq() == module_config.lo.out3_freq
        assert module.out4_lo_freq() == module_config.lo.out4_freq
        assert module.out5_lo_freq() == module_config.lo.out5_freq
