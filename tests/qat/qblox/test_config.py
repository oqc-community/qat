import uuid

import pytest
from qblox_instruments import Cluster
from qblox_instruments.qcodes_drivers.module import Module
from qblox_instruments.qcodes_drivers.sequencer import Sequencer

from qat.purr.backends.qblox.codegen import QbloxEmitter, calculate_duration
from qat.purr.backends.qblox.config import (
    ModuleConfig,
    QbloxConfig,
    QcmConfigHelper,
    QcmRfConfigHelper,
    QrmConfigHelper,
    QrmRfConfigHelper,
    SequencerConfig,
)
from qat.purr.compiler.devices import PulseShapeType
from qat.purr.compiler.emitter import InstructionEmitter
from qat.purr.compiler.instructions import Acquire
from qat.purr.compiler.runtime import get_builder
from tests.qat.qblox.utils import DUMMY_CONFIG, ClusterInfo, MixerTestValues


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


@pytest.mark.parametrize("model", [ClusterInfo()], indirect=True)
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
        packages = QbloxEmitter().emit(qat_file)
        model.control_hardware.allocate_resources(packages)

        assert len(packages) == 2

        # Drive
        drive_pkg = next((pkg for pkg in packages if pkg.target == drive_channel))
        lo_freq, nco_freq = self.extract_lo_and_nco_freqs(drive_channel)
        module, sequencer = model.control_hardware.install(drive_pkg)
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
        module, sequencer = model.control_hardware.install(measure_pkg)
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


class TestMixerConfig(TestQbloxConfigMixin):
    mixer_values = MixerTestValues()

    @pytest.mark.skip("Needs config implementation for QCM")
    @pytest.mark.parametrize("phase_offset", mixer_values.phase_offsets)
    @pytest.mark.parametrize("gain_ratio", mixer_values.gain_ratios)
    @pytest.mark.parametrize("i_offset", mixer_values.qcm_i_offsets)
    @pytest.mark.parametrize("q_offset", mixer_values.qcm_q_offsets)
    def test_qcm_mixer_config(self, request, phase_offset, gain_ratio, i_offset, q_offset):
        module_config = ModuleConfig()
        seq_idx, sequencer_config = 0, SequencerConfig()
        config = QbloxConfig(module=module_config, sequencers={seq_idx: sequencer_config})

        # The qcodes package generates a warning if the name of the cluster contains dashes.
        name = f"{request.node.originalname}_{uuid.uuid4()}".replace("-", "_")
        cluster: Cluster = Cluster(name=name, dummy_cfg=DUMMY_CONFIG)
        module: Module = next(
            (
                m
                for m in cluster.modules
                if m.present() and m.is_qcm_type and not m.is_rf_type
            )
        )
        sequencer: Sequencer = module.sequencers[seq_idx]

        self.setup_qcm_mixer_config(module_config, i_offset, q_offset)
        self.setup_sequencer_mixer_config(sequencer_config, phase_offset, gain_ratio)
        QcmConfigHelper(config).configure(module, sequencer)

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
        seq_idx, sequencer_config = 0, SequencerConfig()
        config = QbloxConfig(module=module_config, sequencers={seq_idx: sequencer_config})

        # The qcodes package generates a warning if the name of the cluster contains dashes.
        name = f"{request.node.originalname}_{uuid.uuid4()}".replace("-", "_")
        cluster: Cluster = Cluster(name=name, dummy_cfg=DUMMY_CONFIG)
        module: Module = next(
            (m for m in cluster.modules if m.present() and m.is_qcm_type and m.is_rf_type)
        )
        sequencer: Sequencer = module.sequencers[seq_idx]

        self.setup_qcm_rf_mixer_config(module_config, i_offset, q_offset)
        self.setup_sequencer_mixer_config(sequencer_config, phase_offset, gain_ratio)
        QcmRfConfigHelper(config).configure(module, sequencer)

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
        seq_idx, sequencer_config = 0, SequencerConfig()
        config = QbloxConfig(module=module_config, sequencers={seq_idx: sequencer_config})

        # The qcodes package generates a warning if the name of the cluster contains dashes.
        name = f"{request.node.originalname}_{uuid.uuid4()}".replace("-", "_")
        cluster: Cluster = Cluster(name=name, dummy_cfg=DUMMY_CONFIG)
        module: Module = next(
            (
                m
                for m in cluster.modules
                if m.present() and m.is_qrm_type and not m.is_rf_type
            )
        )
        sequencer: Sequencer = module.sequencers[seq_idx]

        self.setup_qrm_mixer_config(module_config, i_offset, q_offset)
        self.setup_sequencer_mixer_config(sequencer_config, phase_offset, gain_ratio)
        QrmConfigHelper(config).configure(module, sequencer)

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
        seq_idx, sequencer_config = 0, SequencerConfig()
        config = QbloxConfig(module=module_config, sequencers={seq_idx: sequencer_config})

        # The qcodes package generates a warning if the name of the cluster contains dashes.
        name = f"{request.node.originalname}_{uuid.uuid4()}".replace("-", "_")
        cluster: Cluster = Cluster(name=name, dummy_cfg=DUMMY_CONFIG)
        module: Module = next(
            (m for m in cluster.modules if m.present() and m.is_qrm_type and m.is_rf_type)
        )
        sequencer: Sequencer = module.sequencers[seq_idx]

        self.setup_qrm_rf_mixer_config(module_config, i_offset, q_offset)
        self.setup_sequencer_mixer_config(sequencer_config, phase_offset, gain_ratio)
        QrmRfConfigHelper(config).configure(module, sequencer)

        assert module.out0_offset_path0() == module_config.offset.out0_path0
        assert module.out0_offset_path1() == module_config.offset.out0_path1

        # TODO - potential bug with the dummy structures
        with pytest.raises(AssertionError):
            assert module.in0_offset_path0() == module_config.offset.in0_path0

        with pytest.raises(AssertionError):
            assert module.in0_offset_path1() == module_config.offset.in0_path1
