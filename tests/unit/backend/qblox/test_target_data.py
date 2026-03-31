import pytest

from qat.backend.qblox.target_data import (
    ControlSequencerDescription,
    ModuleDescription,
    Q1asmDescription,
    QcmDescription,
    QcmRfDescription,
    QrcDescription,
    QrmDescription,
    QrmRfDescription,
    ReadoutSequencerDescription,
    SequencerDescription,
)


class TestSpecifiedTargetDataFields:
    @pytest.fixture(scope="class")
    def default_q1asm_data(self):
        return Q1asmDescription()

    @pytest.fixture(scope="class")
    def default_sequencer_data(self):
        return SequencerDescription()

    @pytest.fixture(scope="class")
    def default_control_sequencer_data(self):
        return ControlSequencerDescription()

    @pytest.fixture(scope="class")
    def default_readout_sequencer_data(self):
        return ReadoutSequencerDescription()

    @pytest.fixture(scope="class")
    def default_module_description(self):
        return ModuleDescription()

    @pytest.fixture(scope="class")
    def default_qcm_description(self):
        return QcmDescription()

    @pytest.fixture(scope="class")
    def default_qcm_rf_description(self):
        return QcmRfDescription()

    @pytest.fixture(scope="class")
    def default_qrm_description(self):
        return QrmDescription()

    @pytest.fixture(scope="class")
    def default_qrm_rf_description(self):
        return QrmRfDescription()

    def test_q1asm_data(self, default_q1asm_data):
        assert default_q1asm_data.min_gain == -pow(2, 15)
        assert default_q1asm_data.max_gain == pow(2, 15) - 1
        assert default_q1asm_data.min_offset == -pow(2, 15)
        assert default_q1asm_data.max_offset == pow(2, 15) - 1
        assert default_q1asm_data.max_wait_time == pow(2, 16) - 4
        assert default_q1asm_data.register_size == pow(2, 32) - 1
        assert default_q1asm_data.loop_unroll_threshold == 4

    def test_sequencer_data(self, default_sequencer_data):
        assert default_sequencer_data.grid_time == 4
        assert default_sequencer_data.nco_min_freq == -500e6
        assert default_sequencer_data.nco_max_freq == 500e6
        assert default_sequencer_data.nco_max_phase_steps == int(1e9)
        assert default_sequencer_data.nco_phase_steps_per_deg == 1e9 / 360
        assert default_sequencer_data.nco_freq_steps_per_hz == 4
        assert default_sequencer_data.nco_freq_limit_steps == 2e9
        assert default_sequencer_data.number_of_registers == 64
        assert default_sequencer_data.min_acq_integration_length == 4
        assert default_sequencer_data.max_acq_integration_length == (1 << 24) - 4
        assert default_sequencer_data.min_acq_threshold == -((1 << 24) - 4)
        assert default_sequencer_data.max_acq_threshold == (1 << 24) - 4
        assert default_sequencer_data.max_sample_size_waveforms == 16384
        assert default_sequencer_data.max_num_instructions == 12288

    def test_control_sequencer_data(self, default_control_sequencer_data):
        assert default_control_sequencer_data.grid_time == 4
        assert default_control_sequencer_data.nco_min_freq == -500e6
        assert default_control_sequencer_data.nco_max_freq == 500e6
        assert default_control_sequencer_data.nco_max_phase_steps == int(1e9)
        assert default_control_sequencer_data.nco_phase_steps_per_deg == 1e9 / 360
        assert default_control_sequencer_data.nco_freq_steps_per_hz == 4
        assert default_control_sequencer_data.nco_freq_limit_steps == 2e9
        assert default_control_sequencer_data.number_of_registers == 64
        assert default_control_sequencer_data.min_acq_integration_length == 4
        assert default_control_sequencer_data.max_acq_integration_length == (1 << 24) - 4
        assert default_control_sequencer_data.min_acq_threshold == -((1 << 24) - 4)
        assert default_control_sequencer_data.max_acq_threshold == (1 << 24) - 4
        assert default_control_sequencer_data.max_sample_size_waveforms == 16384
        assert default_control_sequencer_data.max_num_instructions == 16384

    def test_readout_sequencer_data(self, default_readout_sequencer_data):
        assert default_readout_sequencer_data.grid_time == 4
        assert default_readout_sequencer_data.nco_min_freq == -500e6
        assert default_readout_sequencer_data.nco_max_freq == 500e6
        assert default_readout_sequencer_data.nco_max_phase_steps == int(1e9)
        assert default_readout_sequencer_data.nco_phase_steps_per_deg == 1e9 / 360
        assert default_readout_sequencer_data.nco_freq_steps_per_hz == 4
        assert default_readout_sequencer_data.nco_freq_limit_steps == 2e9
        assert default_readout_sequencer_data.number_of_registers == 64
        assert default_readout_sequencer_data.min_acq_integration_length == 4
        assert default_readout_sequencer_data.max_acq_integration_length == (1 << 24) - 4
        assert default_readout_sequencer_data.min_acq_threshold == -((1 << 24) - 4)
        assert default_readout_sequencer_data.max_acq_threshold == (1 << 24) - 4
        assert default_readout_sequencer_data.max_sample_size_waveforms == 16384
        assert default_readout_sequencer_data.max_num_instructions == 12288

    def test_module_description(self, default_module_description):
        assert default_module_description.number_of_sequencers == 6

    def test_qcm_description(self, default_qcm_description):
        assert default_qcm_description.number_of_sequencers == 6
        assert default_qcm_description.min_qcm_offset_v == -2.5
        assert default_qcm_description.max_qcm_offset_v == 2.5

    def test_qcm_rf_description(self, default_qcm_rf_description):
        assert default_qcm_rf_description.number_of_sequencers == 6
        assert default_qcm_rf_description.min_qcm_rf_offset_mv == -84
        assert default_qcm_rf_description.max_qcm_rf_offset_mv == 73
        assert default_qcm_rf_description.min_out_att_db == 0
        assert default_qcm_rf_description.max_out_att_db == 60

    def test_qrm_description(self, default_qrm_description):
        assert default_qrm_description.number_of_sequencers == 6
        assert default_qrm_description.min_sample_size_scope_acquisitions == 4
        assert default_qrm_description.max_sample_size_scope_acquisitions == 16384
        assert default_qrm_description.max_binned_acquisitions == 3_000_000
        assert default_qrm_description.min_qrm_offset_v == -0.09
        assert default_qrm_description.max_qrm_offset_v == 0.09

    def test_qrm_rf_description(self, default_qrm_rf_description):
        assert default_qrm_rf_description.number_of_sequencers == 6
        assert default_qrm_rf_description.min_sample_size_scope_acquisitions == 4
        assert default_qrm_rf_description.max_sample_size_scope_acquisitions == 16384
        assert default_qrm_rf_description.max_binned_acquisitions == 3_000_000
        assert default_qrm_rf_description.min_qrm_rf_offset_v == -0.09
        assert default_qrm_rf_description.max_qrm_rf_offset_v == 0.09
        assert default_qrm_rf_description.min_out_att_db == 0
        assert default_qrm_rf_description.max_out_att_db == 60
        assert default_qrm_rf_description.min_in_att_db == 0
        assert default_qrm_rf_description.max_in_att_db == 30

    def test_qrc_description(self):
        default_qrc_description = QrcDescription()
        assert default_qrc_description.number_of_sequencers == 12
        assert default_qrc_description.number_of_readout_sequencers == 8
        assert default_qrc_description.number_of_control_sequencers == 4
        assert default_qrc_description.min_sample_size_scope_acquisitions == 4
        assert default_qrc_description.max_sample_size_scope_acquisitions == 16384
        assert default_qrc_description.max_binned_acquisitions == 3_000_000
        assert default_qrc_description.min_out_att_db == 0.0
        assert default_qrc_description.max_out_att_db == 31.5
        assert default_qrc_description.min_in_att_db == 0.0
        assert default_qrc_description.max_in_att_db == 31.5
        assert default_qrc_description.output_connections == {
            0: list(range(8)),
            1: list(range(8)),
            2: [0, 4] + list(range(8, 12)),
            3: [1, 5] + list(range(8, 12)),
            4: [2, 6] + list(range(8, 12)),
            5: [3, 7] + list(range(8, 12)),
        }
        assert default_qrc_description.input_connections == {
            0: list(range(8)),
            1: list(range(8)),
        }
