# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import numpy as np
import regex
from qblox_instruments import (
    Cluster,
    DummyBinnedAcquisitionData,
    DummyScopeAcquisitionData,
    Sequencer,
)

from qat.backend.qblox.config.constants import Constants
from qat.backend.qblox.execution import QbloxProgram
from qat.backend.qblox.ir import Sequence
from qat.engines.qblox.live import QbloxLeafInstrument
from qat.purr.compiler.devices import ChannelType
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class DummyQbloxInstrument(QbloxLeafInstrument):
    shot_pattern = regex.compile("jlt( +)R([0-9]+),([0-9]+),@(.*)\n")

    def __init__(self, id, name, address, dummy_config):
        super().__init__(id, name, address)
        self.dummy_config = dummy_config

    def _setup_dummy_scope_acq_data(self, sequencer: Sequencer, sequence: Sequence):
        shot_match = next(self.shot_pattern.finditer(sequence.program), None)
        avg_count = int(shot_match.group(3)) if shot_match else 1

        dummy_data = np.random.random(
            size=(Constants.MAX_SAMPLE_SIZE_SCOPE_ACQUISITIONS, 2)
        )
        dummy_data = [(iq[0], iq[1]) for iq in dummy_data]
        dummy_scope_acquisition_data = DummyScopeAcquisitionData(
            data=dummy_data, out_of_range=(False, False), avg_cnt=(avg_count, avg_count)
        )
        sequencer.set_dummy_scope_acquisition_data(data=dummy_scope_acquisition_data)

    def _setup_dummy_binned_acq_data(self, sequencer: Sequencer, sequence: Sequence):
        shot_match = next(self.shot_pattern.finditer(sequence.program), None)
        avg_count = int(shot_match.group(3)) if shot_match else 1

        for name, acquisition in sequence.acquisitions.items():
            dummy_binned_acquisition_data = [
                DummyBinnedAcquisitionData(
                    data=(np.random.random(), np.random.random()),
                    thres=np.random.choice(2),
                    avg_cnt=avg_count,
                )
            ] * acquisition["num_bins"]
            sequencer.set_dummy_binned_acquisition_data(
                acq_index_name=name,
                data=dummy_binned_acquisition_data,
            )

    def _delete_acquisitions(self, sequencer):
        sequencer.delete_dummy_scope_acquisition_data()
        sequencer.delete_dummy_binned_acquisition_data()

    def connect(self):
        if self._driver is None or not Cluster.is_valid(self._driver):
            self._driver: Cluster = Cluster(name=self.name, dummy_cfg=self.dummy_config)
            self._driver.reset()
            self._modules = {m: True for m in self._driver.get_connected_modules().values()}
            self.is_connected = True

        log.info(self._driver.get_system_status())

    def setup(self, program: QbloxProgram):
        super().setup(program)

        # Stage Scope and Acquisition data
        for pulse_channel_id, sequencer in self._id2seq.items():
            if ChannelType.macq.name in pulse_channel_id:
                package = program.packages[pulse_channel_id]
                self._setup_dummy_scope_acq_data(sequencer, package.sequence)
                self._setup_dummy_binned_acq_data(sequencer, package.sequence)
