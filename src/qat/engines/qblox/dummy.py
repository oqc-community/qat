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

from qat.backend.qblox.execution import QbloxPackage, QbloxProgram
from qat.backend.qblox.target_data import QbloxTargetData
from qat.engines.qblox.live import QbloxLeafInstrument
from qat.purr.compiler.devices import ChannelType
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class DummyQbloxInstrument(QbloxLeafInstrument):
    shot_pattern = regex.compile("jlt( +)R([0-9]+),([0-9]+),@(.*)\n")
    target_data = QbloxTargetData()
    qrm_data = target_data.QRM_DATA

    def __init__(self, id, name, dummy_config):
        super().__init__(id, name, None)
        self.dummy_config = dummy_config

    def _setup_dummy_scope_acq_data(self, sequencer: Sequencer, package: QbloxPackage):
        shot_match = next(self.shot_pattern.finditer(package.sequence.program), None)
        avg_count = int(shot_match.group(3)) if shot_match else 1

        num_paths = 4
        paths = [
            np.random.random(size=self.qrm_data.max_sample_size_scope_acquisitions)
            / avg_count
            for _ in range(num_paths)
        ]
        dummy_scope_acquisition_data = DummyScopeAcquisitionData(
            data=zip(*paths),
            out_of_range=(False,) * num_paths,
            avg_cnt=(avg_count,) * num_paths,
        )
        sequencer.set_dummy_scope_acquisition_data(data=dummy_scope_acquisition_data)

    def _setup_dummy_binned_acq_data(self, sequencer: Sequencer, package: QbloxPackage):
        shot_match = next(self.shot_pattern.finditer(package.sequence.program), None)
        avg_count = int(shot_match.group(3)) if shot_match else 1

        for name, acquisition in package.sequence.acquisitions.items():
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
        # TODO - Follow up on acquisition cleanup inconsistencies: COMPILER-987
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
                if package.sequence.acquisitions:
                    self._setup_dummy_scope_acq_data(sequencer, package)
                    self._setup_dummy_binned_acq_data(sequencer, package)
