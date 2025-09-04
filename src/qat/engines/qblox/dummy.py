import numpy as np
import regex
from qblox_instruments import (
    Cluster,
    DummyBinnedAcquisitionData,
    DummyScopeAcquisitionData,
    Sequencer,
)

from qat.backend.qblox.config.constants import Constants
from qat.backend.qblox.execution import QbloxExecutable
from qat.backend.qblox.ir import Sequence
from qat.engines.qblox.live import QbloxLeafInstrument
from qat.purr.backends.qblox.live import QbloxLiveHardwareModel
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class DummyQbloxInstrument(QbloxLeafInstrument):
    shot_pattern = regex.compile("jlt( +)R([0-9]+),([0-9]+),@(.*)\n")

    def __init__(self, id, name, address, dummy_config):
        super().__init__(id, name, address)
        self.dummy_config = dummy_config

    def _setup_dummy_scope_acq_data(self, module, sequencer: Sequencer, sequence: Sequence):
        shot_match = next(self.shot_pattern.finditer(sequence.program), None)
        avg_count = int(shot_match.group(3)) if shot_match else 1

        dummy_data = np.random.random(
            size=(Constants.MAX_SAMPLE_SIZE_SCOPE_ACQUISITIONS, 2)
        )
        dummy_data = [(iq[0], iq[1]) for iq in dummy_data]
        dummy_scope_acquisition_data = DummyScopeAcquisitionData(
            data=dummy_data, out_of_range=(False, False), avg_cnt=(avg_count, avg_count)
        )
        module.set_dummy_scope_acquisition_data(
            sequencer=sequencer.seq_idx, data=dummy_scope_acquisition_data
        )

    def _setup_dummy_binned_acq_data(
        self, module, sequencer: Sequencer, sequence: Sequence
    ):
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
            module.set_dummy_binned_acquisition_data(
                sequencer=sequencer.seq_idx,
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
            self._connected_modules = self.driver.get_connected_modules()

        log.info(self._driver.get_system_status())
        self.is_connected = True

    def setup(self, executable: QbloxExecutable, model: QbloxLiveHardwareModel):
        super().setup(executable, model)

        # Stage Scope and Acquisition data
        for module, allocations in self._resources.items():
            if module.is_qrm_type:
                for target, sequencer in allocations.items():
                    package = next(
                        (
                            pkg
                            for channel_id, pkg in executable.packages.items()
                            if target.full_id() == channel_id
                        )
                    )
                    self._setup_dummy_scope_acq_data(module, sequencer, package.sequence)
                    self._setup_dummy_binned_acq_data(module, sequencer, package.sequence)
