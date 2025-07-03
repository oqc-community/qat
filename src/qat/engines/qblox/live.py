# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import json
import os
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from functools import reduce
from itertools import groupby
from typing import Dict, List

from qblox_instruments import Cluster
from qblox_instruments.qcodes_drivers.module import Module
from qblox_instruments.qcodes_drivers.sequencer import Sequencer

from qat.backend.passes.legacy.analysis import TILegalisationPass
from qat.backend.qblox.acquisition import Acquisition
from qat.backend.qblox.codegen import QbloxPackage
from qat.backend.qblox.config.constants import Constants
from qat.backend.qblox.config.helpers import (
    QcmConfigHelper,
    QcmRfConfigHelper,
    QrmConfigHelper,
    QrmRfConfigHelper,
)
from qat.backend.qblox.config.specification import (
    ModuleConfig,
    SequencerConfig,
)
from qat.backend.qblox.execution import QbloxExecutable
from qat.backend.qblox.visualisation import plot_executable
from qat.engines import NativeEngine
from qat.engines.qblox.instrument_base import CompositeInstrument, LeafInstrument
from qat.purr.backends.qblox.visualisation import plot_playback
from qat.purr.compiler.devices import (
    PulseChannel,
)
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class QbloxInstrument(LeafInstrument):
    _driver: Cluster

    def __init__(
        self,
        dev_id: str,
        name: str,
        address: str = None,
        dummy_cfg: Dict = None,
    ):
        super().__init__(id=dev_id, name=name, address=address)
        self.dummy_cfg = dummy_cfg

        self.managed_mode = False
        self.dump_sequence = False
        self.plot_packages = False
        self.plot_acquisitions = False

        self._resources: Dict[Module, Dict[PulseChannel, Sequencer]] = {}

    def allocate_resources(self, target: PulseChannel):
        module_id = target.physical_channel.slot_idx - 1  # slot_idx is in range [1..20]
        module: Module = self._driver.modules[module_id]
        allocations = self._resources.setdefault(module, {})
        sequencer = allocations.get(target, None)
        if not sequencer:
            total = set(target.physical_channel.config.sequencers.keys())
            allocated = set([seq.seq_idx for seq in allocations.values()])

            available = total - allocated
            if not available:
                raise ValueError(f"No more available sequencers on module {module}")
            sequencer: Sequencer = module.sequencers[next(iter(available))]
            allocations[target] = sequencer

        log.debug(
            f"Sequencer {sequencer.seq_idx} in Module {module.slot_idx} will be running {target}"
        )
        return module, sequencer

    def configure(
        self,
        target: PulseChannel,
        package: QbloxPackage,
        module: Module,
        sequencer: Sequencer,
    ):
        lo_freq, nco_freq = TILegalisationPass.decompose_freq(target.frequency, target)

        qblox_config = target.physical_channel.config

        # Customise Module config
        module_config: ModuleConfig = qblox_config.module
        if module_config.lo.out0_en:
            module_config.lo.out0_freq = lo_freq
        if module_config.lo.out1_en:
            module_config.lo.out1_freq = lo_freq
        if module_config.lo.out0_in0_en:
            module_config.lo.out0_in0_freq = lo_freq

        module_config.scope_acq.sequencer_select = sequencer.seq_idx
        module_config.scope_acq.trigger_mode_path0 = "sequencer"
        module_config.scope_acq.avg_mode_en_path0 = True
        module_config.scope_acq.trigger_mode_path1 = "sequencer"
        module_config.scope_acq.avg_mode_en_path1 = True

        # Customise Sequencer config
        sequencer_config: SequencerConfig = qblox_config.sequencers[sequencer.seq_idx]
        sequencer_config.nco.freq = nco_freq
        sequencer_config.nco.prop_delay_comp_en = True
        sequencer_config.square_weight_acq.integration_length = (
            package.sequencer_config.square_weight_acq.integration_length
        )
        sequencer_config.thresholded_acq.rotation = (
            package.sequencer_config.thresholded_acq.rotation
        )
        sequencer_config.thresholded_acq.threshold = (
            package.sequencer_config.thresholded_acq.threshold
        )

        log.debug(f"Configuring module {module}, sequencer {sequencer}")
        if module.is_qcm_type:
            if module.is_rf_type:
                config_helper = QcmRfConfigHelper(module_config, sequencer_config)
            else:
                config_helper = QcmConfigHelper(module_config, sequencer_config)
        elif module.is_qrm_type:
            if module.is_rf_type:
                config_helper = QrmRfConfigHelper(module_config, sequencer_config)
            else:
                config_helper = QrmConfigHelper(module_config, sequencer_config)
        else:
            raise ValueError(f"Unknown module type {module.module_type}")

        config_helper.configure(module, sequencer)

        if self.managed_mode:
            log.info("Managed mode enabled")
            config_helper.calibrate_mixer(module, sequencer)

    def _reset_connections(self):
        # TODO - Qblox bug: Hard reset clutters sequencer connections with conflicting defaults
        # TODO - This is a temporary workaround until Qblox fixes the issue

        modules = self._resources.keys() or self._driver.get_connected_modules().values()

        for m in modules:
            log.debug(f"Resetting sequencer connections for module {m.slot_idx}")
            m.disconnect_outputs()
            if m.is_qrm_type:
                m.disconnect_inputs()

    def connect(self):
        if self._driver is None or not Cluster.is_valid(self._driver):
            self._driver: Cluster = Cluster(
                name=self.name,
                identifier=self.address,
                dummy_cfg=self.dummy_cfg if self.address is None else None,
            )
            self._driver.reset()
            self._reset_connections()

        log.info(self._driver.get_system_status())

    def disconnect(self):
        if self._driver is not None:
            try:
                self._driver.close()
            except BaseException as e:
                log.warning(
                    f"Failed to close instrument ID: {self.id} at: {self.address}\n{str(e)}"
                )

    def setup(self, executable: QbloxExecutable):
        if self.plot_packages:
            plot_executable(executable)

        packages: Dict[PulseChannel, QbloxPackage] = executable.packages

        if self.dump_sequence:
            for target, pkg in packages.items():
                filename = f"schedules/target_{target.id}_@_{datetime.now().strftime('%m-%d-%Y_%H%M%S')}.json"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, "w") as f:
                    f.write(json.dumps(asdict(pkg.sequence)))

        try:
            self._resources.clear()
            for target, pkg in packages.items():
                module, sequencer = self.allocate_resources(target)
                self.configure(target, pkg, module, sequencer)
                sequence = asdict(pkg.sequence)
                log.debug(f"Uploading sequence to {module}, sequencer {sequencer}")
                sequencer.sequence(sequence)
        except BaseException as e:
            self._reset_connections()
            self._resources.clear()
            raise e

    def playback(self):
        if not any(self._resources):
            raise ValueError("No resources allocated. Install packages first")

        results: Dict[PulseChannel, List[Acquisition]] = defaultdict(list)
        try:
            for module, allocations in self._resources.items():
                for target, sequencer in allocations.items():
                    sequencer.sync_en(True)

            for module, allocations in self._resources.items():
                for target, sequencer in allocations.items():
                    sequencer.arm_sequencer()

            for module, allocations in self._resources.items():
                for target, sequencer in allocations.items():
                    sequencer.start_sequencer()

            for module, allocations in self._resources.items():
                if module.is_qrm_type:
                    for target, sequencer in allocations.items():
                        # TODO - 60 min tops, make it dynamic by involving flow-aware timeline duration
                        sequencer.get_acquisition_status(timeout=60)
                        acquisitions = sequencer.get_acquisitions()

                        for name in acquisitions:
                            sequencer.store_scope_acquisition(name)

                        # (re)fetch the lot after having stored scope acquisition
                        acquisitions = sequencer.get_acquisitions()

                        integ_length = sequencer.integration_length_acq()
                        start, end = (
                            0,
                            min(
                                integ_length,
                                Constants.MAX_SAMPLE_SIZE_SCOPE_ACQUISITIONS,
                            ),
                        )
                        for name, acquisition in acquisitions.items():
                            acquisition = Acquisition.model_validate(acquisition)
                            acquisition.name = name

                            scope_data = acquisition.acq_data.scope
                            scope_data.i.data = scope_data.i.data[start:end]
                            scope_data.q.data = scope_data.q.data[start:end]

                            integ_data = acquisition.acq_data.bins.integration
                            integ_data.i /= integ_length
                            integ_data.q /= integ_length

                            results[target].append(acquisition)

                        sequencer.delete_acquisition_data(all=True)

            if self.plot_acquisitions:
                plot_playback(results)

            return results
        finally:
            for module, allocations in self._resources.items():
                for target, sequencer in allocations.items():
                    sequencer.sync_en(False)

            self._reset_connections()
            self._resources.clear()


class CompositeQbloxInstrument(CompositeInstrument):
    """
    For daisy-chained Qblox chassis.
    """

    pass


class QbloxEngine(NativeEngine):
    def __init__(self, instrument: QbloxInstrument | CompositeQbloxInstrument):
        self.instrument: QbloxInstrument | CompositeQbloxInstrument = instrument

    @staticmethod
    def combine_playbacks(playbacks: Dict[PulseChannel, List[Acquisition]]):
        """
        Combines acquisition objects from multiple acquire instructions in multiple readout targets.
        Notice that :meth:`groupby` preserves (original) relative order, which makes it honour
        the (sequential) lexicographical order of the loop nest:

        playback[target]["acq_0"] contains (potentially) a list of acquisitions collected in the same
        order as the order in which the packages were sent to the FPGA.

        Although acquisition names are enough for unicity in practice, the playback's structure
        distinguishes different (multiple) acquisitions per readout target, thus making it more robust.
        """

        playback: Dict[PulseChannel, Dict[str, Acquisition]] = {}
        for target, acquisitions in playbacks.items():
            groups_by_name = groupby(acquisitions, lambda acquisition: acquisition.name)
            playback[target] = {
                name: reduce(
                    lambda acq1, acq2: Acquisition.accumulate(acq1, acq2),
                    acqs,
                    Acquisition(),
                )
                for name, acqs in groups_by_name
            }

        return playback

    def execute(
        self, executable: QbloxExecutable
    ) -> Dict[PulseChannel, Dict[str, Acquisition]]:
        packages = executable.packages

        self.instrument.setup(packages)
        playbacks: Dict[PulseChannel, List[Acquisition]] = self.instrument.playback()
        playback: Dict[PulseChannel, Dict[str, Acquisition]] = self.combine_playbacks(
            playbacks
        )
        return playback
