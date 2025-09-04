# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from collections import defaultdict
from dataclasses import asdict
from typing import Dict, List, Union

from qblox_instruments import Cluster
from qblox_instruments.qcodes_drivers.module import Module
from qblox_instruments.qcodes_drivers.sequencer import Sequencer

from qat.backend.passes.purr.analysis import TILegalisationPass
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
from qat.engines.qblox.instrument import CompositeInstrument, LeafInstrument
from qat.purr.backends.qblox.live import QbloxLiveHardwareModel
from qat.purr.compiler.devices import (
    PulseChannel,
)
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class QbloxLeafInstrument(LeafInstrument):
    def __init__(
        self,
        id: str,
        name: str,
        address: str = None,
    ):
        super().__init__(id=id, name=name, address=address)
        self.managed_mode = False

        self._driver: Cluster = None
        self._connected_modules: Dict[int, Module] = {}
        self._resources: Dict[Module, Dict[PulseChannel, Sequencer]] = {}

    @property
    def driver(self):
        return self._driver

    @property
    def connected_modules(self):
        return self._connected_modules

    @property
    def resources(self):
        return self._resources

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

        modules = self._resources.keys() or self._connected_modules.values()

        for m in modules:
            log.debug(f"Resetting sequencer connections for module {m.slot_idx}")
            m.disconnect_outputs()
            if m.is_qrm_type:
                m.disconnect_inputs()

    def connect(self):
        if self._driver is None or not Cluster.is_valid(self._driver):
            self._driver: Cluster = Cluster(name=self.name, identifier=self.address)
            self._driver.reset()
            self._connected_modules = self.driver.get_connected_modules()

        log.info(self._driver.get_system_status())
        self.is_connected = True

    def disconnect(self):
        if self._driver is not None:
            try:
                self._driver.close()
                self._driver = None
                self.is_connected = False
            except BaseException as e:
                log.warning(
                    f"Failed to close instrument ID: {self.id} at: {self.address}\n{str(e)}"
                )

    def setup(self, executable: QbloxExecutable, model: QbloxLiveHardwareModel):
        try:
            self._resources.clear()
            self._reset_connections()
            for channel_id, pkg in executable.packages.items():
                target: PulseChannel = model.get_pulse_channel_from_id(channel_id)
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
        finally:
            for module, allocations in self._resources.items():
                for target, sequencer in allocations.items():
                    sequencer.sync_en(False)

    def collect(self):
        if not any(self._resources):
            raise ValueError(
                "No resources allocated. Install packages first, and then run playback"
            )

        results: Dict[PulseChannel, List[Acquisition]] = defaultdict(list)
        try:
            for module, allocations in self._resources.items():
                if module.is_qrm_type:
                    for target, sequencer in allocations.items():
                        status_obj = sequencer.get_sequencer_status()
                        log.debug(f"Sequencer status - {sequencer}: {status_obj}")
                        if acquisitions := sequencer.get_acquisitions():
                            # TODO - 60 min tops, make it dynamic by involving flow-aware timeline duration
                            # Only wait if you sequencer is expected to have acquisitions
                            # TODO - Precise expectation of acquisitions should come from higher up
                            sequencer.get_acquisition_status(timeout=60)

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

                            scope_data = acquisition.acquisition.scope
                            scope_data.path0.data = scope_data.path0.data[start:end]
                            scope_data.path1.data = scope_data.path1.data[start:end]

                            integ_data = acquisition.acquisition.bins.integration
                            integ_data.path0 /= integ_length
                            integ_data.path1 /= integ_length

                            results[target].append(acquisition)

                        sequencer.delete_acquisition_data(all=True)

            return results
        finally:
            self._resources.clear()
            self._reset_connections()


class QbloxCompositeInstrument(CompositeInstrument):
    """
    For daisy-chained Qblox chassis.
    """

    pass


LiveQbloxInstrument = Union[QbloxLeafInstrument, QbloxCompositeInstrument]
