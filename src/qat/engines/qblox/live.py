# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from collections import defaultdict
from dataclasses import asdict

from qblox_instruments import Cluster
from qblox_instruments.qcodes_drivers.module import Module
from qblox_instruments.qcodes_drivers.sequencer import Sequencer

from qat.backend.qblox.acquisition import Acquisition
from qat.backend.qblox.codegen import QbloxPackage
from qat.backend.qblox.config.constants import Constants
from qat.backend.qblox.config.helpers import (
    QcmConfigHelper,
    QcmRfConfigHelper,
    QrmConfigHelper,
    QrmRfConfigHelper,
)
from qat.backend.qblox.execution import QbloxProgram
from qat.instrument.base import (
    CompositeInstrument,
    LeafInstrument,
)
from qat.purr.compiler.devices import ChannelType
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class QbloxLeafInstrument(LeafInstrument):
    def __init__(
        self,
        id: str,
        name: str,
        address: str = None,
        ref_source: str = None,
    ):
        super().__init__(id=id, name=name, address=address)
        self.ref_source = ref_source or "internal"

        self._driver: Cluster = None
        self._modules: dict[Module, bool] = {}
        self._id2seq: dict[str, Sequencer] = {}

    def _reset_modules(self):
        # TODO - Qblox bug: Hard reset clutters sequencer connections with conflicting defaults
        # TODO - This is a temporary workaround until Qblox fixes the issue

        modules = [mod for mod, is_dirty in self._modules.items() if is_dirty]

        for mod in modules:
            log.debug(f"Resetting sequencer connections for module {mod}")
            mod.disconnect_outputs()
            if mod.is_qrm_type:
                mod.disconnect_inputs()
            self._modules[mod] = False  # Mark as clean

    @property
    def driver(self):
        return self._driver

    @property
    def id2seq(self):
        return self._id2seq

    @property
    def modules(self):
        return self._modules

    def configure(self, package: QbloxPackage):
        module: Module = getattr(self._driver, f"module{package.slot_idx}")
        sequencer = module.sequencers[package.seq_idx]

        seq_config, mod_config = package.seq_config, package.mod_config
        if module.is_qcm_type:
            if module.is_rf_type:
                config_helper = QcmRfConfigHelper(mod_config, seq_config)
            else:
                config_helper = QcmConfigHelper(mod_config, seq_config)
        elif module.is_qrm_type:
            if module.is_rf_type:
                config_helper = QrmRfConfigHelper(mod_config, seq_config)
            else:
                config_helper = QrmConfigHelper(mod_config, seq_config)
        else:
            raise ValueError(f"Unknown module type {module.module_type}")

        try:
            log.debug(f"Configuring module {module}, and sequencer {sequencer}")
            config_helper.configure(module, sequencer)
            sequence = asdict(package.sequence)
            sequencer.sequence(sequence)
        finally:
            self._modules[module] = True  # Mark as dirty
            self._id2seq[package.pulse_channel_id] = sequencer

    def connect(self):
        if self._driver is None or not Cluster.is_valid(self._driver):
            self._driver: Cluster = Cluster(name=self.name, identifier=self.address)
            self._driver.reset()
            self._driver.reference_source(self.ref_source)
            self._modules = {m: True for m in self._driver.get_connected_modules().values()}
            self.is_connected = True

        log.info(self._driver.get_system_status())

    def disconnect(self):
        if self._driver is not None:
            try:
                self._driver.close()
                self._driver = None
                self._modules.clear()
                self.is_connected = False
            except BaseException as e:
                log.warning(
                    f"Failed to close instrument ID: {self.id} at: {self.address}\n{str(e)}"
                )

    def setup(self, program: QbloxProgram):
        try:
            self._id2seq.clear()
            self._reset_modules()
            for pkg in program.packages.values():
                self.configure(pkg)
        except BaseException as e:
            self._id2seq.clear()
            self._reset_modules()
            raise e

    def playback(self):
        if not any(self._id2seq):
            raise ValueError("No allocations found. Install packages and configure first")

        results: dict[str, list[Acquisition]] = defaultdict(list)
        try:
            for pulse_channel_id, sequencer in self._id2seq.items():
                sequencer.sync_en(True)

            for pulse_channel_id, sequencer in self._id2seq.items():
                sequencer.arm_sequencer()

            for pulse_channel_id, sequencer in self._id2seq.items():
                sequencer.start_sequencer()

            for pulse_channel_id, sequencer in self._id2seq.items():
                if ChannelType.macq.name in pulse_channel_id:
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

                        results[pulse_channel_id].append(acquisition)

            return results
        finally:
            for pulse_channel_id, sequencer in self._id2seq.items():
                sequencer.sync_en(False)
                if ChannelType.macq.name in pulse_channel_id:
                    sequencer.delete_acquisition_data(all=True)

            self._id2seq.clear()
            self._reset_modules()


class QbloxCompositeInstrument(CompositeInstrument[QbloxLeafInstrument]):
    """
    Composing Qblox instruments can be achieved by 2 methods:
    1- Daisy-chaining the REF_out of one cluster's CMM to REF_in of the next cluster's CMM.
       This is like a linked list pattern where the first cluster is **allowed** to have both "external"
       or "internal" reference source config, but subsequent clusters **must** have their reference
       source set as "external".
    2- Distribute the clock from a common source to all the REF_in of all clusters' CMM modules.
       This is like a star pattern where an external reference clock is distributed to all the clusters
       in the fleet

    Historically, both methods have been supported, but recent FW versions ditched the first method and only
    support the second method. Regardless of the method followed, this abstraction remains oblivious.
    """

    pass


LiveQbloxInstrument = QbloxLeafInstrument | QbloxCompositeInstrument
