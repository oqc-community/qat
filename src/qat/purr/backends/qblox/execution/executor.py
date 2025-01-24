# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import json
import os
from collections import ChainMap
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from qblox_instruments import Cluster
from qblox_instruments.qcodes_drivers.module import Module
from qblox_instruments.qcodes_drivers.sequencer import Sequencer

from qat.purr.backends.qblox.codegen import QbloxPackage
from qat.purr.backends.qblox.config import (
    QcmConfigHelper,
    QcmRfConfigHelper,
    QrmConfigHelper,
    QrmRfConfigHelper,
)
from qat.purr.backends.qblox.execution.instrument_base import (
    CompositeInstrument,
    InstrumentModel,
    LeafInstrument,
)
from qat.purr.backends.qblox.visualisation import plot_acquisitions, plot_packages
from qat.purr.compiler.devices import PulseChannel
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class LeafExecutor(LeafInstrument):
    def __init__(self, model: InstrumentModel, dummy_cfg: Dict = None):
        super().__init__(model)
        self.dummy_cfg = dummy_cfg

        self.dump_sequence = False
        self.plot_packages = False
        self.plot_acquisitions = False

        self._driver: Optional[Cluster] = None
        self._resources: Dict[Module, Dict[PulseChannel, Sequencer]] = {}

    def allocate_resources(self, packages: List[QbloxPackage]):
        for pkg in packages:
            target = pkg.target
            module_id = target.physical_channel.slot_idx - 1  # slot_idx is in range [1..20]
            module: Module = self._driver.modules[module_id]
            allocations = self._resources.setdefault(module, {})
            if target not in allocations:
                total = set(target.physical_channel.config.sequencers.keys())
                allocated = set([seq.seq_idx for seq in allocations.values()])

                available = total - allocated
                if not available:
                    raise ValueError(f"No more available sequencers on module {module}")
                sequencer: Sequencer = module.sequencers[next(iter(available))]
                allocations[target] = sequencer

        return self._resources

    def _delete_acquisitions(self, sequencer):
        sequencer.delete_acquisition_data(all=True)

    def _prepare_config(self, package: QbloxPackage, sequencer: Sequencer):
        if package.target.fixed_if:  # NCO freq constant
            nco_freq = package.target.baseband_if_frequency
            lo_freq = package.target.frequency - nco_freq
        else:  # LO freq constant
            lo_freq = package.target.baseband_frequency
            nco_freq = package.target.frequency - lo_freq

        qblox_config = package.target.physical_channel.config

        module_config = qblox_config.module
        if module_config.lo.out0_en:
            module_config.lo.out0_freq = lo_freq
        if module_config.lo.out1_en:
            module_config.lo.out1_freq = lo_freq
        if module_config.lo.out0_in0_en:
            module_config.lo.out0_in0_freq = lo_freq

        # Sequencer config from the HwM
        hwm_seq_config = qblox_config.sequencers[sequencer.seq_idx]
        # Sequencer config from QAT IR
        pkg_seq_config = package.sequencer_config
        hwm_seq_config.nco.freq = nco_freq
        hwm_seq_config.square_weight_acq.integration_length = (
            pkg_seq_config.square_weight_acq.integration_length
        )

        return qblox_config

    def _reset_io(self):
        # TODO - Qblox bug: Hard reset clutters sequencer connections with conflicting defaults
        # TODO - This is a temporary workaround until Qblox fixes the issue

        modules = self._resources.keys() or self._driver.get_connected_modules().values()

        for m in modules:
            log.info(f"Resetting sequencer connections for module {m.slot_idx}")
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
            self._reset_io()

        log.info(self._driver.get_system_status())
        self.is_connected = True

    def disconnect(self):
        if self._driver is not None:
            try:
                self._driver.close()
                self.is_connected = False
            except BaseException as e:
                log.warning(f"Failed to close instrument {str(self)}\n{str(e)}")

    def install(self, package: QbloxPackage, module: Module, sequencer: Sequencer):
        """
        Installs and configures the package on the given (module, sequencer) resource.
        Nominally follows resource allocation, but also serves for adhoc package installations
        that aren't necessarily tied to any automatic resource allocation.
        """

        log.debug(f"Configuring module {module}, sequencer {sequencer}")
        config = self._prepare_config(package, sequencer)
        if module.is_qcm_type:
            if module.is_rf_type:
                QcmRfConfigHelper(config).configure(module, sequencer)
            else:
                QcmConfigHelper(config).configure(module, sequencer)
        elif module.is_qrm_type:
            if module.is_rf_type:
                QrmRfConfigHelper(config).configure(module, sequencer)
            else:
                QrmConfigHelper(config).configure(module, sequencer)

        sequence = asdict(package.sequence)
        log.debug(f"Uploading sequence to {module}, sequencer {sequencer}")
        sequencer.sequence(sequence)

        return module, sequencer

    def upload(self, packages: List[QbloxPackage]):
        self._resources.clear()
        self.allocate_resources(packages)

        if self.plot_packages:
            plot_packages(packages)

        try:
            for package in packages:
                module, sequencer = next(
                    (
                        (m, t2s[package.target])
                        for m, t2s in self._resources.items()
                        if m.slot_idx == package.target.physical_channel.slot_idx
                    )
                )

                if self.dump_sequence:
                    filename = f"schedules/target_{package.target.id}_module_{module.slot_idx}_sequencer_{sequencer.seq_idx}_@_{datetime.utcnow().strftime('%m-%d-%Y_%H%M%S')}.json"
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    with open(filename, "w") as f:
                        f.write(json.dumps(asdict(package.sequence)))

                self.install(package, module, sequencer)
        except BaseException as e:
            self._reset_io()
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

                module.disconnect_outputs()
                if module.is_qrm_type:
                    module.disconnect_inputs()

    def collect(self):
        if not self._resources:
            raise ValueError("No resources allocated. Install packages first")

        results = {}
        try:
            for module, allocations in self._resources.items():
                if module.is_qrm_type:
                    for target, sequencer in allocations.items():
                        # TODO - 60 min tops, make it dynamic by involving flow-aware timeline duration
                        sequencer.get_acquisition_status(timeout=60)
                        acquisitions = sequencer.get_acquisitions()

                        for acq_name, acq in acquisitions.items():
                            sequencer.store_scope_acquisition(acq_name)

                            i = np.array(acq["acquisition"]["bins"]["integration"]["path0"])
                            q = np.array(acq["acquisition"]["bins"]["integration"]["path1"])
                            results[acq_name] = (
                                i + 1j * q
                            ) / sequencer.integration_length_acq()

                        if self.plot_acquisitions:
                            plot_acquisitions(
                                sequencer.get_acquisitions(),  # (re)fetch after store
                                integration_length=sequencer.integration_length_acq(),
                            )

                        self._delete_acquisitions(sequencer)
        finally:
            for module, allocations in self._resources.items():
                for target, sequencer in allocations.items():
                    sequencer.sync_en(False)

                module.disconnect_outputs()
                if module.is_qrm_type:
                    module.disconnect_inputs()

        return results


class CompositeExecutor(CompositeInstrument):
    def upload(self, packages: List[QbloxPackage]):
        for comp in self.components.values():
            comp.upload(packages)

    def playback(self, *args, **kwargs):
        for comp in self.components.values():
            comp.playback(*args, **kwargs)

    def collect(self, *args, **kwargs):
        results = [comp.collect(*args, **kwargs) for comp in self.components.values()]
        return dict(ChainMap(*results))
