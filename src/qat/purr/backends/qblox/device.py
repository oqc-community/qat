# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd

import json
import os
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List

from qblox_instruments import Cluster
from qblox_instruments.qcodes_drivers.module import Module
from qblox_instruments.qcodes_drivers.sequencer import Sequencer

from qat.purr.backends.live_devices import ControlHardware, Instrument, LivePhysicalBaseband
from qat.purr.backends.qblox.acquisition import Acquisition
from qat.purr.backends.qblox.codegen import QbloxPackage
from qat.purr.backends.qblox.config import (
    QbloxConfig,
    QcmConfigHelper,
    QcmRfConfigHelper,
    QrmConfigHelper,
    QrmRfConfigHelper,
)
from qat.purr.backends.qblox.constants import Constants
from qat.purr.backends.qblox.visualisation import plot_packages, plot_playback
from qat.purr.compiler.devices import (
    ChannelType,
    PhysicalChannel,
    PulseChannel,
    Qubit,
    Resonator,
)
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class QbloxResonator(Resonator):
    """
    A hack around QBlox acquisition to use the same pulse channel
    for MeasurePulse and Acquire instructions.
    """

    def get_measure_channel(self) -> PulseChannel:
        return self.get_pulse_channel(ChannelType.macq)

    def get_acquire_channel(self) -> PulseChannel:
        return self.get_pulse_channel(ChannelType.macq)


class QbloxPhysicalBaseband(LivePhysicalBaseband):
    def __init__(
        self,
        id_,
        frequency,
        if_frequency,
        instrument: Instrument,
        slot_idx: int,
        config: QbloxConfig,
    ):
        super().__init__(id_, frequency, if_frequency, instrument)
        self.slot_idx = slot_idx
        self.config = config  # TBD, is it worth flattening ?


class QbloxPhysicalChannel(PhysicalChannel):
    baseband: QbloxPhysicalBaseband

    @property
    def slot_idx(self):
        return self.baseband.slot_idx

    @property
    def config(self):
        return self.baseband.config

    def build_resonator(self, resonator_id, *args, **kwargs):
        """Helper method to build a resonator with default channels."""
        kwargs.pop("fixed_if", None)
        reson = QbloxResonator(resonator_id, self)
        reson.create_pulse_channel(ChannelType.macq, *args, fixed_if=False, **kwargs)
        reson.default_pulse_channel_type = ChannelType.macq
        return reson

    def build_qubit(
        self,
        index,
        resonator,
        drive_freq,
        second_state_freq=None,
        channel_scale=(1.0e-8 + 0.0j),
        measure_amp: float = 1.0,
        fixed_drive_if=False,
        qubit_id=None,
    ):
        """
        Helper method tp build a qubit with assumed default values on the channels. Modelled after the live hardware.
        """
        qubit = Qubit(index, resonator, self, drive_amp=measure_amp, id_=qubit_id)
        qubit.create_pulse_channel(
            ChannelType.drive,
            frequency=drive_freq,
            scale=channel_scale,
            fixed_if=False,
        )

        qubit.create_pulse_channel(
            ChannelType.second_state, frequency=second_state_freq, scale=channel_scale
        )

        qubit.measure_acquire["delay"] = 151e-9  # TOF

        return qubit


class QbloxControlHardware(ControlHardware):
    _driver: Cluster

    def __init__(
        self,
        dev_id: str,
        name: str,
        address: str = None,
        dummy_cfg: Dict = None,
    ):
        super().__init__(id_=dev_id)
        self.name = name
        self.address = address
        self.dummy_cfg = dummy_cfg

        self.dump_sequence = False
        self.plot_packages = False
        self.plot_acquisitions = False

        self._modules: Dict[Module, bool] = {}
        self._id2seq: Dict[str, Sequencer] = {}

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
    def id2seq(self):
        return self._id2seq

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
            self._driver: Cluster = Cluster(
                name=self.name,
                identifier=self.address,
                dummy_cfg=self.dummy_cfg if self.address is None else None,
            )
            self._driver.reset()
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

    def set_data(self, packages: List[QbloxPackage]):
        if self.plot_packages:
            plot_packages(packages)

        if self.dump_sequence:
            for pkg in packages:
                filename = f"schedules/target_{pkg.pulse_channel_id}_@_{datetime.now().strftime('%m-%d-%Y_%H%M%S')}.json"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, "w") as f:
                    f.write(json.dumps(asdict(pkg.sequence)))

        try:
            self._id2seq.clear()
            self._reset_modules()
            for pkg in packages:
                self.configure(pkg)
        except BaseException as e:
            self._id2seq.clear()
            self._reset_modules()
            raise e

    def start_playback(self, repetitions: int, repetition_time: float):
        if not any(self._id2seq):
            raise ValueError("No allocations found. Install packages and configure first")

        results: Dict[str, List[Acquisition]] = defaultdict(list)
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

            if self.plot_acquisitions:
                plot_playback(results)

            return results
        finally:
            for pulse_channel_id, sequencer in self._id2seq.items():
                sequencer.sync_en(False)
                if ChannelType.macq.name in pulse_channel_id:
                    sequencer.delete_acquisition_data(all=True)

            self._id2seq.clear()
            self._reset_modules()

    def __getstate__(self) -> Dict:
        results = super(QbloxControlHardware, self).__getstate__()
        results["_driver"] = None
        results["_modules"] = {}
        results["_id2seq"] = {}
        return results

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._driver = None
        self._modules = {}
        self._id2seq = {}
        self.is_connected = False
