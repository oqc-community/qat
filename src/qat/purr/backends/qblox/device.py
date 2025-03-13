# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd

import json
import os
from dataclasses import asdict
from datetime import datetime
from functools import reduce
from typing import Dict, List

import numpy as np
import regex
from qblox_instruments import Cluster, DummyBinnedAcquisitionData, DummyScopeAcquisitionData
from qblox_instruments.qcodes_drivers.module import Module
from qblox_instruments.qcodes_drivers.sequencer import Sequencer

from qat.purr.backends.live_devices import ControlHardware, Instrument, LivePhysicalBaseband
from qat.purr.backends.qblox.codegen import QbloxPackage
from qat.purr.backends.qblox.config import (
    ModuleConfig,
    QbloxConfig,
    QcmConfigHelper,
    QcmRfConfigHelper,
    QrmConfigHelper,
    QrmRfConfigHelper,
)
from qat.purr.backends.qblox.constants import Constants
from qat.purr.backends.qblox.ir import Sequence
from qat.purr.backends.qblox.visualisation import plot_acquisitions, plot_packages
from qat.purr.compiler.devices import (
    ChannelType,
    PhysicalChannel,
    PulseChannel,
    Qubit,
    Resonator,
)
from qat.purr.compiler.instructions import AcquireMode
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

        self._resources: Dict[Module, Dict[PulseChannel, Sequencer]] = {}

    def allocate_resources(self, package: QbloxPackage):
        target = package.target
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
            f"Sequencer {sequencer.seq_idx} in Module {module.slot_idx} will be running {package.target}"
        )
        return module, sequencer

    def configure(self, package: QbloxPackage, module: Module, sequencer: Sequencer):
        if package.target.fixed_if:  # NCO freq constant
            nco_freq = package.target.baseband_if_frequency
            lo_freq = package.target.frequency - nco_freq
        else:  # LO freq constant
            lo_freq = package.target.baseband_frequency
            nco_freq = package.target.frequency - lo_freq

        qblox_config = package.target.physical_channel.config

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
        sequencer_config = qblox_config.sequencers[sequencer.seq_idx]
        sequencer_config.nco.freq = nco_freq
        sequencer_config.square_weight_acq.integration_length = (
            package.sequencer_config.square_weight_acq.integration_length
        )

        log.debug(f"Configuring module {module}, sequencer {sequencer}")
        if module.is_qcm_type:
            if module.is_rf_type:
                QcmRfConfigHelper(module_config, sequencer_config).configure(
                    module, sequencer
                )
            else:
                QcmConfigHelper(module_config, sequencer_config).configure(
                    module, sequencer
                )
        elif module.is_qrm_type:
            if module.is_rf_type:
                QrmRfConfigHelper(module_config, sequencer_config).configure(
                    module, sequencer
                )
            else:
                QrmConfigHelper(module_config, sequencer_config).configure(
                    module, sequencer
                )

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
                log.warning(
                    f"Failed to close instrument ID: {self.id} at: {self.address}\n{str(e)}"
                )

    def set_data(self, packages: List[QbloxPackage]):
        if self.plot_packages:
            plot_packages(packages)

        if self.dump_sequence:
            for pkg in packages:
                filename = f"schedules/target_{pkg.target.id}_@_{datetime.now().strftime('%m-%d-%Y_%H%M%S')}.json"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, "w") as f:
                    f.write(json.dumps(asdict(pkg.sequence)))

        self._resources.clear()

        try:
            for pkg in packages:
                module, sequencer = self.allocate_resources(pkg)
                self.configure(pkg, module, sequencer)
                sequence = asdict(pkg.sequence)
                log.debug(f"Uploading sequence to {module}, sequencer {sequencer}")
                sequencer.sequence(sequence)
        except BaseException as e:
            self._reset_io()
            raise e

    def start_playback(self, repetitions: int, repetition_time: float):
        if not any(self._resources):
            raise ValueError("No resources allocated. Install packages first")

        results = {AcquireMode.SCOPE: {}, AcquireMode.INTEGRATOR: {}}
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

                        for acq_name in acquisitions:
                            sequencer.store_scope_acquisition(acq_name)

                        acquisitions = sequencer.get_acquisitions()  # (re)fetch after store
                        if self.plot_acquisitions:
                            plot_acquisitions(
                                acquisitions,
                                integration_length=sequencer.integration_length_acq(),
                            )

                        start, end = 0, min(
                            sequencer.integration_length_acq(),
                            Constants.MAX_SAMPLE_SIZE_SCOPE_ACQUISITIONS,
                        )
                        for acq_name, acq in acquisitions.items():
                            i = np.array(acq["acquisition"]["bins"]["integration"]["path0"])
                            q = np.array(acq["acquisition"]["bins"]["integration"]["path1"])
                            results[AcquireMode.INTEGRATOR][acq_name] = (
                                i + 1j * q
                            ) / sequencer.integration_length_acq()

                            i = np.array(acq["acquisition"]["scope"]["path0"]["data"])
                            q = np.array(acq["acquisition"]["scope"]["path1"]["data"])
                            results[AcquireMode.SCOPE][acq_name] = (i + 1j * q)[start:end]

                        sequencer.delete_acquisition_data(all=True)
        finally:
            for module, allocations in self._resources.items():
                for target, sequencer in allocations.items():
                    sequencer.sync_en(False)

                module.disconnect_outputs()
                if module.is_qrm_type:
                    module.disconnect_inputs()

        return results

    def __getstate__(self) -> Dict:
        results = super(QbloxControlHardware, self).__getstate__()
        results["_driver"] = None
        results["_resources"] = {}
        return results

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._driver = None
        self._resources = {}
        self.is_connected = False


class DummyQbloxControlHardware(QbloxControlHardware):
    def _setup_dummy_scope_acq_data(self, module, sequencer: Sequencer, sequence: Sequence):
        if sequence.waveforms:
            dummy_data = reduce(
                lambda a, b: a + b, [a["data"] for a in sequence.waveforms.values()]
            )
            norm = np.linalg.norm(dummy_data)
            i_val = min(dummy_data) / norm
            q_val = max(dummy_data) / norm
        else:
            playback_pattern = regex.compile(
                "set_awg_offs( +)([0-9]+),([0-9]+)\nupd_param( +)([0-9]+)"
            )
            match = next(playback_pattern.finditer(sequence.program))
            i_val = int(match.group(2)) / (Constants.MAX_OFFSET_SIZE // 2)
            q_val = int(match.group(3)) / (Constants.MAX_OFFSET_SIZE // 2)

        dummy_data = [(i_val, q_val)] * Constants.MAX_SAMPLE_SIZE_SCOPE_ACQUISITIONS
        dummy_scope_acquisition_data = DummyScopeAcquisitionData(
            data=dummy_data, out_of_range=(False, False), avg_cnt=(1, 1)
        )
        module.set_dummy_scope_acquisition_data(
            sequencer=sequencer.seq_idx, data=dummy_scope_acquisition_data
        )

    def _setup_dummy_binned_acq_data(
        self, module, sequencer: Sequencer, sequence: Sequence
    ):
        for name, acquisition in sequence.acquisitions.items():
            dummy_data = (np.random.random(), np.random.random())
            dummy_binned_acquisition_data = [
                DummyBinnedAcquisitionData(data=dummy_data, thres=1, avg_cnt=1)
            ] * acquisition["num_bins"]
            module.set_dummy_binned_acquisition_data(
                sequencer=sequencer.seq_idx,
                acq_index_name=name,
                data=dummy_binned_acquisition_data,
            )

    def _delete_acquisitions(self, sequencer):
        sequencer.delete_dummy_scope_acquisition_data()
        sequencer.delete_dummy_binned_acquisition_data()

    def set_data(self, packages: List[QbloxPackage]):
        super().set_data(packages)

        # Stage Scope and Acquisition data
        for module, allocations in self._resources.items():
            if module.is_qrm_type:
                for target, sequencer in allocations.items():
                    package = next((pkg for pkg in packages if pkg.target == target))
                    self._setup_dummy_scope_acq_data(module, sequencer, package.sequence)
                    self._setup_dummy_binned_acq_data(module, sequencer, package.sequence)
