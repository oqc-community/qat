import json
import os
from dataclasses import asdict
from datetime import datetime
from functools import reduce
from typing import Dict, List

import numpy as np
from qblox_instruments import Cluster, DummyScopeAcquisitionData
from qblox_instruments.qcodes_drivers.module import Module
from qblox_instruments.qcodes_drivers.sequencer import Sequencer

from qat.purr.backends.live_devices import ControlHardware, Instrument, LivePhysicalBaseband
from qat.purr.backends.qblox.codegen import QbloxPackage
from qat.purr.backends.qblox.config import (
    QbloxConfig,
    QcmConfigHelper,
    QcmRfConfigHelper,
    QrmConfigHelper,
    QrmRfConfigHelper,
)
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
        reson.create_pulse_channel(ChannelType.macq, *args, fixed_if=True, **kwargs)
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
            fixed_if=fixed_drive_if,
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
        dev_id: str = None,
        name: str = None,
        address: str = None,
        cfg: Dict = None,
    ):
        super().__init__(id_=dev_id or os.environ.get("QBLOX_DEV_ID"))
        self.name = name or os.environ.get("QBLOX_DEV_NAME")
        self.address = address or os.environ.get("QBLOX_DEV_IP")
        self.cfg = cfg
        self.dump_sequence = False
        self._resources: Dict[Module, Dict[PulseChannel, Sequencer]] = {}

    def allocate_resources(self, target: PulseChannel):
        module_id = target.physical_channel.slot_idx - 1  # slot_idx is in range [1..20]
        module: Module = self._driver.modules[module_id]
        allocations = self._resources.setdefault(module, {})
        if target in allocations:
            return module, allocations[target]

        total = set(target.physical_channel.config.sequencers.keys())
        allocated = set([seq.seq_idx for seq in allocations.values()])

        available = list(total - allocated)
        if available:
            sequencer: Sequencer = module.sequencers[available[0]]
            allocations[target] = sequencer
            return module, sequencer

        raise ValueError(f"No more available sequencers on module {module}")

    def _get_acquisitions(self, module: Module, sequencer: Sequencer):
        module.get_sequencer_status(sequencer.seq_idx, timeout=1)
        module.get_acquisition_status(sequencer.seq_idx, timeout=1)

        acquisitions = module.get_acquisitions(sequencer.seq_idx)
        for acq_name in acquisitions:
            module.store_scope_acquisition(sequencer.seq_idx, acq_name)

        return module.get_acquisitions(sequencer.seq_idx)

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

    def connect(self):
        if self._driver is None or not Cluster.is_valid(self._driver):
            self._driver: Cluster = Cluster(
                name=self.name,
                identifier=self.address,
                dummy_cfg=(
                    self.cfg if self.address is None else None
                ),  # Ignore dummy config if an address is given
            )
            self._driver.reset()
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

    def install(self, package: QbloxPackage):
        module, sequencer = self.allocate_resources(package.target)
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
        if self.dump_sequence:
            filename = f"schedules/sequence_{module.slot_idx}_{sequencer.seq_idx}_@_{datetime.utcnow()}.json"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w") as f:
                f.write(json.dumps(sequence))

        log.debug(f"Uploading sequence to {module}, sequencer {sequencer}")
        sequencer.sequence(sequence)

        return module, sequencer

    def set_data(self, qblox_packages: List[QbloxPackage]):
        self._resources.clear()
        for package in qblox_packages:
            self.install(package)

    def start_playback(self, repetitions: int, repetition_time: float):
        if not any(self._resources):
            raise ValueError("No resources allocated. Install packages first")

        results = {}
        for module, allocations in self._resources.items():
            for target, sequencer in allocations.items():
                if module.is_qrm_type:
                    sequencer.delete_acquisition_data(all=True)

        for module, allocations in self._resources.items():
            for target, sequencer in allocations.items():
                module.arm_sequencer(sequencer.seq_idx)

        for module, allocations in self._resources.items():
            for target, sequencer in allocations.items():
                module.start_sequencer(sequencer.seq_idx)

        for module, allocations in self._resources.items():
            for target, sequencer in allocations.items():
                if module.is_qrm_type:
                    result_id = target.physical_channel.id
                    if result_id in results:
                        raise ValueError(
                            "Two or more pulse channels on the same physical channel"
                        )
                    acquisitions = self._get_acquisitions(module, sequencer)
                    for acq_name, acq in acquisitions.items():
                        i = np.array(acq["acquisition"]["bins"]["integration"]["path0"])
                        q = np.array(acq["acquisition"]["bins"]["integration"]["path1"])
                        results[result_id] = (
                            i + 1j * q
                        ) / sequencer.integration_length_acq()

        self._resources.clear()
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
    def _setup_dummy_acq_data(self, module, sequencer: Sequencer):
        if module.is_qrm_type:
            waveforms = sequencer.get_waveforms()
            dummy_data = reduce(lambda a, b: a + b, [a["data"] for a in waveforms.values()])
            dummy_data = [(z, z) for z in dummy_data / np.linalg.norm(dummy_data)]
            dummy_scope_acquisition_data = DummyScopeAcquisitionData(
                data=dummy_data, out_of_range=(False, False), avg_cnt=(0, 0)
            )
            module.set_dummy_scope_acquisition_data(
                sequencer=sequencer.seq_idx, data=dummy_scope_acquisition_data
            )

    def set_data(self, qblox_packages: List[QbloxPackage]):
        self._resources.clear()
        for package in qblox_packages:
            module, sequencer = self.install(package)

            self._setup_dummy_acq_data(module, sequencer)

    def start_playback(self, repetitions: int, repetition_time: float):
        if not any(self._resources):
            raise ValueError("No resources allocated. Install packages first")

        results = {}
        for module, allocations in self._resources.items():
            for target, sequencer in allocations.items():
                module.arm_sequencer(sequencer.seq_idx)
                module.start_sequencer(sequencer.seq_idx)

                if module.is_qrm_type:
                    result_id = target.physical_channel.id
                    if result_id in results:
                        raise ValueError(
                            "Two or more pulse channels on the same physical channel"
                        )
                    acquisitions = self._get_acquisitions(module, sequencer)
                    for acq_name, acq in acquisitions.items():
                        i = np.array(acq["acquisition"]["bins"]["integration"]["path0"])
                        q = np.array(acq["acquisition"]["bins"]["integration"]["path1"])
                        results[result_id] = (
                            i + 1j * q
                        ) / sequencer.integration_length_acq()

        self._resources.clear()
        return results
