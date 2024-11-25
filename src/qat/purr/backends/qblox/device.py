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
        dev_id: str = None,
        name: str = None,
        address: str = None,
        dummy_cfg: Dict = None,
    ):
        super().__init__(id_=dev_id or os.environ.get("QBLOX_DEV_ID"))
        self.name = name or os.environ.get("QBLOX_DEV_NAME")
        self.address = address or os.environ.get("QBLOX_DEV_IP")
        self.dummy_cfg = dummy_cfg
        self.dump_sequence = False
        self.plot_packages = False
        self.plot_acquisitions = False
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

    def _reset_io(self, modules=None):
        # TODO - Qblox bug: Hard reset clutters sequencer connections with conflicting defaults
        # TODO - This is a temporary workaround until Qblox fixes the issue

        modules = modules or self._driver.get_connected_modules().values()

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

    def set_data(self, qblox_packages: List[QbloxPackage]):
        self._resources.clear()
        self.allocate_resources(qblox_packages)

        if self.plot_packages:
            plot_packages(qblox_packages)

        try:
            for package in qblox_packages:
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
            self._reset_io(self._resources.keys())
            raise e

    def start_playback(self, repetitions: int, repetition_time: float):
        if not any(self._resources):
            raise ValueError("No resources allocated. Install packages first")

        results = {}
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
                        sequencer.get_acquisition_status(timeout=1)
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
            dummy_data = [(z, z) for z in dummy_data / np.linalg.norm(dummy_data)]
        else:
            playback_pattern = regex.compile(
                "set_awg_offs( +)([0-9]+),([0-9]+)\nupd_param( +)([0-9]+)"
            )
            match = next(playback_pattern.finditer(sequence.program))
            i_steps, q_steps = int(match.group(2)), int(match.group(3))
            num_samples = int(match.group(5))
            dummy_data = [
                (
                    i_steps / (Constants.MAX_OFFSET_SIZE // 2),
                    q_steps / (Constants.MAX_OFFSET_SIZE // 2),
                )
            ] * num_samples

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

    def set_data(self, qblox_packages: List[QbloxPackage]):
        super().set_data(qblox_packages)

        # Stage Scope and Acquisition data
        for module, allocations in self._resources.items():
            if module.is_qrm_type:
                for target, sequencer in allocations.items():
                    package = next((pkg for pkg in qblox_packages if pkg.target == target))
                    self._setup_dummy_scope_acq_data(module, sequencer, package.sequence)
                    self._setup_dummy_binned_acq_data(module, sequencer, package.sequence)
