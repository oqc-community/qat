# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd

import itertools
from datetime import datetime
import json
import os
from typing import Dict, List, Any, Tuple

import numpy as np
from qblox_instruments import Cluster
from qblox_instruments.qcodes_drivers.qcm_qrm import QcmQrm
from qblox_instruments.qcodes_drivers.sequencer import Sequencer

from qat.purr.backends.live import LiveDeviceEngine, LiveHardwareModel
from qat.purr.backends.live_devices import ControlHardware, Instrument, LivePhysicalBaseband
from qat.purr.backends.utilities import evaluate_shape, get_axis_map, PositionData
from qat.purr.backends.virtual_devices import Oscilloscope, ResultType
from qat.purr.compiler.devices import PulseChannel, PhysicalChannel, ChannelType, Resonator, Qubit, PulseShapeType
from qat.purr.compiler.emitter import QatFile
from qat.purr.compiler.execution import SweepIterator
from qat.purr.compiler.instructions import PhaseShift, Acquire, Synchronize, calculate_duration, Delay, Pulse, CustomPulse, \
    Waveform, QuantumInstruction, FrequencyShift, PhaseReset, Id, Repeat, Reset, DeviceUpdate, MeasurePulse, \
    AcquireMode, PostProcessing
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class Constants:
    """
    Useful QBlox constant parameters
    """

    IMMEDIATE_MAX_WAIT_TIME = pow(2, 16) - 4
    """Max size of wait instruction immediates in Q1ASM programs. Max value allowed by
    assembler is 2**16-1, but this is the largest that is a multiple of 4 ns."""
    REGISTER_SIZE = pow(2, 32) - 1
    """Size of registers in Q1ASM programs."""
    NCO_PHASE_STEPS_PER_DEG = 1e9 / 360
    """The number of steps per degree for NCO phase instructions arguments."""
    NCO_FREQ_STEPS_PER_HZ = 4.0
    """The number of steps per Hz for the NCO set_freq instruction."""
    NCO_FREQ_LIMIT_STEPS = 2e9
    """The maximum and minimum frequency expressed in steps for the NCO set_freq instruction.
    For the minimum we multiply by -1."""
    NUMBER_OF_SEQUENCERS_QCM = 6
    """Number of sequencers supported by a QCM/QCM-RF in the latest firmware."""
    NUMBER_OF_SEQUENCERS_QRM = 6
    """Number of sequencers supported by a QRM/QRM-RF in the latest firmware."""
    NUMBER_OF_REGISTERS: int = 64
    """Number of registers available in the Qblox sequencers."""
    MAX_SAMPLE_SIZE_SCOPE_ACQUISITIONS: int = 16384
    """Maximal amount of scope trace acquisition datapoints returned."""
    MAX_SAMPLE_SIZE_WAVEFORMS: int = 16384
    """Maximal amount of samples in the waveforms to be uploaded to a sequencer."""
    GRID_TIME = 4  # ns
    """
    Clock period of the sequencers. All time intervals used must be multiples of this value.
    """


class SequenceFile:
    """
    Represents QBlox's "sequence" object:

    sequence = {
        "waveforms": {
            "some_name": {
                "data": [float],
                "index": int,
            }
        },
        "weights": Dict,
        "acquisitions": Dict,
        "program": str,
    }

    It specifies:
        - Data to upload to QBlox QCM[-RF]/QRM[-RF] modules.
        - Hints that tells what kind of module/sequencer resource it needs to run
    """
    def __init__(self):
        self.waveforms: Dict[str, Dict[str, Any]] = {}
        self.weights: Dict[str, Any] = {}
        self.acquisitions: Dict[str, Any] = {}
        self.instructions: List[str] = []

        self.enable_timeline = False
        self.timeline: List = []
        self.duration: int = 0

        self._wf_memory: int = Constants.MAX_SAMPLE_SIZE_WAVEFORMS
        self._wf_index: int = 0
        self._acq_index: int = 0
        self._phase: float = 0.0  # Accumulation of Phase shifts
        self._frequency: float = 0.0  # Accumulation of Frequency shifts
        self._loops = {}

    def as_dict(self):
        return {
            "waveforms": self.waveforms,
            "weights": self.weights,
            "acquisitions": self.acquisitions,
            "program": "\n".join(self.instructions),
        }

    def __str__(self):
        return json.dumps(self.as_dict())

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def get_nco_phase_arguments(phase_rad: float) -> int:
        phase_deg = np.rad2deg(phase_rad)
        phase_deg %= 360
        return round(phase_deg * Constants.NCO_PHASE_STEPS_PER_DEG)

    @staticmethod
    def get_nco_set_frequency_arguments(frequency_hz: float) -> int:
        """
        Borrowed from Quantify scheduler
        """
        frequency_steps = round(frequency_hz * Constants.NCO_FREQ_STEPS_PER_HZ)

        if (
            frequency_steps < -Constants.NCO_FREQ_LIMIT_STEPS
            or frequency_steps > Constants.NCO_FREQ_LIMIT_STEPS
        ):
            min_max_frequency_in_hz = (
                Constants.NCO_FREQ_LIMIT_STEPS / Constants.NCO_FREQ_STEPS_PER_HZ
            )
            raise ValueError(
                f"Attempting to set NCO frequency. "
                f"The frequency must be between and including "
                f"-{min_max_frequency_in_hz:e} Hz and {min_max_frequency_in_hz:e} Hz. "
                f"Got {frequency_hz:e} Hz."
            )

        return frequency_steps

    def is_empty(self):
        return not (any(self.waveforms) or any(self.acquisitions))

    def append_instruction(self, q1asm_instruction: str):
        self.instructions.append(q1asm_instruction)

    def prepend_instruction(self, q1asm_instruction: str):
        self.instructions.insert(0, q1asm_instruction)

    def label_instruction(self, index, label):
        self.instructions[index] = f"{label}: {self.instructions[index]}"

    def create_loop(self):
        reg_index = max(self._loops.keys(), default=0) % Constants.NUMBER_OF_REGISTERS
        label = f"label{reg_index}"
        self._loops[reg_index] = label
        return reg_index, label

    def play_timeline(self, quantum_target: PulseChannel):
        if 2 * len(self.timeline) > self._wf_memory:  # for both I and Q
            raise ValueError(
                f"No more waveform memory left for timeline on pulse channel {quantum_target}"
            )

        wf_index = self._wf_index
        self.waveforms[f"timeline_I"] = {
            "data": [z.real for z in self.timeline], "index": wf_index
        }
        self.waveforms[f"timeline_Q"] = {
            "data": [z.imag for z in self.timeline], "index": wf_index + 1
        }
        self.append_instruction(
            f"play {wf_index},{wf_index + 1},{Constants.GRID_TIME}"
        )
        self._wf_memory = self._wf_memory - len(self.timeline)
        self._wf_index = wf_index + 2

        return wf_index

    def add_waveform(self, inst: Waveform, quantum_target: PulseChannel, play=False):
        samples = int(calculate_duration(inst, return_samples=True))
        if samples == 0:
            return None

        dt = quantum_target.sample_time
        length = samples * dt
        centre = length / 2.0
        t = np.linspace(
            start=-centre + 0.5 * dt,
            stop=length - centre - 0.5 * dt,
            num=samples
        )
        pulse = evaluate_shape(inst, t, self._phase)
        scale = quantum_target.scale
        if isinstance(inst, (Pulse, CustomPulse)) and inst.ignore_channel_scale:
            scale = 1

        pulse *= scale
        pulse += quantum_target.bias

        wf_name = type(inst).__name__
        if isinstance(inst, Pulse):
            wf_name = inst.shape.name

        if 2 * pulse.size > self._wf_memory:  # for both I and Q
            raise ValueError(
                f"No more waveform memory left for {wf_name} on pulse channel {quantum_target}"
            )

        index = len(self.waveforms)
        self.waveforms[f"{wf_name}_{index}_I"] = {
            "data": pulse.real.tolist(), "index": self._wf_index
        }
        self.waveforms[f"{wf_name}_{index}_Q"] = {
            "data": pulse.imag.tolist(), "index": self._wf_index + 1
        }

        self._wf_memory = self._wf_memory - pulse.size
        self.duration = self.duration + inst.duration
        self.timeline.extend(pulse)

        i_idx, q_idx = self._wf_index, self._wf_index + 1
        self._wf_index = self._wf_index + 2

        if play:
            self.append_instruction(f"play {i_idx},{q_idx},{Constants.GRID_TIME}")

        return i_idx, q_idx

    def add_delay(self, inst: Delay, quantum_target):
        # TODO - Convert to Q1ASM loop
        if inst.duration > 0:
            duration_nanos = int(inst.duration * 1e9)
            while duration_nanos > Constants.IMMEDIATE_MAX_WAIT_TIME:
                self.append_instruction(
                    f"wait {Constants.IMMEDIATE_MAX_WAIT_TIME}"
                )
                duration_nanos = duration_nanos - Constants.IMMEDIATE_MAX_WAIT_TIME
            if duration_nanos > 0:
                self.append_instruction(f"wait {duration_nanos}")

            self.timeline.extend(np.repeat(0 + 0j, int(calculate_duration(inst, return_samples=True))))
            self.duration = self.duration + inst.duration

    def add_acquisition(self, inst: Acquire, num_bins: int, quantum_target: PulseChannel):
        int_length = int(calculate_duration(inst, return_samples=True))
        acq_index = self._acq_index
        self.acquisitions[inst.output_variable] = {
            "num_bins": num_bins, "index": acq_index
        }
        self._acq_index = acq_index + 1
        return acq_index, int_length

    @staticmethod
    def add_repeat(inst: Repeat, seq_files):
        for quantum_target, seq_file in seq_files.items():
            if seq_file.is_empty():
                continue

            reg_idx, label = seq_file.create_loop()
            seq_file.prepend_instruction(f"upd_param {Constants.GRID_TIME}")
            seq_file.prepend_instruction("reset_ph")
            seq_file.label_instruction(0, f"{label}")
            seq_file.prepend_instruction("nop")
            seq_file.prepend_instruction(f"move {inst.repeat_count},R{reg_idx}")
            seq_file.add_delay(Delay(quantum_target, inst.repetition_period), quantum_target)
            seq_file.append_instruction(f"wait {Constants.GRID_TIME}")
            seq_file.append_instruction(f"loop R{reg_idx},@{label}")

    @staticmethod
    def add_synchronise(inst: Synchronize, quantum_target: PulseChannel, seq_files: Dict):
        durations = {qt: seq_file.duration for qt, seq_file in seq_files.items()}
        inst_qt_durations = {
            qt: durations.setdefault(qt, 0) for qt in inst.quantum_targets
        }
        max_length = max(inst_qt_durations.values(), default=0)
        for qt in inst.quantum_targets:
            if not isinstance(qt, PulseChannel):
                raise ValueError(f"{qt} is not a PulseChannel.")
            seq_file = seq_files.setdefault(qt, SequenceFile())
            delay_time = max_length - seq_file.duration
            seq_file.add_delay(Delay(qt, delay_time), qt)

    def add_phase_shift(self, inst, quantum_target):
        self._phase = self._phase + inst.phase
        value = self.get_nco_phase_arguments(self._phase)
        self.append_instruction(f"set_ph_delta {value}")
        self.append_instruction(f"upd_param {Constants.GRID_TIME}")

    def add_frequency_shift(self, inst, quantum_target):
        old_frequency = quantum_target.frequency + self._frequency
        new_frequency = old_frequency + inst.frequency
        if new_frequency < quantum_target.min_frequency or new_frequency > quantum_target.max_frequency:
            raise ValueError(
                f"Cannot shift pulse channel frequency from '{old_frequency}' to '{new_frequency}'."
            )

        self._frequency = self._frequency + inst.frequency
        # TODO - discuss w/t quantum_target.fixed_if is True or False
        shifted_nco_freq = quantum_target.baseband_if_frequency + self._frequency
        value = self.get_nco_set_frequency_arguments(shifted_nco_freq)  # 1 MHZ <-> 4e6
        self.append_instruction(f"set_freq {value}")
        self.append_instruction(f"upd_param {Constants.GRID_TIME}")

    @staticmethod
    def add_phase_reset(inst, quantum_target, seq_files):
        for quantum_target, seq_file in seq_files.items():
            seq_file._phase = 0.0
            seq_file.append_instruction("reset_ph")
            seq_file.append_instruction(f"upd_param {Constants.GRID_TIME}")

    def add_macq(self, inst, acquire_inst, quantum_target):
        i_idx, q_idx = self.add_waveform(inst, quantum_target)
        if (i_idx, q_idx) is None:
            raise ValueError("No measure pulse added. Verify that its duration is > 0.")

        num_bins = 50
        acq_idx, int_length = self.add_acquisition(acquire_inst, num_bins, quantum_target)
        for bin_idx in range(num_bins):
            self.append_instruction(f"play {i_idx},{q_idx},{Constants.GRID_TIME}")
            # seq_file.append_instruction(f"wait {1000}")
            self.append_instruction(
                f"acquire {acq_idx},{bin_idx},{Constants.MAX_SAMPLE_SIZE_SCOPE_ACQUISITIONS}"
            )
        return int_length

    def add_id(self, inst, quantum_target):
        self.append_instruction("nop")


class SequenceEmitter:
    # @Oscilloscope(result_type=ResultType.SEQUENCE_FILE)
    def emit(self, qat_file: QatFile) -> Dict[PulseChannel, SequenceFile]:
        seq_files: Dict[PulseChannel, SequenceFile] = {}

        inst_iter = iter(qat_file.instructions)

        while (inst := next(inst_iter, None)) is not None:
            if not isinstance(inst, QuantumInstruction):  # Ignore classical instructions
                continue

            if isinstance(inst, PostProcessing):
                continue

            for quantum_target in inst.quantum_targets:
                if not isinstance(quantum_target, PulseChannel):
                    raise ValueError(f"{quantum_target} is not a PulseChannel.")

                seq_file: SequenceFile = seq_files.setdefault(
                    quantum_target, SequenceFile()
                )

                if isinstance(inst, MeasurePulse):
                    acquire_inst = next(inst_iter, None)
                    if acquire_inst is None or not isinstance(acquire_inst, Acquire):
                        raise ValueError("Found a MeasurePulse but no Acquire instruction followed.")

                    seq_file.add_macq(inst, acquire_inst, quantum_target)
                elif isinstance(inst, Waveform):
                    seq_file.add_waveform(inst, quantum_target, play=True)
                elif isinstance(inst, Delay):
                    seq_file.add_delay(inst, quantum_target)
                elif isinstance(inst, Synchronize):
                    seq_file.add_synchronise(inst, quantum_target, seq_files)
                elif isinstance(inst, PhaseShift):  # FIXME - PhaseShift is optimised away
                    seq_file.add_phase_shift(inst, quantum_target)
                elif isinstance(inst, FrequencyShift):
                    seq_file.add_frequency_shift(inst, quantum_target)
                elif isinstance(inst, PhaseReset):
                    seq_file.add_phase_reset(inst, quantum_target, seq_files)
                elif isinstance(inst, Id):
                    seq_file.add_id(inst, quantum_target)

        self._prune_empty(seq_files)

        SequenceFile.add_repeat(qat_file.repeat, seq_files)

        for quantum_target, seq_file in seq_files.items():
            seq_file.prepend_instruction(f"wait {Constants.GRID_TIME}")
            seq_file.prepend_instruction(f"upd_param {Constants.GRID_TIME}")
            seq_file.prepend_instruction(f"wait_sync {Constants.GRID_TIME}")
            seq_file.prepend_instruction(f"set_mrk 3")

            seq_file.append_instruction("stop")

            # Register timeline waveforms if enabled
            if seq_file.enable_timeline:
                wf_index = seq_file.play_timeline(quantum_target)
                seq_file.append_instruction(
                    f"play {wf_index},{wf_index + 1},{Constants.GRID_TIME}"
                )

        return seq_files

    def _prune_empty(self, seq_files):
        ignored = []
        for quantum_target, seq_file in seq_files.items():
            if seq_file.is_empty():
                ignored.append(quantum_target)
                continue

        for pc in ignored:
            del seq_files[pc]


class QbloxPhysicalBaseband(LivePhysicalBaseband):
    """
    A wrapper over the PhysicalBaseband, that connects to QBlox.
    This is equivalent to an LO.
    """
    def __init__(
        self,
        id_,
        frequency,
        if_frequency,
        instrument: Instrument,
        slot_idx: int,
        lo_idx: int,
        sequencer_connections: Dict[int, str]
    ):
        super().__init__(id_, frequency, if_frequency, instrument)
        self.slot_idx = slot_idx
        self.lo_idx = lo_idx
        self.sequencer_connections = sequencer_connections


class QbloxResonator(Resonator):
    """
    A hack around QBlox acquisition to use the sampe pulse channel
    for MeasurePulse and Acquire instructions
    """
    def get_measure_channel(self) -> PulseChannel:
        return self.get_pulse_channel(ChannelType.macq)

    def get_acquire_channel(self) -> PulseChannel:
        return self.get_pulse_channel(ChannelType.macq)


class QbloxPhysicalChannel(PhysicalChannel):
    def __init__(self, id_, baseband: QbloxPhysicalBaseband, *args, **kwargs):
        super().__init__(id_, *args, baseband, **kwargs)
        self.baseband: QbloxPhysicalBaseband = baseband
        self._pc2seq_idx: Dict[Any, Any] = {}
        self._seq_idx_cycle = itertools.cycle(self.sequencer_connections.keys())

    @property
    def slot_idx(self):
        return self.baseband.slot_idx

    @property
    def lo_idx(self):
        return self.baseband.lo_idx

    @property
    def sequencer_connections(self):
        return self.baseband.sequencer_connections

    def get_seq_idx(self, pulse_channel):
        return self._pc2seq_idx[pulse_channel]

    def create_pulse_channel(
        self,
        id_: str,
        frequency=0.0,
        bias=0.0 + 0.0j,
        scale=1.0 + 0.0j,
        fixed_if: bool = False,
    ):
        pulse_channel = super().create_pulse_channel(id_, frequency, bias, scale, fixed_if)

        self._pc2seq_idx[pulse_channel] = next(self._seq_idx_cycle)

        return pulse_channel

    def build_resonator(self, resonator_id, *args, **kwargs):
        """ Helper method to build a resonator with default channels. """
        kwargs.pop('fixed_if', None)
        reson = QbloxResonator(resonator_id, self)
        reson.default_pulse_channel_type = ChannelType.macq
        pulse_channel = reson.create_pulse_channel(ChannelType.macq, *args, fixed_if=True, **kwargs)
        self._pc2seq_idx[pulse_channel] = next(self._seq_idx_cycle)
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
        qubit_id=None
    ):
        """
        Helper method tp build a qubit with assumed default values on the channels. Modelled after the live hardware.
        """
        qubit = Qubit(index, resonator, self, drive_amp=measure_amp, id_=qubit_id)
        drive_channel = qubit.create_pulse_channel(
            ChannelType.drive,
            frequency=drive_freq,
            scale=channel_scale,
            fixed_if=fixed_drive_if
        )
        self._pc2seq_idx[drive_channel] = next(self._seq_idx_cycle)

        second_state_channel = qubit.create_pulse_channel(
            ChannelType.second_state, frequency=second_state_freq, scale=channel_scale
        )
        self._pc2seq_idx[second_state_channel] = next(self._seq_idx_cycle)

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
        super().__init__(id_=dev_id or os.environ["QBLOX_DEV_ID"])
        self.name = name or os.environ["QBLOX_DEV_NAME"]
        self.address = address or os.environ["QBLOX_DEV_IP"]
        self.cfg = cfg
        self.active_components: Dict[PulseChannel, List[Tuple[QcmQrm, Sequencer]]] = {}

    # @Oscilloscope(result_type=ResultType.ACQUISITION, sample_start=0, sample_end=3000)
    def get_acquisitions(self, module: QcmQrm, sequencer: Sequencer):
        module.get_acquisition_state(sequencer.seq_idx, timeout=1)

        acquisitions = module.get_acquisitions(sequencer.seq_idx)
        for acq_name in acquisitions:
            module.store_scope_acquisition(sequencer.seq_idx, acq_name)

        return module.get_acquisitions(sequencer.seq_idx)

    def configure_components(self, pulse_channel):
        physical_channel = pulse_channel.physical_channel

        if not isinstance(physical_channel, QbloxPhysicalChannel):
            raise ValueError("Not a Qblox PhysicalChannel")

        module_id = physical_channel.slot_idx - 1  # slot_idx is in range [1..20]
        seq_id = physical_channel.get_seq_idx(pulse_channel)
        connections = physical_channel.sequencer_connections

        module = self._driver.modules[module_id]
        sequencer = module.sequencers[seq_id]

        module.connect_sequencer(sequencer.seq_idx, connections[sequencer.seq_idx])

        # baseband_frequency = physical_channel.baseband_frequency  # or sweep over np.linspace(2e9, 18e9, 9)
        bb_if_frequency = physical_channel.baseband_if_frequency  # NCO, maybe diff per sequencer ?
        bb_frequency = pulse_channel.frequency - bb_if_frequency  # TMP
        if module.is_qcm_type:
            # Configure the LO
            module.out0_lo_en(True)
            module.out1_lo_en(True)
            module.out0_lo_freq(bb_frequency)
            module.out1_lo_freq(bb_frequency)

            # Configure attenuation
            module.out0_att(0)
            module.out1_att(0)

            # Set offset in mV
            module.out0_offset_path0(0)
            module.out0_offset_path1(0)
            module.out1_offset_path0(0)
            module.out1_offset_path1(0)

            # Configure the sequencer
            sequencer.mod_en_awg(True)
            sequencer.nco_freq(bb_if_frequency)
            sequencer.sync_en(True)
            sequencer.nco_prop_delay_comp_en(True) # NCO delay compensation

        elif module.is_qrm_type:
            # Configure the LO
            module.out0_in0_lo_en(True)  # Switch the LO on
            module.out0_in0_lo_freq(bb_frequency)

            # Configure attenuation
            module.out0_att(20)
            module.in0_att(0)

            # Set offset in mV
            module.out0_offset_path0(0)
            module.out0_offset_path1(0)

            # Configure hw averaging
            module.scope_acq_avg_mode_en_path0(True)
            module.scope_acq_avg_mode_en_path1(True)

            # Configure scope mode
            module.scope_acq_sequencer_select(sequencer.seq_idx)
            module.scope_acq_trigger_mode_path0("sequencer")
            module.scope_acq_trigger_mode_path1("sequencer")

            # Configure the sequencer
            sequencer.mod_en_awg(True)
            sequencer.demod_en_acq(True)
            sequencer.nco_freq(bb_if_frequency)
            sequencer.integration_length_acq(1000)
            sequencer.sync_en(True)
            sequencer.nco_prop_delay_comp_en(True) # NCO delay compensation

        return module, sequencer

    def connect(self):
        if self._driver is None or not Cluster.is_valid(self._driver):
            self._driver: Cluster = Cluster(
                name=self.name,
                identifier=self.address,
                dummy_cfg=self.cfg if self.address is None else
                None  # Ignore dummy config if an address is given
            )
            self._driver.reset()
        log.info(self._driver.get_system_state())
        super().connect()

    def clear(self):
        self.active_components.clear()
        for module in self._driver.modules:
            if module.present():
                module.disconnect_outputs()  # For both QCM(RF) and QRM(RF) types
                if module.is_qrm_type:
                    module.disconnect_inputs()  # Only for QRM(RF)

    def set_data(self, seq_files: Dict[PulseChannel, SequenceFile]):
        self.clear()
        for pulse_channel, seq_file in seq_files.items():
            module, sequencer = self.configure_components(pulse_channel)

            # Save module and sequencer as active for this experiment
            self.active_components.setdefault(pulse_channel, []).append((module, sequencer))

            sequence = seq_file.as_dict()
            # Temp dump for investigation of the sequence objects
            with open(
                f"sequence_{module.slot_idx}_{sequencer.seq_idx}_@_{datetime.utcnow()}.json",
                "w"
            ) as f:
                f.write(json.dumps(sequence))

            log.info(f"Uploading sequence to {module}, sequencer {sequencer}")
            sequencer.sequence(sequence)

    def start_playback(self, repetitions: int, repetition_time: float):
        if not any(self.active_components):
            raise ValueError("No active components found")

        results = {}
        for pulse_channel, components in self.active_components.items():
            physical_channel = pulse_channel.physical_channel
            for module, sequencer in components:
                if module.is_qrm_type:
                    sequencer.delete_acquisition_data(all=True)

                module.arm_sequencer(sequencer.seq_idx)
                module.start_sequencer(sequencer.seq_idx)
                log.info(module.get_sequencer_state(sequencer.seq_idx, timeout=1))

                if module.is_qrm_type:
                    acquisitions = self.get_acquisitions(module, sequencer)
                    for acq_name, acq in acquisitions.items():
                        i = np.array(acq["acquisition"]["bins"]["integration"]["path0"])
                        q = np.array(acq["acquisition"]["bins"]["integration"]["path1"])
                        if physical_channel.id in results:
                            raise ValueError("Two or more pulse channels on the same physical channel")
                        results[physical_channel.id] = (i + 1j * q) / sequencer.integration_length_acq()

        return results


class QbloxLiveEngine(LiveDeviceEngine):
    def __init__(self, model: LiveHardwareModel):
        super().__init__(model)

    def startup(self):
        if self.model.control_hardware is None:
            raise ValueError(f"Please add a control hardware first!")
        self.model.control_hardware.connect()

    def shutdown(self):
        if self.model.control_hardware is None:
            raise ValueError(f"Please add a control hardware first!")
        self.model.control_hardware.close()

    def _execute_on_hardware(self, sweep_iterator: SweepIterator, package: QatFile):
        """
        Result processing is defaulted away until I know more.
        """
        if self.model.control_hardware is None:
            raise ValueError("Please add a control hardware first!")

        results = {}
        while not sweep_iterator.is_finished():
            sweep_iterator.do_sweep(package.instructions)

            position_map = self.create_duration_timeline(package.instructions)

            seq_files = SequenceEmitter().emit(package)
            aq_map = self.build_acquire_list(position_map)
            self.model.control_hardware.set_data(seq_files)

            repetitions = package.repeat.repeat_count
            repetition_time = package.repeat.repetition_period

            for aqs in aq_map.values():
                if len(aqs) > 1:
                    raise ValueError(
                        "Multiple acquisitions are not supported on the same channel in one sweep step"
                    )
                for aq in aqs:
                    physical_channel = aq.physical_channel
                    dt = physical_channel.sample_time
                    physical_channel.readout_start = aq.start * dt + aq.delay
                    physical_channel.readout_length = aq.samples * dt
                    physical_channel.acquire_mode_integrator = aq.mode == AcquireMode.INTEGRATOR

            playback_results: Dict[str, np.ndarray] = self.model.control_hardware.start_playback(
                repetitions=repetitions, repetition_time=repetition_time
            )

            for channel, aqs in aq_map.items():
                if len(aqs) > 1:
                    raise ValueError(
                        "Multiple acquisitions are not supported on the same channel in one sweep step"
                    )
                for aq in aqs:
                    response = playback_results[aq.physical_channel.id]
                    response_axis = get_axis_map(aq.mode, response)
                    for pp in package.get_pp_for_variable(aq.output_variable):
                        response, response_axis = self.run_post_processing(pp, response, response_axis)
                    var_result = results.setdefault(
                        aq.output_variable,
                        np.empty(
                            sweep_iterator.get_results_shape(response.shape),
                            response.dtype
                        )
                    )
                    sweep_iterator.insert_result_at_sweep_position(var_result, response)

        return results
