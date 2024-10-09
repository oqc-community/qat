from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from qat.purr.backends.qblox.config import SequencerConfig
from qat.purr.backends.qblox.constants import Constants
from qat.purr.backends.qblox.ir import Opcode, Sequence, SequenceBuilder
from qat.purr.backends.utilities import evaluate_shape
from qat.purr.compiler.devices import PulseChannel, PulseShapeType
from qat.purr.compiler.emitter import QatFile
from qat.purr.compiler.instructions import (
    Acquire,
    CustomPulse,
    Delay,
    FrequencyShift,
    Id,
    MeasurePulse,
    PhaseReset,
    PhaseShift,
    PostProcessing,
    Pulse,
    QuantumInstruction,
    Repeat,
    Synchronize,
    Waveform,
)
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


def get_nco_phase_arguments(phase_rad: float) -> int:
    phase_deg = np.rad2deg(phase_rad)
    phase_deg %= 360
    return round(phase_deg * Constants.NCO_PHASE_STEPS_PER_DEG)


def get_nco_set_frequency_arguments(frequency_hz: float) -> int:
    frequency_steps = round(frequency_hz * Constants.NCO_FREQ_STEPS_PER_HZ)

    if (
        frequency_steps < -Constants.NCO_FREQ_LIMIT_STEPS
        or frequency_steps > Constants.NCO_FREQ_LIMIT_STEPS
    ):
        min_max_frequency_in_hz = (
            Constants.NCO_FREQ_LIMIT_STEPS / Constants.NCO_FREQ_STEPS_PER_HZ
        )
        raise ValueError(
            f"NCO frequency must be in [-{min_max_frequency_in_hz:e}, {min_max_frequency_in_hz:e}] Hz. "
            f"Got {frequency_hz:e} Hz"
        )

    return frequency_steps


def calculate_duration(instruction, return_samples: bool = True):
    """
    Calculates the duration of the instruction. The duration is either in nanoseconds
    or in number of samples.
    """
    target_channels = [
        target for target in instruction.quantum_targets if isinstance(target, PulseChannel)
    ]
    if not any(target_channels):
        return 0

    # TODO: Allow for multiple pulse channel targets.
    if len(target_channels) > 1 and not isinstance(instruction, PhaseReset):
        log.warning(
            f"Attempted to calculate duration of {str(instruction)} that has multiple"
            f" target channels. We're arbitrarily using the duration of the first channel "
            f"to calculate instruction duration."
        )

    pc = target_channels[0].physical_channel
    block_size = pc.block_size
    block_time = pc.block_time
    block_number = np.ceil(
        round(instruction.duration / block_time, 4)
    )  # round to remove floating point errors
    if return_samples:
        calc_sample = block_number * block_size
    else:
        calc_sample = block_number * block_time

    return calc_sample


@dataclass
class QbloxPackage:
    target: PulseChannel = None
    sequence: Sequence = None
    sequencer_config: SequencerConfig = field(default_factory=lambda: SequencerConfig())
    timeline: np.ndarray = None


class QbloxEmitter:
    def emit(self, qat_file: QatFile) -> List[QbloxPackage]:
        contexts: Dict[PulseChannel, QbloxContext] = {}

        with ExitStack() as stack:
            inst_iter = iter(qat_file.instructions)
            while (inst := next(inst_iter, None)) is not None:
                if not isinstance(inst, QuantumInstruction):
                    continue  # Ignore classical instructions

                if isinstance(inst, PostProcessing):
                    continue  # Ignore postprocessing

                if isinstance(inst, Synchronize):
                    for target in inst.quantum_targets:
                        if not isinstance(target, PulseChannel):
                            raise ValueError(f"{target} is not a PulseChannel")
                        contexts.setdefault(
                            target, stack.enter_context(QbloxContext(qat_file.repeat))
                        )
                    QbloxContext.synchronize(inst, contexts)
                    continue

                if isinstance(inst, PhaseReset):
                    QbloxContext.reset_phase(inst, contexts)
                    continue

                for target in inst.quantum_targets:
                    if not isinstance(target, PulseChannel):
                        raise ValueError(f"{target} is not a PulseChannel")

                    context = contexts.setdefault(
                        target, stack.enter_context(QbloxContext(qat_file.repeat))
                    )

                    if isinstance(inst, MeasurePulse):
                        acquire = next(inst_iter, None)
                        if acquire is None or not isinstance(acquire, Acquire):
                            raise ValueError(
                                "Found a MeasurePulse but no Acquire instruction followed"
                            )
                        context.measure_acquire(inst, acquire, target)
                    elif isinstance(inst, Waveform):
                        context.waveform(inst, target)
                    elif isinstance(inst, Delay):
                        context.delay(inst)
                    elif isinstance(inst, PhaseShift):
                        context.shift_phase(inst)
                    elif isinstance(inst, FrequencyShift):
                        context.shift_frequency(inst, target)
                    elif isinstance(inst, Id):
                        context.id()

        contexts = self.optimize(contexts)

        return [context.create_package(target) for target, context in contexts.items()]

    def optimize(self, qblox_contexts: Dict) -> Dict:
        # Remove empty contexts
        qblox_contexts = {
            target: context
            for target, context in qblox_contexts.items()
            if not context.is_empty()
        }

        # Remove Opcode.WAIT_SYNC instructions when the experiment contains only a singleton context
        if len(qblox_contexts) == 1:
            context = list(qblox_contexts.values())[0]
            context.sequence_builder.q1asm_instructions = [
                inst
                for inst in context.sequence_builder.q1asm_instructions
                if not inst.opcode == Opcode.WAIT_SYNC
            ]

        return qblox_contexts


class QbloxContext:
    def __init__(self, repeat: Repeat):
        self.sequence_builder = SequenceBuilder()
        self.sequencer_config = SequencerConfig()

        self._repeat_count = repeat.repeat_count
        self._repeat_period = repeat.repetition_period
        self._repeat_reg = None
        self._repeat_label = None

        self._duration: int = 0
        self._timeline: np.ndarray = np.empty(0, dtype=complex)

        self._num_hw_avg = 1  # Technically disabled
        self._wf_memory: int = Constants.MAX_SAMPLE_SIZE_WAVEFORMS
        self._wf_index: int = 0
        self._acq_index: int = 0

        self._frequency: float = 0.0  # Keeps record of the frequency shift on the target

        self._registers: List[str] = sorted(
            f"R{index}" for index in range(Constants.NUMBER_OF_REGISTERS)
        )
        self._label_counters = {}

    def _reg_alloc(self) -> str:
        if len(self._registers) < 1:
            raise IndexError(
                "Out of registers. Attempting to use more registers "
                "than available in the Q1 sequence processor"
            )
        return self._registers.pop(0)

    def _reg_free(self, register: str) -> None:
        if register in self._registers:
            raise RuntimeError(f"Cannot free register '{register}' as it's not in use")
        self._registers.append(register)

    def _generate_label(self, prefix: str):
        counter = self._label_counters.setdefault(prefix, 0)
        self._label_counters[prefix] = counter + 1
        return f"{prefix}_{counter}"

    @contextmanager
    def _loop(self, label: str, iter_count: int = 1):
        """
        This context can be used to avoid unrolling loops but needs manual addition of the 0 iteration
        either immediately after the yield or immediately after the context has exited.
        """
        register = self._reg_alloc()
        label = self._generate_label(label)
        self.sequence_builder.move(iter_count, register)
        self.sequence_builder.nop()
        self.sequence_builder.label(label)
        yield register
        self.sequence_builder.loop(register, label)
        self._reg_free(register)

    @contextmanager
    def _inc_jlt(self, label: str, iter_count: int = 1, start=0, step=1):
        """
        The `loop` instruction first subtracts 1 from the register and then exits if that is equal to 0.
        This means the effective range will be [iter_count-1, 1], 0 is not included. This context uses
        the `jlt` jump instruction as it's more flexible than `loop`.

        It allows to incrementally iterate `iter_count` times from `start` (inclusive).
        """
        register = self._reg_alloc()
        label = self._generate_label(label)
        self.sequence_builder.move(start, register)
        self.sequence_builder.nop()
        self.sequence_builder.label(label)
        yield register
        self.sequence_builder.add(register, step, register)
        self.sequence_builder.nop()
        self.sequence_builder.jlt(register, iter_count, label)
        self._reg_free(register)

    def __enter__(self):
        self.clear()
        self._repeat_reg = self._reg_alloc()
        self._repeat_label = self._generate_label("repeat_label")

        self.sequence_builder.set_mrk(3)
        self.sequence_builder.upd_param(Constants.GRID_TIME)
        self.sequence_builder.move(0, self._repeat_reg)
        self.sequence_builder.nop()
        self.sequence_builder.label(self._repeat_label)
        self.sequence_builder.reset_ph()
        self.sequence_builder.upd_param(Constants.GRID_TIME)
        self.sequence_builder.wait_sync(Constants.GRID_TIME)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._wait_seconds(self._repeat_period)
        self.sequence_builder.add(self._repeat_reg, 1, self._repeat_reg)
        self.sequence_builder.nop()
        self.sequence_builder.jlt(self._repeat_reg, self._repeat_count, self._repeat_label)
        self.sequence_builder.stop()

        self._reg_free(self._repeat_reg)
        self._repeat_label = None
        self._repeat_reg = None

    @property
    def duration(self):
        return self._duration

    def is_empty(self):
        """
        Masks away yet-to-be-supported second-state, cancellation, and cross cancellation targets
        This is temporary and criteria will change with more features coming in
        """
        return not (
            self.sequence_builder.waveforms
            or self.sequence_builder.acquisitions
            or [
                inst
                for inst in self.sequence_builder.q1asm_instructions
                if inst.opcode
                in [
                    Opcode.PLAY,
                    Opcode.SET_AWG_OFFSET,
                    Opcode.SET_AWG_GAIN,
                    Opcode.ACQUIRE,
                    Opcode.ACQUIRE_TTL,
                    Opcode.ACQUIRE_WEIGHED,
                ]
            ]
        )

    def clear(self):
        self.sequence_builder.waveforms.clear()
        self.sequence_builder.acquisitions.clear()
        self.sequence_builder.weights.clear()
        self.sequence_builder.q1asm_instructions.clear()

    def create_package(self, target: PulseChannel):
        sequence = self.sequence_builder.build()
        return QbloxPackage(target, sequence, self.sequencer_config, self._timeline)

    def _wait_seconds(self, duration: float):
        if duration <= 0:
            return

        self._wait_nanoseconds(int(duration * 1e9))

    def _wait_nanoseconds(self, duration: int):
        if duration <= 0:
            return

        quotient = duration // Constants.MAX_WAIT_TIME
        remainder = duration % Constants.MAX_WAIT_TIME
        if quotient > Constants.LOOP_UNROLL_THRESHOLD:
            with self._loop("wait_label", quotient):
                self.sequence_builder.wait(Constants.MAX_WAIT_TIME)
        elif quotient >= 1:
            for _ in range(quotient):
                self.sequence_builder.wait(Constants.MAX_WAIT_TIME)

        if remainder > 0:
            self.sequence_builder.wait(remainder)

    def _evaluate_waveform(self, waveform: Waveform, target: PulseChannel):
        """
        The waveform is evaluated as a 1d complex array. In QBlox, the Real and Imag parts of the pulse
        represent the digital offset on the AWG. They both must be within range [-1, 1].
        """

        num_samples = int(calculate_duration(waveform, return_samples=True))
        if num_samples < Constants.GRID_TIME:
            if num_samples == 0:
                log.warning(f"Empty pulse.")
            else:
                log.warning(
                    f"""
                    Minimum pulse width is {Constants.GRID_TIME} ns with a resolution of {1} ns.
                    Please round up the width to at least {Constants.GRID_TIME} nanoseconds.
                    """
                )
            return None

        dt = target.sample_time
        length = num_samples * dt
        centre = length / 2.0
        t = np.linspace(
            start=-centre + 0.5 * dt, stop=length - centre - 0.5 * dt, num=num_samples
        )
        pulse = evaluate_shape(waveform, t)
        scale = target.scale
        if isinstance(waveform, (Pulse, CustomPulse)) and waveform.ignore_channel_scale:
            scale = 1

        pulse *= scale
        pulse += target.bias

        if ((pulse.real < -1) | (pulse.real > 1)).any():
            raise ValueError(
                "Voltage range for I exceeded. Make sure all I values are within the range [-1, 1]"
            )
        if ((pulse.imag < -1) | (pulse.imag > 1)).any():
            raise ValueError(
                "Voltage range for Q exceeded. Make sure all I values are within the range [-1, 1]"
            )

        if ((pulse.real > 0.3) | (pulse.imag > 0.3)).any():
            log.warning(
                """
                Values above 0.3 will overdrive the mixer and  produce intermodulation distortion.
                Consider adjusting attenuation instead.
                """
            )

        return pulse

    def _register_signal(self, waveform, target, data, name):
        index = self.sequence_builder.lookup_waveform_by_data(data)
        if index is not None:
            log.debug(
                f"Reusing signal {name} at index {index} for pulse {waveform} on channel {target}"
            )
        elif data.size > self._wf_memory:
            raise ValueError(
                f"No more waveform memory left for signal {name} of pulse {waveform} on channel {target}"
            )
        else:
            wf_hash = hash(waveform)
            wf_name = type(waveform).__name__
            if isinstance(waveform, Pulse):
                wf_name = waveform.shape.name

            index = self._wf_index
            self.sequence_builder.add_waveform(
                f"{wf_name}_{wf_hash}_{name}", index, data.tolist()
            )
            self._wf_memory = self._wf_memory - data.size
            self._wf_index = self._wf_index + 1

        return index

    def _register_waveform(self, waveform, target, data):
        i_index = self._register_signal(waveform, target, data.real, "I")
        q_index = self._register_signal(waveform, target, data.imag, "Q")
        return i_index, q_index

    def _register_acquisition(self, acquire: Acquire):
        acq_index = self._acq_index
        self.sequence_builder.add_acquisition(
            acquire.output_variable, acq_index, self._repeat_count
        )
        self._acq_index = acq_index + 1
        return acq_index

    def _binned_acquisition(
        self, delay, acq_index, num_samples, pulse_shape, i_steps, q_steps, i_index, q_index
    ):
        capped_num_samples = min(num_samples, Constants.MAX_WAIT_TIME)
        flight_nanos = int(delay * 1e9)
        capped_flight_nanos = min(flight_nanos, Constants.MAX_WAIT_TIME)
        if pulse_shape == PulseShapeType.SQUARE:
            self.sequence_builder.set_awg_offs(i_steps, q_steps)
            self._wait_seconds(delay)
            self.sequence_builder.acquire(
                acq_index,
                self._repeat_reg,
                capped_num_samples,
            )
            self._wait_seconds((num_samples - capped_num_samples) / 1e9)
            self.sequence_builder.set_awg_offs(0, 0)
            self.sequence_builder.upd_param(Constants.GRID_TIME)
        else:
            self.sequence_builder.play(i_index, q_index, capped_flight_nanos)
            self._wait_seconds((flight_nanos - capped_flight_nanos) / 1e9)
            self.sequence_builder.acquire(
                acq_index,
                self._repeat_reg,
                capped_num_samples,
            )
            self._wait_seconds((num_samples - capped_num_samples) / 1e9)

    def id(self):
        self.sequence_builder.nop()

    def delay(self, inst: Delay):
        self._wait_seconds(inst.duration)

        if inst.duration <= 0:
            return

        self._duration = self._duration + inst.duration
        num_samples = int(calculate_duration(inst))
        self._timeline = np.append(self._timeline, [0] * num_samples)

    def waveform(self, waveform: Waveform, target: PulseChannel):
        pulse = self._evaluate_waveform(waveform, target)
        if pulse is None:
            log.warning("This pulse will be ignored.")
            return

        num_samples = pulse.size
        max_duration = min(num_samples, Constants.MAX_WAIT_TIME)
        if isinstance(waveform, Pulse) and waveform.shape == PulseShapeType.SQUARE:
            i_offs_steps = int(
                pulse[0].real * (Constants.MAX_OFFSET_SIZE // 2)  # Signed integer
            )
            q_offs_steps = int(
                pulse[0].imag * (Constants.MAX_OFFSET_SIZE // 2)  # Signed integer
            )
            self.sequence_builder.set_awg_offs(i_offs_steps, q_offs_steps)
            self.sequence_builder.upd_param(max_duration)
            self._wait_seconds((num_samples - max_duration) / 1e9)
            self.sequence_builder.set_awg_offs(0, 0)
            self.sequence_builder.upd_param(Constants.GRID_TIME)
        else:
            i_index, q_index = self._register_waveform(waveform, target, pulse)
            self.sequence_builder.play(i_index, q_index, max_duration)
            self._wait_seconds((num_samples - max_duration) / 1e9)

        self._duration = self._duration + waveform.duration
        self._timeline = np.append(self._timeline, pulse)

    def measure_acquire(
        self, measure: MeasurePulse, acquire: Acquire, target: PulseChannel
    ):
        pulse = self._evaluate_waveform(measure, target)
        if pulse is None:
            log.warning("This pulse will be ignored.")
            return

        acq_index = self._register_acquisition(acquire)
        self.sequencer_config.square_weight_acq.integration_length = calculate_duration(
            acquire
        )

        i_steps, q_steps = None, None
        i_index, q_index = None, None
        if measure.shape == PulseShapeType.SQUARE:
            i_steps = int(pulse[0].real * (Constants.MAX_OFFSET_SIZE // 2))
            q_steps = int(pulse[0].imag * (Constants.MAX_OFFSET_SIZE // 2))
        else:
            i_index, q_index = self._register_waveform(measure, target, pulse)

        if self._num_hw_avg <= Constants.LOOP_UNROLL_THRESHOLD:
            for _ in range(self._num_hw_avg):
                self._binned_acquisition(
                    acquire.delay,
                    acq_index,
                    pulse.size,
                    measure.shape,
                    i_steps,
                    q_steps,
                    i_index,
                    q_index,
                )
        else:
            with self._loop("avg_label", self._num_hw_avg + 1):
                self._binned_acquisition(
                    acquire.delay,
                    acq_index,
                    pulse.size,
                    measure.shape,
                    i_steps,
                    q_steps,
                    i_index,
                    q_index,
                )

        self._duration = self._duration + measure.duration
        self._timeline = np.append(self._timeline, pulse)

    @staticmethod
    def synchronize(inst: Synchronize, contexts: Dict):
        max_duration = max([cxt.duration for cxt in contexts.values()])
        for target in inst.quantum_targets:
            cxt = contexts[target]
            delay_time = max_duration - cxt.duration
            cxt.delay(Delay(target, delay_time))
            # TODO - For now, enable only logical time padding
            # TODO - Enable when finer grained SYNC groups are supported
            # cxt.sequence_builder.wait_sync(Constants.GRID_TIME)

    @staticmethod
    def reset_phase(inst: PhaseReset, contexts: Dict):
        for target in inst.quantum_targets:
            cxt = contexts[target]
            cxt._phase = 0.0
            cxt.sequence_builder.reset_ph()
            cxt.sequence_builder.upd_param(Constants.GRID_TIME)

    def shift_phase(self, inst: PhaseShift):
        value = get_nco_phase_arguments(inst.phase)
        self.sequence_builder.set_ph_delta(value)
        self.sequence_builder.upd_param(Constants.GRID_TIME)

    def shift_frequency(self, inst, target):
        # TODO - discuss w/t quantum_target.fixed_if is True or False
        old_frequency = target.frequency + self._frequency
        new_frequency = old_frequency + inst.frequency
        if new_frequency < target.min_frequency or new_frequency > target.max_frequency:
            raise ValueError(
                f"Cannot shift pulse channel frequency from '{old_frequency}' to '{new_frequency}'"
            )

        self._frequency = self._frequency + inst.frequency
        shifted_nco_freq = target.baseband_if_frequency + self._frequency
        value = get_nco_set_frequency_arguments(shifted_nco_freq)  # 1 MHZ <-> 4e6
        self.sequence_builder.set_freq(value)
        self.sequence_builder.upd_param(Constants.GRID_TIME)
