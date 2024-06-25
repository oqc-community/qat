from abc import ABC
from contextlib import contextmanager
from typing import List, Dict

import numpy as np

from qat.purr.backends.qblox.codegen import calculate_duration
from qat.purr.backends.qblox.ir import Constants, SequenceBuilder
from qat.purr.backends.utilities import evaluate_shape
from qat.purr.compiler.devices import PulseChannel
from qat.purr.compiler.instructions import Sweep, Repeat, Waveform, CustomPulse, \
    Pulse, Acquire, Delay, MeasurePulse, Synchronize, PhaseReset, PhaseShift


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


class RegisterManager:
    def __init__(self):
        self._registers: List[str] = sorted(
            f"R{index}" for index in range(Constants.NUMBER_OF_REGISTERS)
        )

    def reg_alloc(self) -> str:
        if len(self._registers) < 1:
            raise IndexError(
                "Out of registers. Attempting to use more registers "
                "than available in the Q1 sequence processor"
            )
        return self._registers.pop(0)

    def reg_free(self, register: str) -> None:
        if register in self._registers:
            raise RuntimeError(f"Cannot free register '{register}' as it's not in use")
        self._registers.append(register)


class AbstractContext(ABC):
    def __init__(self, builder=None, manager=None, nested=None):
        self.builder = builder or SequenceBuilder()
        self.manager = manager or RegisterManager()
        self.nested: AbstractContext = nested

        self.reg = None
        self.label = None

        self._label_counters = {}

        ###

        self._duration: int = 0
        self._num_hw_avg = 1  # Technically disabled
        self._wf_memory: int = Constants.MAX_SAMPLE_SIZE_WAVEFORMS
        self._wf_index: int = 0
        self._acq_index: int = 0

        self._phase: float = 0.0  # Keeps record of the phase on the target
        self._frequency: float = 0.0  # Keeps record of the shift on the target

    @property
    def duration(self):
        return self._duration

    def is_empty(self):
        return not (any(self.builder.waveforms) or any(self.builder.acquisitions))

    def clear(self):
        self.builder.waveforms.clear()
        self.builder.acquisitions.clear()
        self.builder.weights.clear()
        self.builder.q1asm_instructions.clear()

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
        register = self.manager.reg_alloc()
        label = self._generate_label(label)
        self.builder.move(iter_count, register)
        self.builder.nop()
        self.builder.label(label)
        yield register
        self.builder.loop(register, label)
        self.manager.reg_free(register)

    @contextmanager
    def _inc_jlt(self, label: str, iter_count: int = 1, start=0, step=1):
        """
        The `loop` instruction first subtracts 1 from the register and then exits if that is equal to 0.
        This means the effective range will be [iter_count-1, 1], 0 is not included. This context uses
        the `jlt` jump instruction as it's more flexible than `loop`.

        It allows to incrementally iterate `iter_count` times from `start` (inclusive).
        """
        register = self.manager.reg_alloc()
        label = self._generate_label(label)
        self.builder.move(start, register)
        self.builder.nop()
        self.builder.label(label)
        yield register
        self.builder.add(register, step, register)
        self.builder.nop()
        self.builder.jlt(register, iter_count, label)
        self.manager.reg_free(register)

    def _wait(self, duration: float):
        if duration == 0:
            return
        if duration < 0:
            raise ValueError(f"Wait duration must be positive, got {duration} instead")

        duration_nanos = int(duration * 1e9)
        quotient = duration_nanos // Constants.IMMEDIATE_MAX_WAIT_TIME
        remainder = duration_nanos % Constants.IMMEDIATE_MAX_WAIT_TIME
        if quotient > Constants.LOOP_UNROLL_THRESHOLD:
            with self._loop("wait", quotient):
                self.builder.wait(Constants.IMMEDIATE_MAX_WAIT_TIME)
        elif quotient >= 1:
            for _ in range(quotient):
                self.builder.wait(Constants.IMMEDIATE_MAX_WAIT_TIME)

        if remainder > 0:
            self.builder.wait(remainder)

    def id(self):
        self.builder.nop()

    def delay(self, inst: Delay):
        self._wait(inst.duration)
        self._duration = self._duration + inst.duration

    def _process_waveform(self, waveform: Waveform, target: PulseChannel):
        # TODO - Support pulse stitching using q1asm
        samples = int(calculate_duration(waveform, return_samples=True))
        if samples == 0:
            return None, None, samples

        dt = target.sample_time
        length = samples * dt
        centre = length / 2.0
        t = np.linspace(
            start=-centre + 0.5 * dt, stop=length - centre - 0.5 * dt, num=samples
        )
        pulse = evaluate_shape(waveform, t, self._phase)
        scale = target.scale
        if isinstance(waveform, (Pulse, CustomPulse)) and waveform.ignore_channel_scale:
            scale = 1

        pulse *= scale
        pulse += target.bias

        if 2 * pulse.size > self._wf_memory:  # for both I and Q
            raise ValueError(
                f"No more waveform memory left for {waveform} on pulse channel {target}"
            )

        wf_name = type(waveform).__name__
        if isinstance(waveform, Pulse):
            wf_name = waveform.shape.name

        index = len(self.builder.waveforms)
        i_index, q_index = self._wf_index, self._wf_index + 1
        self.builder.add_waveform(f"{wf_name}_{index}_I", i_index, pulse.real.tolist())
        self.builder.add_waveform(f"{wf_name}_{index}_Q", q_index, pulse.imag.tolist())

        self._wf_memory = self._wf_memory - pulse.size
        self._wf_index = self._wf_index + 2

        return i_index, q_index, pulse.size

    def waveform(self, waveform: Waveform, target: PulseChannel):
        i_index, q_index, num_samples = self._process_waveform(waveform, target)
        if (i_index, q_index, num_samples) == (None, None, 0):
            return
        self.builder.play(i_index, q_index, Constants.GRID_TIME)
        self.builder.wait(num_samples - Constants.GRID_TIME)
        self._duration = self._duration + waveform.duration

    def _process_acquire(self, acquire: Acquire, target: PulseChannel):
        acq_index = self._acq_index
        self.builder.add_acquisition(acquire.output_variable, acq_index, 131072)# self._repeat_count)
        self._acq_index = acq_index + 1
        return acq_index

    def measure_acquire(
        self, measure: MeasurePulse, acquire: Acquire, target: PulseChannel
    ):
        i_index, q_index, num_samples = self._process_waveform(measure, target)
        if (i_index, q_index, num_samples) == (None, None, 0):
            raise ValueError(f"Measure pulse must not be empty")
        self._duration = self._duration + measure.duration
        acq_index = self._process_acquire(acquire, target)

        def _add_binned_acq_instructions():
            self.builder.play(i_index, q_index, Constants.GRID_TIME)
            self._wait(acquire.delay)
            self.builder.acquire(
                acq_index, self.reg, Constants.MAX_SAMPLE_SIZE_SCOPE_ACQUISITIONS
            )

        if self._num_hw_avg <= Constants.LOOP_UNROLL_THRESHOLD:
            for _ in range(self._num_hw_avg):
                _add_binned_acq_instructions()
        else:
            with self._loop("avg", self._num_hw_avg + 1):
                _add_binned_acq_instructions()

    @staticmethod
    def synchronize(inst: Synchronize, contexts: Dict):
        max_duration = max([cxt.duration for cxt in contexts.values()])
        for target in inst.quantum_targets:
            cxt = contexts[target]
            delay_time = max_duration - cxt.duration
            cxt.delay(Delay(target, delay_time))
            cxt.builder.wait_sync(Constants.GRID_TIME)

    @staticmethod
    def reset_phase(inst: PhaseReset, contexts: Dict):
        for target in inst.quantum_targets:
            cxt = contexts[target]
            cxt._phase = 0.0
            cxt.builder.reset_ph()
            cxt.builder.upd_param(Constants.GRID_TIME)

    def shift_phase(self, inst: PhaseShift):
        value = get_nco_phase_arguments(inst.phase)
        self.builder.set_ph_delta(value)
        self.builder.upd_param(Constants.GRID_TIME)
        self._phase = self._phase + inst.phase

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
        self.builder.set_freq(value)
        self.builder.upd_param(Constants.GRID_TIME)


class QbloxContext(AbstractContext):
    def __init__(self, builder=None, manager=None, nested=None):
        super().__init__(builder, manager, nested)

    def __enter__(self):
        self.clear()
        self.builder.set_mrk(3)
        self.builder.upd_param(Constants.GRID_TIME)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.builder.stop()


class RepeatContext(AbstractContext):
    def __init__(self, repeat: Repeat, builder=None, manager=None, nested=None):
        super().__init__(builder, manager, nested)
        self._repeat_count = repeat.repeat_count
        self._repeat_period = repeat.repetition_period

        self.label = self._generate_label("repeat")
        self.reg = None

    def __enter__(self):
        self.reg = self.manager.reg_alloc()

        self.builder.move(0, self.reg)
        self.builder.nop()
        self.builder.label(self.label)
        self.builder.reset_ph()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._wait(self._repeat_period)
        self.builder.add(self.reg, 1, self.reg)
        self.builder.nop()
        self.builder.jlt(self.reg, self._repeat_count, self.label)

        self.manager.reg_free(self.reg)
        self.reg = None


class SweepContext(AbstractContext):
    def __init__(self, sweep: Sweep, builder=None, manager=None, nested=None):
        super().__init__(builder, manager, nested)
        self.name = None
        self.begin = None
        self.end = None
        self.step = None
        self.count = None
        if len(sweep.variables) == 1:
            for name, value in sweep.variables.items():
                self.name = name
                self.begin, self.end, self.step, self.count = self._process_value(value)

        self.label = self._generate_label(f"sweep_{self.name}")
        self.reg = None

    def _process_value(self, value):
        if value is None:
            return None, None, None, None

        if isinstance(value, np.ndarray):
            value = value.astype(int).tolist()
        elif isinstance(value, List):
            value = [int(e) for e in value]

        begin = None
        end = None
        step = None
        count = len(value)

        if value:
            begin = value[0]
            end = value[-1]
            if count > 1:
                step = (end - begin) // (count - 1)

        # for i in range(len(value) - 1):
        #     if not np.isclose(step - (value[i + 1] - value[i]), 0):
        #         raise ValueError(f"Not a regularly partitioned space {value}")

        return begin, end, step, count

    def __enter__(self):
        self.reg = self.manager.reg_alloc()

        self.builder.move(self.begin, self.reg)
        self.builder.nop()
        self.builder.label(self.label)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.builder.add(self.reg, self.step, self.reg)
        self.builder.nop()
        self.builder.jlt(self.reg, self.end, self.label)

        self.manager.reg_free(self.reg)
        self.reg = None
