# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
from abc import ABC
from bisect import insort
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from itertools import chain
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np

from qat.purr.backends.qblox.analysis_passes import (
    BindingResult,
    CFGPass,
    CFGResult,
    IterBound,
    QbloxLegalisationPass,
    ReadWriteResult,
    ScopingResult,
    TriageResult,
)
from qat.purr.backends.qblox.codegen_base import DfsTraversal
from qat.purr.backends.qblox.config import SequencerConfig
from qat.purr.backends.qblox.constants import Constants
from qat.purr.backends.qblox.graph import ControlFlowGraph
from qat.purr.backends.qblox.ir import Opcode, Sequence, SequenceBuilder
from qat.purr.backends.qblox.metrics_base import MetricsManager
from qat.purr.backends.qblox.pass_base import AnalysisPass, InvokerMixin, PassManager, QatIR
from qat.purr.backends.qblox.result_base import ResultManager
from qat.purr.backends.utilities import evaluate_shape
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import PulseChannel, PulseShapeType
from qat.purr.compiler.emitter import QatFile
from qat.purr.compiler.instructions import (
    Acquire,
    CustomPulse,
    Delay,
    DeviceUpdate,
    FrequencyShift,
    Id,
    Instruction,
    MeasurePulse,
    PhaseReset,
    PhaseShift,
    PostProcessing,
    Pulse,
    QuantumInstruction,
    Repeat,
    Sweep,
    Synchronize,
    Variable,
    Waveform,
    calculate_duration,
)
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


@dataclass
class QbloxPackage:
    target: PulseChannel = None
    sequence: Sequence = None
    sequencer_config: SequencerConfig = field(default_factory=lambda: SequencerConfig())
    timeline: np.ndarray = None


@dataclass
class AllocationManager:
    _reg_pool: List[str] = field(
        default_factory=lambda: sorted(
            f"R{index}" for index in range(Constants.NUMBER_OF_REGISTERS)
        )
    )
    _lbl_counters: Dict[str, int] = field(default_factory=dict)

    registers: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)

    def reg_alloc(self, name: str) -> str:
        if name in self.registers:
            log.warning(f"Returning a register already allocated for Variable {name}")
            register = self.registers[name]
        elif len(self._reg_pool) < 1:
            raise IndexError(
                "Out of registers. Attempting to use more registers than available in the Q1 sequence processor"
            )
        else:
            register = self._reg_pool.pop(0)
            self.registers[name] = register

        return register

    def reg_free(self, register: str) -> None:
        if register in self._reg_pool:
            raise RuntimeError(f"Cannot free register '{register}' as it's not in use")
        insort(self._reg_pool, register)
        self.registers = {
            var: reg for var, reg in self.registers.items() if reg != register
        }

    @contextmanager
    def reg_borrow(self, name: str):
        """
        Short-lived register allocation
        """

        register = self.reg_alloc(name)
        yield register
        self.reg_free(register)

    def label_gen(self, name: str):
        counter = self._lbl_counters.setdefault(name, 0)
        self._lbl_counters[name] += 1
        label = f"{name}_{counter}"
        self.labels[name] = label
        return label


class AbstractContext(ABC):
    def __init__(self, alloc_mgr: AllocationManager = None):
        self.alloc_mgr = alloc_mgr or AllocationManager()
        self.sequence_builder = SequenceBuilder()
        self.sequencer_config = SequencerConfig()

        self._num_hw_avg = 1  # Technically disabled
        self._wf_memory: int = Constants.MAX_SAMPLE_SIZE_WAVEFORMS
        self._wf_index: int = 0
        self._acq_index: int = 0
        self._wgt_index: int = 0

        self._frequency: float = 0.0  # Tracks frequency shifts on the target

    def create_package(self, target: PulseChannel):
        sequence = self.sequence_builder.build()
        return QbloxPackage(target, sequence, self.sequencer_config)

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

    @contextmanager
    def _loop(self, label: str, count: int = 1):
        """
        This context can be used to avoid unrolling loops but needs manual addition of the 0 iteration
        either immediately after the yield or immediately after the context has exited.
        """

        label = self.alloc_mgr.label_gen(label)
        register = self.alloc_mgr.reg_alloc(label)
        self.sequence_builder.move(count, register)
        self.sequence_builder.nop()
        self.sequence_builder.label(label)
        yield register
        self.sequence_builder.loop(register, label)
        self.alloc_mgr.reg_free(register)

    @contextmanager
    def _modulo_reg(self, a: str, n: int):
        """
        Helpful context that prepares a temporary register b such that a ≅ b [n] where
        n is some immediate constant batch size.
        """

        if not isinstance(n, int) or n < 2:
            raise ValueError(
                f"Invalid batch size. Expected an integer >= 2, but got {n} instead"
            )

        batch_enter = self.alloc_mgr.label_gen("batch_enter")
        batch_exit = self.alloc_mgr.label_gen("batch_exit")

        iter_reg = self.alloc_mgr.reg_alloc("tmp")
        if a != iter_reg:
            self.sequence_builder.move(
                a,
                iter_reg,
                f"Avoids cluttering {a} as it's likely used as an accumulator",
            )
            self.sequence_builder.nop()

        self.sequence_builder.jlt(iter_reg, n, batch_exit)
        self.sequence_builder.label(batch_enter)
        self.sequence_builder.sub(iter_reg, n, iter_reg)
        self.sequence_builder.nop()
        self.sequence_builder.jge(iter_reg, n, batch_enter)
        self.sequence_builder.label(batch_exit)
        yield iter_reg
        self.alloc_mgr.reg_free(iter_reg)

    def _wait_imm(self, duration: int):
        """
        Waits for `duration` nanoseconds expressed as an immediate.

        `duration` must be positive
        """

        if duration < 0:
            raise ValueError(f"Duration must be positive, got {duration} instead")

        quotient = duration // Constants.MAX_WAIT_TIME
        remainder = duration % Constants.MAX_WAIT_TIME

        if quotient >= Constants.LOOP_UNROLL_THRESHOLD:
            with self._loop("wait_label", quotient):
                self.sequence_builder.wait(Constants.MAX_WAIT_TIME)
        elif quotient >= 1:
            for _ in range(quotient):
                self.sequence_builder.wait(Constants.MAX_WAIT_TIME)

        if remainder >= Constants.GRID_TIME:
            self.sequence_builder.wait(remainder)
        else:
            log.debug(
                f"Ignoring (remainder) {duration} ns as it's less than the threshold {Constants.GRID_TIME} ns"
            )
            return

    def _wait_reg(self, duration: str):
        """
        A mere `wait RX` in general has undefined runtime behaviour.

        This is a useful helper to dynamically wait for a `duration` nanoseconds expressed as a register.
        `iter_reg` is a short-lived register only used as an interator. Customer is responsible for (de)allocation.
        """

        batch_enter = self.alloc_mgr.label_gen("batch_enter")
        batch_exit = self.alloc_mgr.label_gen("batch_exit")
        remainder = self.alloc_mgr.label_gen("remainder")

        with self.alloc_mgr.reg_borrow("tmp") as iter_reg:
            if duration != iter_reg:
                self.sequence_builder.move(
                    duration,
                    iter_reg,
                    f"Avoids cluttering {duration} as it's likely used as an accumulator",
                )
                self.sequence_builder.nop()

            self.sequence_builder.jlt(iter_reg, Constants.GRID_TIME, batch_exit)
            self.sequence_builder.jlt(iter_reg, Constants.MAX_WAIT_TIME, remainder)
            self.sequence_builder.label(batch_enter)
            self.sequence_builder.wait(Constants.MAX_WAIT_TIME)
            self.sequence_builder.sub(iter_reg, Constants.MAX_WAIT_TIME, iter_reg)
            self.sequence_builder.nop()
            self.sequence_builder.jge(iter_reg, Constants.MAX_WAIT_TIME, batch_enter)
            self.sequence_builder.jlt(iter_reg, Constants.GRID_TIME, batch_exit)
            self.sequence_builder.label(remainder)
            self.sequence_builder.wait(iter_reg)
            self.sequence_builder.label(batch_exit)

    def _upd_param_imm(self, duration: int):
        if duration < Constants.GRID_TIME:
            log.debug(
                f"Rounding up duration {duration} ns as it's less than the threshold {Constants.GRID_TIME} ns"
            )
            duration = Constants.GRID_TIME

        self.sequence_builder.upd_param(Constants.GRID_TIME)
        self._wait_imm(duration - Constants.GRID_TIME)

    def _upd_param_reg(self, duration: str):
        """
        Update latched parameters and wait for `duration` nanoseconds expressed as a register

        If `duration` is less than the threshold Constants.GRID_TIME, the generates assembly rounds up
        and waits for Constants.GRID_TIME.
        """

        batch_enter = self.alloc_mgr.label_gen("batch_enter")
        batch_exit = self.alloc_mgr.label_gen("batch_exit")
        remainder = self.alloc_mgr.label_gen("remainder")

        self.sequence_builder.upd_param(Constants.GRID_TIME)
        with self.alloc_mgr.reg_borrow("tmp") as iter_reg:
            self.sequence_builder.jlt(
                duration,
                Constants.GRID_TIME,
                batch_exit,
                "Guards against underflow likely causable by the following subtraction",
            )
            self.sequence_builder.sub(duration, Constants.GRID_TIME, iter_reg)
            self.sequence_builder.nop()

            self.sequence_builder.jlt(iter_reg, Constants.GRID_TIME, batch_exit)
            self.sequence_builder.jlt(iter_reg, Constants.MAX_WAIT_TIME, remainder)
            self.sequence_builder.label(batch_enter)
            self.sequence_builder.wait(Constants.MAX_WAIT_TIME)
            self.sequence_builder.sub(iter_reg, Constants.MAX_WAIT_TIME, iter_reg)
            self.sequence_builder.nop()
            self.sequence_builder.jge(iter_reg, Constants.MAX_WAIT_TIME, batch_enter)
            self.sequence_builder.jlt(iter_reg, Constants.GRID_TIME, batch_exit)
            self.sequence_builder.label(remainder)
            self.sequence_builder.wait(iter_reg)
            self.sequence_builder.label(batch_exit)

    def _evaluate_waveform(self, waveform: Waveform, target: PulseChannel):
        """
        The waveform is evaluated as a 1d complex array. In QBlox, the Real and Imag parts of the pulse
        represent the digital offset on the AWG. They both must be within range [-1, 1].
        """

        num_samples = int(calculate_duration(waveform))
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

    def _register_weight(self, name: str, weight: np.ndarray):
        wgt_index = self._wgt_index
        self.sequence_builder.add_weight(name, wgt_index, weight.tolist())
        self._wgt_index = wgt_index + 1
        return wgt_index

    @contextmanager
    def _wrapper_pulse(
        self,
        delay_seconds,
        pulse_shape,
        i_steps,
        q_steps,
        i_index,
        q_index,
    ):
        flight_nanos = max(int(delay_seconds * 1e9), Constants.GRID_TIME)
        if pulse_shape == PulseShapeType.SQUARE:
            self.sequence_builder.set_awg_offs(i_steps, q_steps)
            self.sequence_builder.upd_param(flight_nanos)
        else:
            self.sequence_builder.play(i_index, q_index, flight_nanos)
        yield
        if pulse_shape == PulseShapeType.SQUARE:
            self.sequence_builder.set_awg_offs(0, 0)
            self.sequence_builder.upd_param(Constants.GRID_TIME)

    def id(self):
        self.sequence_builder.nop()

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
        value = QbloxLegalisationPass.freq_as_steps(shifted_nco_freq)  # 1 MHZ <-> 4e6
        self.sequence_builder.set_freq(value)
        self.sequence_builder.upd_param(Constants.GRID_TIME)


class QbloxContext(AbstractContext):
    def __init__(self, repeat: Repeat):
        super().__init__()

        self._repeat_count = repeat.repeat_count
        self._repeat_period = repeat.repetition_period
        self._repeat_reg = None
        self._repeat_label = None

        self._duration: float = 0.0
        self._phase: float = 0.0  # Tracks phase shifts on the target
        self._timeline: np.ndarray = np.empty(0, dtype=complex)

    def __enter__(self):
        self.clear()
        self._repeat_reg = self.alloc_mgr.reg_alloc("shot")
        self._repeat_label = self.alloc_mgr.label_gen("shot")

        self.sequence_builder.set_mrk(3)
        self.sequence_builder.upd_param(Constants.GRID_TIME)
        self.sequence_builder.move(0, self._repeat_reg, "Shot / Repeat iteration")
        self.sequence_builder.label(self._repeat_label)
        self.sequence_builder.reset_ph("Reset phase at the beginning of shot")
        self.sequence_builder.upd_param(Constants.GRID_TIME)
        self.sequence_builder.wait_sync(
            Constants.GRID_TIME, "Sync at the beginning of shot"
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._wait_imm(int(self._repeat_period * 1e9))
        self.sequence_builder.add(
            self._repeat_reg, 1, self._repeat_reg, "Increment Shot / Repeat iterator"
        )
        self.sequence_builder.nop()
        self.sequence_builder.jlt(self._repeat_reg, self._repeat_count, self._repeat_label)
        self.sequence_builder.stop()

        self.alloc_mgr.reg_free(self._repeat_reg)
        self._repeat_label = None
        self._repeat_reg = None

    @property
    def duration(self):
        return self._duration

    def create_package(self, target: PulseChannel):
        sequence = self.sequence_builder.build()
        return QbloxPackage(target, sequence, self.sequencer_config, self._timeline)

    def _register_acquisition(self, acquire: Acquire):
        acq_index = self._acq_index
        self.sequence_builder.add_acquisition(
            acquire.output_variable, acq_index, self._repeat_count
        )
        self._acq_index = acq_index + 1
        return acq_index

    def delay(self, inst: Delay):
        duration = int(calculate_duration(inst))
        self._wait_imm(duration)

        self._duration = self._duration + inst.duration
        self._timeline = np.append(self._timeline, np.zeros(duration, dtype=complex))

    def waveform(self, waveform: Waveform, target: PulseChannel):
        pulse = self._evaluate_waveform(waveform, target)
        pulse_width = pulse.size
        if pulse_width < Constants.GRID_TIME:
            log.debug(
                f"""
                Minimum pulse width is {Constants.GRID_TIME} ns with a resolution of {1} ns.
                Please round up the width to at least {Constants.GRID_TIME} nanoseconds.
                This pulse will be ignored.
                """
            )
            return

        if isinstance(waveform, Pulse) and waveform.shape == PulseShapeType.SQUARE:
            i_steps = int(pulse[0].real * Constants.MAX_OFFSET)
            q_steps = int(pulse[0].imag * Constants.MAX_OFFSET)
            self.sequence_builder.set_awg_offs(i_steps, q_steps)
            self._upd_param_imm(pulse_width)
            self.sequence_builder.set_awg_offs(0, 0)
            self.sequence_builder.upd_param(Constants.GRID_TIME)
        else:
            i_index, q_index = self._register_waveform(waveform, target, pulse)
            max_width = min(pulse_width, Constants.MAX_WAIT_TIME)
            self.sequence_builder.play(i_index, q_index, max_width)
            self._wait_imm(pulse_width - max_width)

        self._duration = self._duration + waveform.duration
        self._timeline = np.append(self._timeline, pulse * np.exp(1.0j * self._phase))

    def measure_acquire(
        self, measure: MeasurePulse, acquire: Acquire, target: PulseChannel
    ):
        pulse = self._evaluate_waveform(measure, target)
        pulse_width = pulse.size
        if pulse_width < Constants.GRID_TIME:
            log.debug(
                f"""
                Minimum pulse width is {Constants.GRID_TIME} ns with a resolution of {1} ns.
                Please round up the width to at least {Constants.GRID_TIME} nanoseconds.
                This pulse will be ignored. 
                """
            )
            return

        acq_index = self._register_acquisition(acquire)
        acq_width = int(calculate_duration(acquire))
        self.sequencer_config.square_weight_acq.integration_length = acq_width
        self.sequencer_config.thresholded_acq.threshold = acquire.threshold
        self.sequencer_config.thresholded_acq.rotation = acquire.rotation

        i_steps, q_steps = None, None
        i_index, q_index = None, None
        if measure.shape == PulseShapeType.SQUARE:
            i_steps = int(pulse[0].real * Constants.MAX_OFFSET)
            q_steps = int(pulse[0].imag * Constants.MAX_OFFSET)
        else:
            i_index, q_index = self._register_waveform(measure, target, pulse)

        with self._wrapper_pulse(
            acquire.delay, measure.shape, i_steps, q_steps, i_index, q_index
        ):
            if acquire.filter:
                weight = acquire.filter.samples
                wgt_i_index = self._register_weight(
                    f"{acquire.output_variable}_weight_I", weight.real
                )
                wgt_q_index = self._register_weight(
                    f"{acquire.output_variable}_weight_Q", weight.imag
                )

                # TODO - silly convoluted way of using the `acquire_weighed` instruction
                # TODO - simplify when Qblox supports more freedom of memory addressing
                with (
                    self.alloc_mgr.reg_borrow("wgt_i") as wgt_i_reg,
                    self.alloc_mgr.reg_borrow("wgt_q") as wgt_q_reg,
                ):
                    self.sequence_builder.move(wgt_i_index, wgt_i_reg)
                    self.sequence_builder.move(wgt_q_index, wgt_q_reg)

                self.sequence_builder.acquire_weighed(
                    acq_index, self._repeat_reg, wgt_i_reg, wgt_q_reg, acq_width
                )
            else:
                self.sequence_builder.acquire(acq_index, self._repeat_reg, acq_width)

        self._duration = self._duration + measure.duration
        self._timeline = np.append(self._timeline, pulse * np.exp(1.0j * self._phase))

    @staticmethod
    def reset_phase(inst: PhaseReset, contexts: Dict):
        for target in inst.quantum_targets:
            cxt = contexts[target]
            cxt.sequence_builder.reset_ph()
            cxt.sequence_builder.upd_param(Constants.GRID_TIME)

            cxt._phase = 0.0

    def shift_phase(self, inst: PhaseShift):
        value = QbloxLegalisationPass.phase_as_steps(inst.phase)
        self.sequence_builder.set_ph_delta(value)
        self.sequence_builder.upd_param(Constants.GRID_TIME)

        self._phase = self._phase + inst.phase

    @staticmethod
    def synchronize(inst: Synchronize, contexts: Dict):
        # TODO - For now, enable only logical time padding
        # TODO - Enable when finer grained SYNC groups are supported
        max_duration = max([cxt.duration for cxt in contexts.values()])
        for target in inst.quantum_targets:
            cxt = contexts[target]
            delay_time = max_duration - cxt.duration
            cxt.delay(Delay(target, delay_time))
            # cxt.sequence_builder.wait_sync(Constants.GRID_TIME)


class NewQbloxContext(AbstractContext):
    def __init__(
        self,
        alloc_mgr: AllocationManager,
        scoping_result: ScopingResult,
        rw_result: ReadWriteResult,
        iter_bounds: Dict[str, IterBound],
    ):
        super().__init__(alloc_mgr)
        self.scope2symbols: Dict[Tuple[Instruction, Optional[Instruction]], Set[str]] = (
            scoping_result.scope2symbols
        )
        self.symbol2scopes: Dict[str, List[Tuple[Instruction, Optional[Instruction]]]] = (
            scoping_result.symbol2scopes
        )
        self.reads: Dict[str, List[Instruction]] = rw_result.reads
        self.writes: Dict[str, List[Instruction]] = rw_result.writes
        self.iter_bounds = iter_bounds

        self._durations: List[Union[float, str]] = []
        self._phases: List[Union[float, str]] = []

    def _register_acquisition(self, acquire: Acquire):
        name = f"acquire_{hash(acquire)}"
        num_bins = self.iter_bounds[name].count
        acq_index = self._acq_index
        self.sequence_builder.add_acquisition(acquire.output_variable, acq_index, num_bins)
        self._acq_index = acq_index + 1
        return acq_index

    def delay(self, inst: Delay):
        value = inst.duration

        if isinstance(value, Variable):
            register = self.alloc_mgr.registers[value.name]
            self._wait_reg(register)
        else:
            duration = int(calculate_duration(inst))
            self._wait_imm(duration)

        self._durations.append(value)

    def waveform(self, waveform: Waveform, target: PulseChannel):
        attr2var = {
            attr: var
            for attr, var in waveform.__dict__.items()
            if isinstance(var, Variable)
        }

        if isinstance(waveform, Pulse) and waveform.shape == PulseShapeType.SQUARE:
            if "amp" in attr2var:
                pulse_amp = self.alloc_mgr.registers[attr2var["amp"].name]

                i_steps = pulse_amp
                q_steps = self.alloc_mgr.registers["zero"]
            else:
                bias = target.bias
                scale = 1.0 + 0.0j if waveform.ignore_channel_scale else target.scale

                pulse_amp = waveform.scale_factor * waveform.amp
                pulse_amp = scale * pulse_amp + bias

                i_steps = int(pulse_amp.real * Constants.MAX_OFFSET)
                q_steps = int(pulse_amp.imag * Constants.MAX_OFFSET)

            self.sequence_builder.set_awg_offs(i_steps, q_steps)

            if "width" in attr2var:
                pulse_width = self.alloc_mgr.registers[attr2var["width"].name]
                self._upd_param_reg(pulse_width)
            else:
                pulse_width = int(calculate_duration(waveform))
                self._upd_param_imm(pulse_width)

            self.sequence_builder.set_awg_offs(0, 0)
            self.sequence_builder.upd_param(Constants.GRID_TIME)
        else:
            pulse = self._evaluate_waveform(waveform, target)
            pulse_width = pulse.size
            if pulse_width < Constants.GRID_TIME:
                log.debug(
                    f"""
                    Minimum pulse width is {Constants.GRID_TIME} ns with a resolution of {1} ns.
                    Please round up the width to at least {Constants.GRID_TIME} nanoseconds.
                    This pulse will be ignored. 
                    """
                )
                return

            max_width = min(pulse_width, Constants.MAX_WAIT_TIME)
            i_index, q_index = self._register_waveform(waveform, target, pulse)
            self.sequence_builder.play(i_index, q_index, max_width)
            self._wait_imm(pulse_width - max_width)

        self._durations.append(waveform.duration)

    def measure_acquire(
        self, measure: MeasurePulse, acquire: Acquire, target: PulseChannel
    ):
        pulse = self._evaluate_waveform(measure, target)
        pulse_width = pulse.size
        if pulse_width < Constants.GRID_TIME:
            log.debug(
                f"""
                Minimum pulse width is {Constants.GRID_TIME} ns with a resolution of {1} ns.
                Please round up the width to at least {Constants.GRID_TIME} nanoseconds.
                This pulse will be ignored. 
                """
            )
            return

        acq_index = self._register_acquisition(acquire)
        name = f"acquire_{hash(acquire)}"
        bin_reg = self.alloc_mgr.registers[name]
        acq_width = int(calculate_duration(acquire))
        self.sequencer_config.square_weight_acq.integration_length = acq_width
        self.sequencer_config.thresholded_acq.threshold = acquire.threshold
        self.sequencer_config.thresholded_acq.rotation = acquire.rotation

        i_steps, q_steps = None, None
        i_index, q_index = None, None
        if measure.shape == PulseShapeType.SQUARE:
            i_steps = int(pulse[0].real * Constants.MAX_OFFSET)
            q_steps = int(pulse[0].imag * Constants.MAX_OFFSET)
        else:
            i_index, q_index = self._register_waveform(measure, target, pulse)

        with self._wrapper_pulse(
            acquire.delay, measure.shape, i_steps, q_steps, i_index, q_index
        ):
            self.sequence_builder.acquire(acq_index, bin_reg, acq_width)

        self._durations.append(measure.duration)

    def device_update(self, du_inst: DeviceUpdate):
        value = du_inst.value
        if isinstance(value, Variable):
            val_reg = self.alloc_mgr.registers[value.name]

            if du_inst.attribute == "frequency":
                self.sequence_builder.set_freq(val_reg)
                self.sequence_builder.upd_param(Constants.GRID_TIME)
            elif du_inst.attribute == "phase":
                with self._modulo_reg(val_reg, Constants.NCO_MAX_PHASE_STEPS) as iter_reg:
                    self.sequence_builder.set_ph(iter_reg)

                self.sequence_builder.upd_param(Constants.GRID_TIME)

                self._phases.clear()
                self._phases.append(value)
            else:
                raise NotImplementedError(
                    f"Unsupported processing of attribute {du_inst.attribute} for instruction {du_inst}"
                )
        else:
            raise NotImplementedError(
                f"Unsupported processing of immediate values for instruction {du_inst}"
            )

    @staticmethod
    def reset_phase(inst: PhaseReset, contexts: Dict):
        for target in inst.quantum_targets:
            cxt = contexts[target]
            cxt.sequence_builder.reset_ph()
            cxt.sequence_builder.upd_param(Constants.GRID_TIME)

            cxt._phases.clear()

    def shift_phase(self, inst: PhaseShift):
        value = inst.phase

        if isinstance(value, Variable):
            ph_reg = self.alloc_mgr.registers[value.name]
            with self._modulo_reg(ph_reg, Constants.NCO_MAX_PHASE_STEPS) as iter_reg:
                self.sequence_builder.set_ph_delta(iter_reg)
        else:
            ph_imm = QbloxLegalisationPass.phase_as_steps(value)
            self.sequence_builder.set_ph_delta(ph_imm)

        self.sequence_builder.upd_param(Constants.GRID_TIME)

        self._phases.append(value)

    @staticmethod
    def synchronize(inst: Synchronize, contexts: Dict):
        """
        Potential presence of dynamic time render static time padding method inviable.
        In such case, we simply revert to the `wait_sync` instruction (although it may limit muxing capabilities).
        """

        for target in inst.quantum_targets:
            cxt = contexts[target]
            cxt.sequence_builder.wait_sync(Constants.GRID_TIME)

    @staticmethod
    def enter_repeat(inst: Repeat, contexts: Dict):
        iter_name = f"repeat_{hash(inst)}"
        for context in contexts.values():
            register = context.alloc_mgr.registers[iter_name]
            bound = context.iter_bounds[iter_name]
            context.sequence_builder.move(bound.start, register, "Shot / Repeat iteration")

            label = context.alloc_mgr.labels[iter_name]
            context.sequence_builder.label(label)
            context.sequence_builder.reset_ph("Reset phase at the beginning of shot")
            context.sequence_builder.upd_param(Constants.GRID_TIME)
            context.sequence_builder.wait_sync(
                Constants.GRID_TIME, "Sync at the beginning of shot"
            )

    @staticmethod
    def exit_repeat(inst: Repeat, contexts: Dict):
        iter_name = f"repeat_{hash(inst)}"
        for context in contexts.values():
            register = context.alloc_mgr.registers[iter_name]
            label = context.alloc_mgr.labels[iter_name]
            bound = context.iter_bounds[iter_name]

            context._wait_imm(int(inst.repetition_period * 1e9))
            context.sequence_builder.add(
                register, bound.step, register, "Increment Shot / Repeat iterator"
            )
            context.sequence_builder.nop()
            context.sequence_builder.jlt(register, bound.end + bound.step, label)

    @staticmethod
    def enter_sweep(inst: Sweep, contexts: Dict):
        iter_name = f"sweep_{hash(inst)}"
        for context in contexts.values():
            var_names = [n for n in inst.variables if n in context.reads] + [iter_name]
            for name in var_names:
                register = context.alloc_mgr.registers[name]
                bound = context.iter_bounds[name]
                context.sequence_builder.move(
                    bound.start,
                    register,
                    f"Initialise register allocated for variable {name}",
                )

            label = context.alloc_mgr.labels[iter_name]
            context.sequence_builder.label(label)

    @staticmethod
    def exit_sweep(inst: Sweep, contexts: Dict):
        iter_name = f"sweep_{hash(inst)}"
        for context in contexts.values():
            var_names = [n for (n, insts) in context.writes.items() if inst in insts]
            for name in var_names:
                register = context.alloc_mgr.registers[name]
                bound = context.iter_bounds[name]
                context.sequence_builder.add(
                    register,
                    bound.step,
                    register,
                    f"Increment register for variable {name}",
                )
                context.sequence_builder.nop()

            register = context.alloc_mgr.registers[iter_name]
            bound = context.iter_bounds[iter_name]
            label = context.alloc_mgr.labels[iter_name]
            context.sequence_builder.jlt(register, bound.end + bound.step, label)

    @staticmethod
    def prologue(contexts: Dict):
        for context in contexts.values():
            context.sequence_builder.set_mrk(3)
            context.sequence_builder.upd_param(Constants.GRID_TIME)

            # TODO - probably not the right place
            context.alloc_mgr.reg_alloc("zero")

            for name, register in context.alloc_mgr.registers.items():
                context.sequence_builder.move(
                    0,
                    register,
                    f"Precautionary initialisation for variable {name}",
                )

    @staticmethod
    def epilogue(contexts: Dict):
        for context in contexts.values():
            context.sequence_builder.stop()


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


@dataclass
class PreCodegenResult:
    alloc_mgrs: Dict[PulseChannel, AllocationManager] = field(
        default_factory=lambda: defaultdict(lambda: AllocationManager())
    )


class PreCodegenPass(AnalysisPass):
    def run(self, ir: QatIR, res_mgr: ResultManager, *args, **kwargs):
        """
        Precedes assembly codegen.
        Performs a naive register allocation through a manager object.
        Computes useful information in the form of attributes.
        """

        builder = ir.value
        if not isinstance(builder, InstructionBuilder):
            raise ValueError(f"Expected InstructionBuilder, got {type(builder)}")

        triage_result: TriageResult = res_mgr.lookup_by_type(TriageResult)
        binding_result: BindingResult = res_mgr.lookup_by_type(BindingResult)
        result = PreCodegenResult()

        for target in triage_result.target_map:
            alloc_mgr = result.alloc_mgrs[target]
            iter_bound_result = binding_result.iter_bound_results[target]
            reads = binding_result.rw_results[target].reads
            writes = binding_result.rw_results[target].writes

            names = set(chain(*[iter_bound_result.keys(), reads.keys(), writes.keys()]))
            for name in names:
                alloc_mgr.reg_alloc(name)
                alloc_mgr.label_gen(name)

        res_mgr.add(result)


class NewQbloxEmitter(InvokerMixin):
    def build_pass_pipeline(self, *args, **kwargs):
        return PassManager() | PreCodegenPass() | CFGPass()

    def emit_packages(
        self, ir: QatIR, res_mgr: ResultManager, met_mgr: MetricsManager
    ) -> List[QbloxPackage]:
        self.run_pass_pipeline(ir, res_mgr, met_mgr)

        triage_result: TriageResult = res_mgr.lookup_by_type(TriageResult)
        binding_result: BindingResult = res_mgr.lookup_by_type(BindingResult)
        precodegen_result: PreCodegenResult = res_mgr.lookup_by_type(PreCodegenResult)

        scoping_results: Dict[PulseChannel, ScopingResult] = binding_result.scoping_results
        rw_results: Dict[PulseChannel, ReadWriteResult] = binding_result.rw_results
        iter_bound_results: Dict[PulseChannel, Dict[str, IterBound]] = (
            binding_result.iter_bound_results
        )
        alloc_mgrs: Dict[PulseChannel, AllocationManager] = precodegen_result.alloc_mgrs

        contexts = {
            t: NewQbloxContext(
                alloc_mgr=alloc_mgrs[t],
                scoping_result=scoping_results[t],
                rw_result=rw_results[t],
                iter_bounds=iter_bound_results[t],
            )
            for t in triage_result.target_map
        }

        cfg_result: CFGResult = res_mgr.lookup_by_type(CFGResult)
        cfg_walker = QbloxCFGWalker(contexts)
        cfg_walker.walk(cfg_result.cfg)

        # Discard empty contexts
        return [
            context.create_package(target)
            for target, context in contexts.items()
            if not context.is_empty()
        ]


class QbloxCFGWalker(DfsTraversal):
    def __init__(self, contexts: Dict[PulseChannel, NewQbloxContext]):
        super().__init__()
        self.contexts = contexts

    def enter(self, block):
        iterator = block.iterator()
        while (inst := next(iterator, None)) is not None:
            if isinstance(inst, Sweep):
                NewQbloxContext.enter_sweep(inst, self.contexts)
            elif isinstance(inst, Repeat):
                NewQbloxContext.enter_repeat(inst, self.contexts)
            elif isinstance(inst, QuantumInstruction):
                if isinstance(inst, PostProcessing):
                    continue
                elif isinstance(inst, Synchronize):
                    NewQbloxContext.synchronize(inst, self.contexts)
                    continue
                elif isinstance(inst, PhaseReset):
                    NewQbloxContext.reset_phase(inst, self.contexts)
                    continue

                for target in inst.quantum_targets:
                    context = self.contexts[target]
                    if isinstance(inst, DeviceUpdate):
                        context.device_update(inst)
                    if isinstance(inst, MeasurePulse):
                        next_inst = next(iterator, None)
                        if next_inst is None or not isinstance(next_inst, Acquire):
                            raise ValueError(
                                "Found a MeasurePulse but no Acquire instruction followed"
                            )

                        context.measure_acquire(inst, next_inst, target)
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

    def exit(self, block):
        iterator = block.iterator()
        while (inst := next(iterator, None)) is not None:
            if isinstance(inst, Repeat):
                NewQbloxContext.exit_repeat(inst, self.contexts)
            elif isinstance(inst, Sweep):
                NewQbloxContext.exit_sweep(inst, self.contexts)

    def walk(self, cfg: ControlFlowGraph):
        # TODO - run as visit to the entry block
        NewQbloxContext.prologue(self.contexts)
        self.run(cfg)
        # TODO - run as visit to the exit block
        NewQbloxContext.epilogue(self.contexts)
