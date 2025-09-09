# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd

from abc import ABC
from collections import OrderedDict
from contextlib import ExitStack, contextmanager
from numbers import Number
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from compiler_config.config import CompilerConfig

from qat.backend.base import BaseBackend
from qat.backend.codegen_base import DfsTraversal
from qat.backend.passes.purr.analysis import (
    CFGPass,
    CFGResult,
    ReadWriteResult,
    ScopingResult,
    TILegalisationPass,
)
from qat.backend.qblox.config.constants import Constants
from qat.backend.qblox.config.specification import SequencerConfig
from qat.backend.qblox.execution import QbloxExecutable, QbloxPackage
from qat.backend.qblox.ir import Opcode, SequenceBuilder
from qat.backend.qblox.passes.analysis import (
    AllocationManager,
    BindingResult,
    IterBound,
    PreCodegenPass,
    PreCodegenResult,
    QbloxLegalisationPass,
    TriageResult,
)
from qat.core.metrics_base import MetricsManager
from qat.core.pass_base import InvokerMixin, PassManager
from qat.core.result_base import ResultManager
from qat.purr.backends.qblox.live import QbloxLiveHardwareModel
from qat.purr.backends.utilities import evaluate_shape
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import PulseChannel, PulseShapeType
from qat.purr.compiler.execution import DeviceInjectors, SweepIterator
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
    PostProcessType,
    ProcessAxis,
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

        self._durations: List[Union[float, str]] = []
        self._frequency: float = 0.0  # Tracks frequency shifts on the target
        self._phases: List[Union[float, str]] = []
        self._timeline: np.ndarray = np.empty(0, dtype=complex)

    @property
    def durations(self):
        return self._durations

    @property
    def duration(self):
        if all((isinstance(d, Number) for d in self._durations)):
            return sum(self._durations)

        ValueError("Cannot determine duration statically in dynamic settings")

    @property
    def phase(self):
        if all((isinstance(p, Number) for p in self._phases)):
            return sum(self._phases)

        ValueError("Cannot determine phase statically in dynamic settings")

    def create_package(self):
        sequence = self.sequence_builder.build()
        return QbloxPackage(sequence, self.sequencer_config, self._timeline)

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

    def wait_imm(self, duration: int):
        """
        Waits for `duration` nanoseconds expressed as an immediate.

        `duration` must be positive
        """

        if duration <= 0:
            log.debug(f"Expected positive duration, but got {duration}")
            return

        quotient = duration // Constants.MAX_WAIT_TIME
        remainder = duration % Constants.MAX_WAIT_TIME

        if remainder < Constants.GRID_TIME:
            log.debug(f"Rounding up {remainder} ns to {Constants.GRID_TIME} ns")
            remainder = Constants.GRID_TIME
            duration = quotient * Constants.MAX_WAIT_TIME + remainder

        if quotient >= Constants.LOOP_UNROLL_THRESHOLD:
            with self._loop("wait_label", quotient):
                self.sequence_builder.wait(Constants.MAX_WAIT_TIME)
        elif quotient >= 1:
            for _ in range(quotient):
                self.sequence_builder.wait(Constants.MAX_WAIT_TIME)

        self.sequence_builder.wait(remainder)

    def wait_reg(self, duration: str):
        """
        A mere `wait RX` in general has undefined runtime behaviour.

        This is a useful helper to dynamically wait for a `duration` nanoseconds expressed as a register.
        `iter_reg` is a short-lived register only used as an interator. Customer is responsible for (de)allocation.
        """

        batch_enter = self.alloc_mgr.label_gen("batch_enter")
        batch_exit = self.alloc_mgr.label_gen("batch_exit")
        remainder = self.alloc_mgr.label_gen("remainder")
        round_up = self.alloc_mgr.label_gen("round_up")

        with self.alloc_mgr.reg_borrow("tmp") as iter_reg:
            if duration != iter_reg:
                self.sequence_builder.move(
                    duration,
                    iter_reg,
                    f"Avoids cluttering {duration} as it's likely used as an accumulator",
                )
                self.sequence_builder.nop()

            self.sequence_builder.jlt(iter_reg, Constants.GRID_TIME, round_up)
            self.sequence_builder.jlt(iter_reg, Constants.MAX_WAIT_TIME, remainder)
            self.sequence_builder.label(batch_enter)
            self.sequence_builder.wait(Constants.MAX_WAIT_TIME)
            self.sequence_builder.sub(iter_reg, Constants.MAX_WAIT_TIME, iter_reg)
            self.sequence_builder.nop()
            self.sequence_builder.jge(iter_reg, Constants.MAX_WAIT_TIME, batch_enter)
            self.sequence_builder.jge(iter_reg, Constants.GRID_TIME, remainder)
            self.sequence_builder.label(round_up)
            self.sequence_builder.jlt(iter_reg, 1, batch_exit)
            self.sequence_builder.move(Constants.GRID_TIME, iter_reg, "Rounding up")
            self.sequence_builder.nop()
            self.sequence_builder.label(remainder)
            self.sequence_builder.wait(iter_reg)
            self.sequence_builder.label(batch_exit)

    def ledger(self, duration: Union[int, str], pulse: np.ndarray = None):
        if isinstance(duration, int):
            pulse = np.zeros(duration, dtype=complex) if pulse is None else pulse
            if all((isinstance(p, Number) for p in self._phases)):
                total_phase = sum(self._phases)
                pulse = pulse * np.exp(1.0j * total_phase)
            assert pulse.size == duration
            self._durations.append(duration)
            self._timeline = np.append(self._timeline, pulse)
        elif isinstance(duration, str):
            self._durations.append(duration)
            self._timeline = None  # Destroy the timeline as we are in dynamic setting
        else:
            raise ValueError(f"Expected legal immediate or register, but got {duration}")

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

    def _upd_param_imm(self, duration: int):
        self.sequence_builder.upd_param(Constants.GRID_TIME)
        self.wait_imm(duration - Constants.GRID_TIME)

    def _upd_param_reg(self, duration: str):
        """
        Update latched parameters and wait for `duration` nanoseconds expressed as a register

        If `duration` is less than the threshold Constants.GRID_TIME, the generates assembly rounds up
        and waits for Constants.GRID_TIME.

        Note that the `up_param` instruction is defined only for an immediate operand which must be
        at least Constants.GRID_TIME. Therefore, technically we're already forced to round up.
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
                "Guards against risky underflow",
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

    def _evaluate_square_pulse(self, pulse: Pulse, target: PulseChannel):
        bias = target.bias
        scale = 1.0 + 0.0j if pulse.ignore_channel_scale else target.scale

        pulse_amp = pulse.scale_factor * pulse.amp
        pulse_amp = scale * pulse_amp + bias

        if np.abs(pulse_amp.real) > 1 or np.abs(pulse_amp.imag) > 1:
            raise ValueError(
                f"""
                Voltage range for I or Q exceeded. Current values:
                waveform.scale_factor: {pulse.scale_factor}
                waveform.amp: {pulse.amp}
                (Channel) scale: {scale}
                (Channel) bias: {bias}
                """
            )

        return pulse_amp

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

    def _register_acquisition(self, acq_name: str, num_bins: Union[int, str]):
        acq_index = self._acq_index
        self.sequence_builder.add_acquisition(acq_name, acq_index, num_bins)
        self._acq_index = acq_index + 1
        return acq_index

    def _register_weight(self, name: str, weight: np.ndarray):
        wgt_index = self._wgt_index
        self.sequence_builder.add_weight(name, wgt_index, weight.tolist())
        self._wgt_index = wgt_index + 1
        return wgt_index

    def _do_acquire(self, wgt_name, acq_filter, acq_index, acq_bin, acq_width):
        legal_acq_width = min(acq_width, Constants.MAX_WAIT_TIME)
        if acq_filter:
            weight = acq_filter.samples
            wgt_i_index = self._register_weight(f"{wgt_name}_weight_I", weight.real)
            wgt_q_index = self._register_weight(f"{wgt_name}_weight_Q", weight.imag)

            # TODO - silly convoluted way of using the `acquire_weighed` instruction
            # TODO - simplify when Qblox supports more freedom of memory addressing
            with (
                self.alloc_mgr.reg_borrow("wgt_i") as wgt_i_reg,
                self.alloc_mgr.reg_borrow("wgt_q") as wgt_q_reg,
            ):
                self.sequence_builder.move(wgt_i_index, wgt_i_reg)
                self.sequence_builder.move(wgt_q_index, wgt_q_reg)

                self.sequence_builder.acquire_weighed(
                    acq_index, acq_bin, wgt_i_reg, wgt_q_reg, legal_acq_width
                )
        else:
            self.sequence_builder.acquire(acq_index, acq_bin, legal_acq_width)

        self.wait_imm(acq_width - legal_acq_width)

    @contextmanager
    def _wrapper_pulse(
        self,
        delay_width,
        pulse_width,
        pulse_shape,
        i_steps,
        q_steps,
        i_index,
        q_index,
    ):
        effective_width = max(min(pulse_width, delay_width), Constants.GRID_TIME)
        if pulse_shape == PulseShapeType.SQUARE:
            self.sequence_builder.set_awg_offs(i_steps, q_steps)
            self.sequence_builder.upd_param(effective_width)
        else:
            self.sequence_builder.play(i_index, q_index, effective_width)
        remaining_width = pulse_width - effective_width
        if remaining_width < Constants.GRID_TIME:
            raise ValueError(
                f"""
                Remaining width after delay must be at least {Constants.GRID_TIME} ns.
                Got pulse width = {pulse_width} ns, delay width = {delay_width} ns. This leaves {remaining_width} ns,
                which isn't enough to trigger acquisition afterwards. Adjust the pulse width or delay width values so
                that the following holds:
                pulse_width - max(min(pulse_width, delay_width), Constants.GRID_TIME) >= Constants.GRID_TIME
                """
            )
        yield remaining_width
        if pulse_shape == PulseShapeType.SQUARE:
            self.sequence_builder.set_awg_offs(0, 0)
            self.sequence_builder.upd_param(Constants.GRID_TIME)

            self.ledger(Constants.GRID_TIME)

    @contextmanager
    def _wrapper_cond(self, mask, operator, duration):
        """
        A wrapper for conditional regions
        """

        self.sequence_builder.set_cond(
            1, mask, operator, duration, "Start of conditional region"
        )
        yield
        self.sequence_builder.set_cond(
            0, mask, operator, duration, "End of conditional region"
        )

    def id(self):
        self.sequence_builder.nop()

    def delay(self, inst: Delay):
        if isinstance(inst.duration, Variable):
            duration = self.alloc_mgr.registers[inst.duration.name]
            self.wait_reg(duration)
        elif isinstance(inst.duration, Number):
            duration = int(calculate_duration(inst))
            self.wait_imm(duration)
        else:
            raise ValueError(f"Expected a Variable or a Number but got {inst.duration}")

        self.ledger(duration)

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

        self.ledger(Constants.GRID_TIME)

    def shift_phase(self, inst: PhaseShift):
        if isinstance(inst.phase, Variable):
            ph_reg = self.alloc_mgr.registers[inst.phase.name]
            with self._modulo_reg(ph_reg, Constants.NCO_MAX_PHASE_STEPS) as iter_reg:
                self.sequence_builder.set_ph_delta(iter_reg)
            self.sequence_builder.upd_param(Constants.GRID_TIME)
        elif isinstance(inst.phase, Number):
            ph_imm = QbloxLegalisationPass.phase_as_steps(inst.phase)
            self.sequence_builder.set_ph_delta(ph_imm)
            self.sequence_builder.upd_param(Constants.GRID_TIME)
        else:
            raise ValueError(f"Expected a Variable or a Number but got {inst.phase}")

        self._phases.append(inst.phase)
        self.ledger(Constants.GRID_TIME)

    def reset_phase(self):
        self.sequence_builder.reset_ph()
        self.sequence_builder.upd_param(Constants.GRID_TIME)

        self._phases.clear()
        self.ledger(Constants.GRID_TIME)

    def waveform(self, waveform: Waveform, target: PulseChannel):
        attr2var = {
            attr: var for attr, var in vars(waveform).items() if isinstance(var, Variable)
        }

        if isinstance(waveform, Pulse) and waveform.shape == PulseShapeType.SQUARE:
            if "amp" in attr2var:
                pulse_amp = attr2var["amp"]
                i_steps = self.alloc_mgr.registers[pulse_amp.name]
                q_steps = self.alloc_mgr.registers["zero"]
            else:
                pulse_amp = self._evaluate_square_pulse(waveform, target)
                i_steps = int(pulse_amp.real * Constants.MAX_OFFSET)
                q_steps = int(pulse_amp.imag * Constants.MAX_OFFSET)

            self.sequence_builder.set_awg_offs(i_steps, q_steps)

            if "width" in attr2var:
                pulse_width = attr2var["width"]
                pulse_width = self.alloc_mgr.registers[pulse_width.name]
                self._upd_param_reg(pulse_width)
            else:
                pulse_width = int(calculate_duration(waveform))
                if pulse_width < Constants.GRID_TIME:
                    log.debug(
                        f"""
                        Minimum pulse width is {Constants.GRID_TIME} ns with a resolution of {1} ns.
                        Please round up the width to at least {Constants.GRID_TIME} nanoseconds.
                        This pulse will be ignored.
                        """
                    )
                    return
                self._upd_param_imm(pulse_width)

            self.sequence_builder.set_awg_offs(0, 0)
            self.sequence_builder.upd_param(Constants.GRID_TIME)

            self.ledger(Constants.GRID_TIME)

            pulse = (
                np.ones(pulse_width, dtype=complex) * pulse_amp
                if isinstance(pulse_width, Number) and isinstance(pulse_amp, Number)
                else None
            )
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
            self.wait_imm(pulse_width - max_width)

        self.ledger(pulse_width, pulse)

    @staticmethod
    def synchronize(inst: Synchronize, contexts: Dict):
        """
        Favours static time padding whenever possible, or else uses SYNC.
        TODO - Default to SYNC when Qblox supports finer grained SYNC groups
        """

        is_static = all(
            all((isinstance(d, Number) for d in contexts[target].durations))
            for target in inst.quantum_targets
        )

        for target in inst.quantum_targets:
            cxt = contexts[target]
            if is_static:
                duration_offset = (
                    max((contexts[target].duration for target in inst.quantum_targets))
                    - cxt.duration
                )
                cxt.wait_imm(duration_offset)
                cxt.ledger(duration_offset)
            else:
                cxt.sequence_builder.wait_sync(Constants.GRID_TIME)
                cxt.ledger(Constants.GRID_TIME)

    def device_update(self, du_inst: DeviceUpdate):
        if isinstance(du_inst.value, Variable):
            val_reg = self.alloc_mgr.registers[du_inst.value.name]
            if du_inst.attribute == "frequency":
                self.sequence_builder.set_freq(val_reg)
                self.sequence_builder.upd_param(Constants.GRID_TIME)

                self.ledger(Constants.GRID_TIME)
            elif du_inst.attribute == "scale":
                pass
            else:
                raise NotImplementedError(
                    f"Unsupported processing of attribute {du_inst.attribute} for instruction {du_inst}"
                )
        elif isinstance(du_inst.value, Number):
            if du_inst.attribute == "frequency":
                lo_freq, nco_freq = TILegalisationPass.decompose_freq(
                    du_inst.value, du_inst.target
                )
                freq_imm = QbloxLegalisationPass.freq_as_steps(nco_freq)
                self.sequence_builder.set_freq(freq_imm)
                self.sequence_builder.upd_param(Constants.GRID_TIME)

                self.ledger(Constants.GRID_TIME)
            elif du_inst.attribute == "scale":
                pass
            else:
                raise NotImplementedError(
                    f"Unsupported processing of attribute {du_inst.attribute} for instruction {du_inst}"
                )
        else:
            raise ValueError(f"Expected a Variable or a Number but got {du_inst.value}")


class QbloxContext1(AbstractContext):
    def __init__(self, repeat: Repeat):
        super().__init__()

        self._repeat_count = repeat.repeat_count
        self._repeat_period = repeat.passive_reset_time
        self._repeat_reg = None
        self._repeat_label = None

    def __enter__(self):
        self.clear()
        self._repeat_reg = self.alloc_mgr.reg_alloc("shot")
        self._repeat_label = self.alloc_mgr.label_gen("shot")

        self.sequence_builder.set_mrk(3)
        self.sequence_builder.set_latch_en(1, 4)
        self.sequence_builder.upd_param(Constants.GRID_TIME)

        self.sequence_builder.move(0, self._repeat_reg, "Shot / Repeat iteration")

        self.sequence_builder.label(self._repeat_label)
        self.sequence_builder.wait_sync(
            Constants.GRID_TIME, "Sync at the beginning of shot"
        )
        self.reset_phase()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.wait_imm(int(self._repeat_period * 1e9))
        self.sequence_builder.add(
            self._repeat_reg, 1, self._repeat_reg, "Increment Shot / Repeat iterator"
        )
        self.sequence_builder.nop()
        self.sequence_builder.jlt(self._repeat_reg, self._repeat_count, self._repeat_label)
        self.sequence_builder.stop()

        self.alloc_mgr.reg_free(self._repeat_reg)
        self._repeat_label = None
        self._repeat_reg = None

    def measure_acquire(
        self,
        measure: MeasurePulse,
        acquire: Acquire,
        post_procs: List[PostProcessing],
        target: PulseChannel,
    ):
        pulse = self._evaluate_waveform(measure, target)
        pulse_width = pulse.size
        delay_width = int(calculate_duration(Delay(acquire.channel, acquire.delay)))

        if pulse_width < Constants.GRID_TIME:
            log.debug(
                f"""
                Minimum pulse width is {Constants.GRID_TIME} ns with a resolution of {1} ns.
                Please round up the width to at least {Constants.GRID_TIME} nanoseconds.
                This pulse will be ignored. 
                """
            )
            return

        if pulse_width < delay_width + Constants.GRID_TIME:
            raise ValueError(
                f"""
                Expected pulse width >= delay width + {Constants.GRID_TIME} ns.
                Got pulse width = {pulse_width} ns and delay width = {delay_width} ns.
                """
            )

        requires_shot_avg = any(
            (
                pp.process == PostProcessType.MEAN and ProcessAxis.SEQUENCE in pp.axes
                for pp in post_procs
            )
        )
        num_bins = 1 if requires_shot_avg else self._repeat_count
        acq_bin = 0 if requires_shot_avg else self._repeat_reg
        acq_width = int(calculate_duration(acquire))
        acq_index = self._register_acquisition(acquire.output_variable, num_bins)
        self.sequencer_config.square_weight_acq.integration_length = acq_width
        self.sequencer_config.thresholded_acq.rotation = np.rad2deg(acquire.rotation)
        self.sequencer_config.thresholded_acq.threshold = acquire.threshold

        i_steps, q_steps = None, None
        i_index, q_index = None, None
        if measure.shape == PulseShapeType.SQUARE:
            i_steps = int(pulse[0].real * Constants.MAX_OFFSET)
            q_steps = int(pulse[0].imag * Constants.MAX_OFFSET)
        else:
            i_index, q_index = self._register_waveform(measure, target, pulse)

        with self._wrapper_pulse(
            delay_width, pulse_width, measure.shape, i_steps, q_steps, i_index, q_index
        ) as remaining_width:
            self._do_acquire(
                acquire.output_variable,
                acquire.filter,
                acq_index,
                acq_bin,
                remaining_width,
            )

        self.ledger(pulse_width, pulse)


class QbloxContext2(AbstractContext):
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

    def __enter__(self):
        """
        Serves as the prologue
        """

        self.sequence_builder.set_mrk(3)
        self.sequence_builder.set_latch_en(1, 4)
        self.sequence_builder.upd_param(Constants.GRID_TIME)

        for name, register in self.alloc_mgr.registers.items():
            self.sequence_builder.move(
                0,
                register,
                f"Precautionary initialisation for variable {name}",
            )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Serves as the epilogue
        """

        self.sequence_builder.stop()

    def measure_acquire(
        self, measure: MeasurePulse, acquire: Acquire, target: PulseChannel
    ):
        pulse = self._evaluate_waveform(measure, target)
        pulse_width = pulse.size
        delay_width = int(calculate_duration(Delay(acquire.channel, acquire.delay)))

        if pulse_width < Constants.GRID_TIME:
            log.debug(
                f"""
                Minimum pulse width is {Constants.GRID_TIME} ns with a resolution of {1} ns.
                Please round up the width to at least {Constants.GRID_TIME} nanoseconds.
                This pulse will be ignored. 
                """
            )
            return

        if pulse_width < delay_width + Constants.GRID_TIME:
            raise ValueError(
                f"""
                Expected pulse width >= delay width + {Constants.GRID_TIME} ns.
                Got pulse width = {pulse_width} ns and delay width = {delay_width} ns.
                """
            )

        name = f"acquire_{hash(acquire)}"
        num_bins = self.iter_bounds[name].count
        acq_bin = self.alloc_mgr.registers[name]
        acq_width = int(calculate_duration(acquire))
        acq_index = self._register_acquisition(acquire.output_variable, num_bins)
        self.sequencer_config.square_weight_acq.integration_length = acq_width
        self.sequencer_config.thresholded_acq.rotation = acquire.rotation
        self.sequencer_config.thresholded_acq.threshold = acquire.threshold

        i_steps, q_steps = None, None
        i_index, q_index = None, None
        if measure.shape == PulseShapeType.SQUARE:
            i_steps = int(pulse[0].real * Constants.MAX_OFFSET)
            q_steps = int(pulse[0].imag * Constants.MAX_OFFSET)
        else:
            i_index, q_index = self._register_waveform(measure, target, pulse)

        with self._wrapper_pulse(
            delay_width, pulse_width, measure.shape, i_steps, q_steps, i_index, q_index
        ) as remaining_width:
            self._do_acquire(
                acquire.output_variable,
                acquire.filter,
                acq_index,
                acq_bin,
                remaining_width,
            )

        self.ledger(pulse_width, pulse)

    @staticmethod
    def enter_repeat(inst: Repeat, contexts: Dict):
        iter_name = f"repeat_{hash(inst)}"
        for context in contexts.values():
            register = context.alloc_mgr.registers[iter_name]
            bound = context.iter_bounds[iter_name]
            context.sequence_builder.move(bound.start, register, "Shot / Repeat iteration")

            label = context.alloc_mgr.labels[iter_name]
            context.sequence_builder.label(label)
            context.sequence_builder.wait_sync(
                Constants.GRID_TIME, "Sync at the beginning of shot"
            )
            context.reset_phase()

    @staticmethod
    def exit_repeat(inst: Repeat, contexts: Dict):
        iter_name = f"repeat_{hash(inst)}"
        for context in contexts.values():
            register = context.alloc_mgr.registers[iter_name]
            label = context.alloc_mgr.labels[iter_name]
            bound = context.iter_bounds[iter_name]

            context.wait_imm(int(inst.passive_reset_time * 1e9))
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


class QbloxBackend1(BaseBackend, InvokerMixin):
    def __init__(self, model: QbloxLiveHardwareModel, ignore_empty=True):
        super().__init__(model)
        self.ignore_empty = ignore_empty

    def build_pass_pipeline(self, *args, **kwargs):
        pass

    def emit(
        self,
        ir: InstructionBuilder,
        res_mgr: Optional[ResultManager] = None,
        met_mgr: Optional[MetricsManager] = None,
        compiler_config: Optional[CompilerConfig] = None,
    ) -> Dict[int, QbloxExecutable]:
        triage_result: TriageResult = res_mgr.lookup_by_type(TriageResult)
        sweeps = triage_result.sweeps

        switerator = SweepIterator.from_sweeps(sweeps)
        dinjectors = DeviceInjectors(triage_result.device_updates)
        ordered_executables: Dict[int, QbloxExecutable] = OrderedDict()
        try:
            dinjectors.inject()
            while not switerator.is_finished():
                switerator.do_sweep(triage_result.quantum_instructions)
                ordered_executables[switerator.accumulated_sweep_iteration] = self._do_emit(
                    triage_result
                )
            return ordered_executables
        except BaseException as e:
            raise e
        finally:
            switerator.revert(triage_result.quantum_instructions)
            dinjectors.revert()

    def _do_emit(self, triage_result: TriageResult) -> QbloxExecutable:
        repeat = next(iter(triage_result.repeats))
        contexts: Dict[PulseChannel, QbloxContext1] = {}

        with ExitStack() as stack:
            inst_iter = iter(triage_result.quantum_instructions)
            while (inst := next(inst_iter, None)) is not None:
                if isinstance(inst, PostProcessing):
                    continue  # Ignore postprocessing

                if isinstance(inst, Synchronize):
                    for target in inst.quantum_targets:
                        if not isinstance(target, PulseChannel):
                            raise ValueError(f"{target} is not a PulseChannel")
                        contexts.setdefault(
                            target, stack.enter_context(QbloxContext1(repeat))
                        )
                    QbloxContext1.synchronize(inst, contexts)
                    continue

                for target in inst.quantum_targets:
                    if not isinstance(target, PulseChannel):
                        raise ValueError(f"{target} is not a PulseChannel")

                    context = contexts.setdefault(
                        target, stack.enter_context(QbloxContext1(repeat))
                    )

                    if isinstance(inst, DeviceUpdate):
                        context.device_update(inst)
                    elif isinstance(inst, PhaseReset):
                        context.reset_phase()
                    elif isinstance(inst, MeasurePulse):
                        acquire = next(inst_iter, None)
                        if acquire is None or not isinstance(acquire, Acquire):
                            raise ValueError(
                                "Found a MeasurePulse but no Acquire instruction followed"
                            )
                        post_procs = triage_result.pp_map[acquire.output_variable]
                        context.measure_acquire(inst, acquire, post_procs, target)
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

        packages = {
            target.full_id(): context.create_package()
            for target, context in contexts.items()
            if not (self.ignore_empty and context.is_empty())
        }

        return QbloxExecutable(packages=packages, triage_result=triage_result)


class QbloxBackend2(BaseBackend, InvokerMixin):
    def __init__(self, model: QbloxLiveHardwareModel, ignore_empty=True):
        super().__init__(model)
        self.ignore_empty = ignore_empty

    def build_pass_pipeline(self, *args, **kwargs):
        return PassManager() | PreCodegenPass() | CFGPass()

    def emit(
        self,
        ir: InstructionBuilder,
        res_mgr: Optional[ResultManager] = None,
        met_mgr: Optional[MetricsManager] = None,
        compiler_config: Optional[CompilerConfig] = None,
    ) -> QbloxExecutable:
        self.run_pass_pipeline(ir, res_mgr, met_mgr)

        triage_result: TriageResult = res_mgr.lookup_by_type(TriageResult)
        binding_result: BindingResult = res_mgr.lookup_by_type(BindingResult)
        precodegen_result: PreCodegenResult = res_mgr.lookup_by_type(PreCodegenResult)
        cfg_result: CFGResult = res_mgr.lookup_by_type(CFGResult)

        scoping_results: Dict[PulseChannel, ScopingResult] = binding_result.scoping_results
        rw_results: Dict[PulseChannel, ReadWriteResult] = binding_result.rw_results
        iter_bound_results: Dict[PulseChannel, Dict[str, IterBound]] = (
            binding_result.iter_bound_results
        )
        alloc_mgrs: Dict[PulseChannel, AllocationManager] = precodegen_result.alloc_mgrs

        with ExitStack() as stack:
            contexts = {
                t: stack.enter_context(
                    QbloxContext2(
                        alloc_mgr=alloc_mgrs[t],
                        scoping_result=scoping_results[t],
                        rw_result=rw_results[t],
                        iter_bounds=iter_bound_results[t],
                    )
                )
                for t in triage_result.target_map
            }
            QbloxCFGWalker(contexts).run(cfg_result.cfg)

        packages = {
            target.full_id(): context.create_package()
            for target, context in contexts.items()
            if not (self.ignore_empty and context.is_empty())
        }

        return QbloxExecutable(packages=packages, triage_result=triage_result)


class QbloxCFGWalker(DfsTraversal):
    def __init__(self, contexts: Dict[PulseChannel, QbloxContext2]):
        super().__init__()
        self.contexts = contexts

    def enter(self, block):
        iterator = block.iterator()
        while (inst := next(iterator, None)) is not None:
            if isinstance(inst, Sweep):
                QbloxContext2.enter_sweep(inst, self.contexts)
            elif isinstance(inst, Repeat):
                QbloxContext2.enter_repeat(inst, self.contexts)
            elif isinstance(inst, QuantumInstruction):
                if isinstance(inst, PostProcessing):
                    continue
                elif isinstance(inst, Synchronize):
                    QbloxContext2.synchronize(inst, self.contexts)
                    continue

                for target in inst.quantum_targets:
                    context = self.contexts[target]
                    if isinstance(inst, DeviceUpdate):
                        context.device_update(inst)
                    elif isinstance(inst, PhaseReset):
                        context.reset_phase()
                    elif isinstance(inst, MeasurePulse):
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
                QbloxContext2.exit_repeat(inst, self.contexts)
            elif isinstance(inst, Sweep):
                QbloxContext2.exit_sweep(inst, self.contexts)
