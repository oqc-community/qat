from collections import defaultdict
from contextlib import contextmanager
from functools import reduce
from operator import mul
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from qat.purr.backends.qblox.codegen import (
    QbloxPackage,
    calculate_duration,
    get_nco_phase_arguments,
    get_nco_set_frequency_arguments,
)
from qat.purr.backends.qblox.config import SequencerConfig
from qat.purr.backends.qblox.constants import Constants
from qat.purr.backends.qblox.ir import SequenceBuilder
from qat.purr.backends.utilities import evaluate_shape
from qat.purr.compiler.analysis import Attribute, extract_iter_bounds
from qat.purr.compiler.control_flow.graph import DfsTraversal, EmitterMixin
from qat.purr.compiler.devices import PulseChannel
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
)
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class RegisterManager:
    def __init__(self):
        self._registers: List[str] = sorted(
            f"R{index}" for index in range(Constants.NUMBER_OF_REGISTERS)
        )

    def allocate(self) -> str:
        if len(self._registers) < 1:
            raise IndexError(
                "Out of registers. Attempting to use more registers "
                "than available in the Q1 sequence processor"
            )
        return self._registers.pop(0)

    def free(self, register: str) -> None:
        if register in self._registers:
            raise RuntimeError(f"Cannot free register '{register}' as it's not in use")
        self._registers.append(register)


class LabelManager:
    def __init__(self):
        self._counters: Dict[str, int] = {}

    def generate(self, prefix: str):
        counter = self._counters.setdefault(prefix, 0)
        self._counters[prefix] += 1
        return f"{prefix}_{counter}"


class FastQbloxEmitter(DfsTraversal, EmitterMixin):
    def __init__(self, instructions: List[Instruction]):
        super().__init__()
        super(DfsTraversal, self).__init__(instructions)
        self.contexts: Dict[PulseChannel, FastQbloxContext] = self._init_contexts()

    def _init_contexts(self) -> Dict:
        targets: Set[PulseChannel] = set()

        for inst in self.instructions:
            if isinstance(inst, QuantumInstruction):
                if isinstance(inst, PostProcessing):
                    for qt in inst.quantum_targets:
                        if isinstance(qt, Acquire):
                            targets.update(qt.quantum_targets)
                        else:
                            targets.add(qt)
                else:
                    targets.update(inst.quantum_targets)

        for t in targets:
            if not isinstance(t, PulseChannel):
                raise ValueError(f"{t} is not a PulseChannel")

        contexts = {t: FastQbloxContext() for t in targets}

        for i, inst in enumerate(self.instructions):
            if isinstance(inst, Sweep):
                name, value = next(iter(inst.variables.items()))
                for context in contexts.values():
                    context.allocate_register(inst)
                    context.allocate_label(inst, name)

                inst_iter = iter(self.instructions[i + 1 :])
                targets.clear()
                while (du_inst := next(inst_iter, None)) is not None:
                    if not isinstance(du_inst, DeviceUpdate):
                        break
                    if not isinstance(du_inst.target, PulseChannel):
                        raise NotImplementedError(
                            f"Cannot analyse non PulseChannel {du_inst.target}"
                        )
                    if du_inst.attribute != "frequency":
                        raise NotImplementedError(
                            f"Unsupported processing of attribute {du_inst.attribute}"
                        )
                    if du_inst.target.fixed_if:
                        log.warning(
                            f"Ignoring Fixed IF constraint on target {du_inst.target}"
                        )
                    du_value = [x - du_inst.target.baseband_frequency for x in value]
                    start, step, end, count = extract_iter_bounds(du_value)
                    du_bounds = tuple(
                        [
                            get_nco_set_frequency_arguments(int(x))
                            for x in [start, step, end]
                        ]
                        + [count]
                    )
                    context = contexts[du_inst.target]
                    context.set_iter_bounds(inst, du_bounds)
                    register = context.get_attribute(inst, "register")
                    targets.add(du_inst.target)
                    if isinstance(du_inst.value, Variable):
                        context.set_attribute(du_inst, du_inst.attribute, register)
                    else:
                        context.set_attribute(du_inst, du_inst.attribute, du_inst.value)

                diff = {t: c for t, c in contexts.items() if t not in targets}
                if diff:
                    start, step, end, count = extract_iter_bounds(value)
                    bounds = (int(start), int(step), int(end), int(count))
                    if start < 0 or end > Constants.MAX_REGISTER_VALUE:
                        log.warning(f"Cannot lower sweep iteration bounds {bounds}")
                        bounds = (0, 1, count, count)
                        log.warning(f"Attempting count-based bounds {bounds}")
                    for target, context in diff.items():
                        log.warning(f"Using bounds {bounds} for target {target}")
                        context.set_iter_bounds(inst, bounds)
            elif isinstance(inst, Repeat):
                bounds = (0, 1, inst.repeat_count, inst.repeat_count)
                for context in contexts.values():
                    context.allocate_register(inst)
                    context.allocate_label(inst, "repeat")
                    context.set_iter_bounds(inst, bounds)
            elif isinstance(inst, Acquire):
                for target in inst.quantum_targets:
                    contexts[target].allocate_register(inst)
        return contexts

    def enter(self, block):
        iterator = block.iterator()
        while (i := next(iterator, None)) is not None:
            inst = self.instructions[i]
            if isinstance(inst, Repeat):
                FastQbloxContext.enter_repeat(inst, self.contexts)
            elif isinstance(inst, Sweep):
                FastQbloxContext.enter_sweep(inst, self.contexts)
            elif isinstance(inst, QuantumInstruction):
                if isinstance(inst, PostProcessing):
                    continue
                elif isinstance(inst, Synchronize):
                    FastQbloxContext.synchronize(inst, self.contexts)
                    continue
                elif isinstance(inst, PhaseReset):
                    FastQbloxContext.reset_phase(inst, self.contexts)
                    continue

                for target in inst.quantum_targets:
                    context = self.contexts[target]
                    if isinstance(inst, DeviceUpdate):
                        context.device_update(inst)
                    if isinstance(inst, MeasurePulse):
                        j = next(iterator, None)
                        if j is None or not isinstance(
                            acquire := self.instructions[j], Acquire
                        ):
                            raise ValueError(
                                "Found a MeasurePulse but no Acquire instruction followed"
                            )
                        scope_heads = [self.instructions[b.head()] for b in self._entered]
                        counts = [
                            context.get_attribute(h, "iter_bounds")[-1] or 1
                            for h in scope_heads
                        ]
                        context.set_acq_size(acquire, reduce(mul, counts, 1))
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

    def exit(self, block):
        iterator = block.iterator()
        while (i := next(iterator, None)) is not None:
            inst = self.instructions[i]
            if isinstance(inst, Repeat):
                FastQbloxContext.exit_repeat(inst, self.contexts)
            elif isinstance(inst, Sweep):
                FastQbloxContext.exit_sweep(inst, self.contexts)

    def emit_packages(self) -> List[QbloxPackage]:
        cfg = self.emit_cfg()

        FastQbloxContext.prologue(self.contexts)
        self.run(cfg)
        FastQbloxContext.epilogue(self.contexts)

        # Remove empty contexts
        self.contexts = {t: c for t, c in self.contexts.items() if not c.is_empty()}

        # Remove Opcode.WAIT_SYNC instructions when the experiment contains only a singleton context
        if len(self.contexts) == 1:
            context = next(iter(self.contexts.values()))
            context.builder.optimize()

        return [context.create_package(target) for target, context in self.contexts.items()]


class FastQbloxContext:
    def __init__(self):
        self.builder = SequenceBuilder()
        self.reg_mgr = RegisterManager()
        self.lbl_mgr = LabelManager()
        self.attributes: Dict[Instruction, List[Attribute]] = defaultdict(list)
        self.config = SequencerConfig()

        self._duration: int = 0
        self._num_hw_avg = 1  # Technically disabled
        self._wf_memory: int = Constants.MAX_SAMPLE_SIZE_WAVEFORMS
        self._wf_index: int = 0
        self._acq_index: int = 0

        # Keeps record of the frequency shift on the target
        self._frequency: float = 0.0

    @property
    def duration(self):
        return self._duration

    def is_empty(self):
        return not (any(self.builder.waveforms) or any(self.builder.acquisitions))

    def get_attribute(self, inst, name):
        return next((attr.value for attr in self.attributes[inst] if attr.name == name))

    def set_attribute(self, inst, name: str, value: Any):
        self.attributes[inst].append(Attribute(name, value))

    def allocate_register(self, inst: Instruction):
        register = self.reg_mgr.allocate()
        self.set_attribute(inst, "register", register)

    def allocate_label(self, inst: Instruction, prefix: str):
        label = self.lbl_mgr.generate(prefix)
        self.set_attribute(inst, "label", label)

    def set_iter_bounds(self, inst: Instruction, iter_bounds: Tuple):
        self.set_attribute(inst, "iter_bounds", iter_bounds)

    def set_acq_size(self, inst: Instruction, acq_size: int):
        self.set_attribute(inst, "acq_size", acq_size)

    @contextmanager
    def _loop(self, label: str, iter_count: int = 1):
        """
        This context can be used to avoid unrolling loops but needs manual addition of the 0 iteration
        either immediately after the yield or immediately after the context has exited.
        """
        register = self.reg_mgr.allocate()
        label = self.lbl_mgr.generate(label)
        self.builder.move(iter_count, register)
        self.builder.nop()
        self.builder.label(label)
        yield register
        self.builder.loop(register, label)
        self.reg_mgr.free(register)

    def _wait(self, duration: float):
        if duration == 0:
            return
        if duration < 0:
            raise ValueError(f"Wait duration must be positive, got {duration} instead")

        duration_nanos = int(duration * 1e9)
        quotient = duration_nanos // Constants.IMMEDIATE_MAX_WAIT_TIME
        remainder = duration_nanos % Constants.IMMEDIATE_MAX_WAIT_TIME
        if quotient > Constants.LOOP_UNROLL_THRESHOLD:
            with self._loop("wait_label", quotient):
                self.builder.wait(Constants.IMMEDIATE_MAX_WAIT_TIME)
        elif quotient >= 1:
            for _ in range(quotient):
                self.builder.wait(Constants.IMMEDIATE_MAX_WAIT_TIME)

        if remainder > 0:
            self.builder.wait(remainder)

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
        pulse = evaluate_shape(waveform, t)
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

    def _process_acquire(self, acquire: Acquire):
        num_bins = self.get_attribute(acquire, "acq_size")
        acq_index = self._acq_index
        self.builder.add_acquisition(acquire.output_variable, acq_index, num_bins)
        self._acq_index = acq_index + 1
        return acq_index

    def id(self):
        self.builder.nop()

    def delay(self, inst: Delay):
        self._wait(inst.duration)
        self._duration = self._duration + inst.duration

    def waveform(self, waveform: Waveform, target: PulseChannel):
        i_index, q_index, num_samples = self._process_waveform(waveform, target)
        if (i_index, q_index, num_samples) == (None, None, 0):
            return
        self.builder.play(i_index, q_index, Constants.GRID_TIME)
        self.builder.wait(num_samples - Constants.GRID_TIME)
        self._duration = self._duration + waveform.duration

    def measure_acquire(
        self, measure: MeasurePulse, acquire: Acquire, target: PulseChannel
    ):
        i_index, q_index, num_samples = self._process_waveform(measure, target)
        if (i_index, q_index, num_samples) == (None, None, 0):
            raise ValueError(f"Measure pulse must not be empty")

        acq_index = self._process_acquire(acquire)
        bin_reg = self.get_attribute(acquire, "register")
        self.config.square_weight_acq.integration_length = int(acquire.duration * 1e9)

        def _add_binned_acq_instructions():
            self.builder.play(i_index, q_index, Constants.GRID_TIME)
            self._wait(acquire.delay)
            self.builder.acquire(
                acq_index,
                bin_reg,
                Constants.MAX_SAMPLE_SIZE_SCOPE_ACQUISITIONS,
            )
            self.builder.add(bin_reg, 1, bin_reg)
            self.builder.nop()

        if self._num_hw_avg <= Constants.LOOP_UNROLL_THRESHOLD:
            for _ in range(self._num_hw_avg):
                _add_binned_acq_instructions()
        else:
            with self._loop("avg_label", self._num_hw_avg + 1):
                _add_binned_acq_instructions()

        self._duration = self._duration + measure.duration

    def shift_phase(self, inst: PhaseShift):
        value = get_nco_phase_arguments(inst.phase)
        self.builder.set_ph_delta(value)
        self.builder.upd_param(Constants.GRID_TIME)

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

    def device_update(self, du_inst: DeviceUpdate):
        value = du_inst.value
        if isinstance(value, Variable):
            value = self.get_attribute(du_inst, du_inst.attribute)

        if du_inst.attribute == "frequency":
            self.builder.set_freq(value)
        else:
            raise NotImplementedError(
                f"Unsupported processing of attribute {du_inst.attribute}"
            )

    @staticmethod
    def enter_repeat(inst: Repeat, contexts: Dict):
        for context in contexts.values():
            register = context.get_attribute(inst, "register")
            label = context.get_attribute(inst, "label")
            start, _, _, _ = context.get_attribute(inst, "iter_bounds")

            context.builder.move(start, register)
            context.builder.nop()
            context.builder.label(label)
            context.builder.reset_ph()
            context.builder.upd_param(Constants.GRID_TIME)

    @staticmethod
    def exit_repeat(inst: Repeat, contexts: Dict):
        for context in contexts.values():
            register = context.get_attribute(inst, "register")
            label = context.get_attribute(inst, "label")
            _, step, _, count = context.get_attribute(inst, "iter_bounds")

            context._wait(inst.repetition_period)
            context.builder.add(register, step, register)
            context.builder.nop()
            context.builder.jlt(register, count, label)

    @staticmethod
    def enter_sweep(inst: Sweep, contexts: Dict):
        for context in contexts.values():
            register = context.get_attribute(inst, "register")
            label = context.get_attribute(inst, "label")
            start, step, end, count = context.get_attribute(inst, "iter_bounds")

            context.builder.move(start, register)
            context.builder.nop()
            context.builder.label(label)

    @staticmethod
    def exit_sweep(inst: Sweep, contexts: Dict):
        for context in contexts.values():
            register = context.get_attribute(inst, "register")
            label = context.get_attribute(inst, "label")
            start, step, end, count = context.get_attribute(inst, "iter_bounds")

            context.builder.add(register, step, register)
            context.builder.nop()
            context.builder.jlt(register, end, label)

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

    @staticmethod
    def prologue(contexts: Dict):
        for context in contexts.values():
            context.builder.clear()
            context.builder.set_mrk(3)
            for attrs in context.attributes.values():
                for attr in attrs:
                    if attr.name == "register":
                        context.builder.move(0, attr.value)

    @staticmethod
    def epilogue(contexts: Dict):
        for context in contexts.values():
            context.builder.stop()

    def create_package(self, target: PulseChannel):
        return QbloxPackage(target, self.builder.build(), self.config)
