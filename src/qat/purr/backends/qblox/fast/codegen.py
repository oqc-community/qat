from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

import numpy as np

from qat.ir.pass_base import AnalysisPass, InvokerMixin, PassManager, PassResultSet
from qat.purr.backends.analysis_passes import CFGPass, VariableBoundsPass
from qat.purr.backends.codegen import CodegenResultType, DfsTraversal
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
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import PulseChannel
from qat.purr.compiler.instructions import (
    Acquire,
    CustomPulse,
    Delay,
    DeviceUpdate,
    FrequencyShift,
    Id,
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


class AttributeType(Enum):
    BOUNDS = "bounds"
    LABEL = "label"
    NUM_BINS = "num_bins"
    REGISTER = "register"


@dataclass
class Attribute:
    name: str = None
    value: Any = None


class AttributeManager:
    def __init__(self):
        self.attributes = defaultdict(list)

    def get_attribute(self, key, name):
        return next((attr.value for attr in self.attributes[key] if attr.name == name))

    def set_attribute(self, key, name: str, value: Any):
        self.attributes[key].append(Attribute(name, value))


class FastQbloxContext:
    def __init__(self):
        self.builder = SequenceBuilder()
        self.reg_mgr = RegisterManager()
        self.lbl_mgr = LabelManager()
        self.attr_mgr = AttributeManager()
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
        quotient = duration_nanos // Constants.MAX_WAIT_TIME
        remainder = duration_nanos % Constants.MAX_WAIT_TIME
        if quotient > Constants.LOOP_UNROLL_THRESHOLD:
            with self._loop("wait_label", quotient):
                self.builder.wait(Constants.MAX_WAIT_TIME)
        elif quotient >= 1:
            for _ in range(quotient):
                self.builder.wait(Constants.MAX_WAIT_TIME)

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
        num_bins = self.attr_mgr.get_attribute(acquire, "num_bins")
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
        bin_reg = self.attr_mgr.get_attribute(acquire, "register")
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
            value = self.attr_mgr.get_attribute(value.name, "register")

        if du_inst.attribute == "frequency":
            self.builder.set_freq(value)
        else:
            raise NotImplementedError(
                f"Unsupported processing of attribute {du_inst.attribute}"
            )

    @staticmethod
    def enter_repeat(inst: Repeat, contexts: Dict):
        for context in contexts.values():
            register = context.attr_mgr.get_attribute(inst, "register")
            label = context.attr_mgr.get_attribute(inst, "label")

            context.builder.move(0, register)
            context.builder.nop()
            context.builder.label(label)
            context.builder.reset_ph()
            context.builder.upd_param(Constants.GRID_TIME)

    @staticmethod
    def exit_repeat(inst: Repeat, contexts: Dict):
        for context in contexts.values():
            register = context.attr_mgr.get_attribute(inst, "register")
            label = context.attr_mgr.get_attribute(inst, "label")

            context._wait(inst.repetition_period)
            context.builder.add(register, 1, register)
            context.builder.nop()
            context.builder.jlt(register, inst.repeat_count, label)

    @staticmethod
    def enter_sweep(inst: Sweep, contexts: Dict):
        name = next(iter(inst.variables.keys()))
        for context in contexts.values():
            register = context.attr_mgr.get_attribute(name, "register")
            label = context.attr_mgr.get_attribute(name, "label")
            start, _, _, _ = context.attr_mgr.get_attribute(name, "bounds")

            context.builder.move(start, register)
            context.builder.nop()
            context.builder.label(label)

    @staticmethod
    def exit_sweep(inst: Sweep, contexts: Dict):
        name = next(iter(inst.variables.keys()))
        for context in contexts.values():
            register = context.attr_mgr.get_attribute(name, "register")
            label = context.attr_mgr.get_attribute(name, "label")
            start, step, end, count = context.attr_mgr.get_attribute(name, "bounds")

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
            context.builder.set_mrk(3)

    @staticmethod
    def epilogue(contexts: Dict):
        for context in contexts.values():
            context.builder.stop()

    def create_package(self, target: PulseChannel):
        return QbloxPackage(target, self.builder.build(), self.config)


class PreCodegenPass(AnalysisPass):
    """
    Precedes assembly codegen.
    Performs a naive register allocation through a manager object.
    Computes useful information in the form of attributes.
    """

    def run(self, builder: InstructionBuilder, *args, **kwargs):
        analyses: PassResultSet = args[0]
        target_view = analyses.get_result(CodegenResultType.TARGET_VIEW)
        var_bounds = analyses.get_result(CodegenResultType.VARIABLE_BOUNDS)
        contexts = {t: FastQbloxContext() for t in target_view}

        for context in contexts.values():
            context.builder.set_mrk(3)

        for i, inst in enumerate(builder.instructions):
            if isinstance(inst, Sweep):
                name, value = next(iter(inst.variables.items()))
                for t, context in contexts.items():
                    register = context.reg_mgr.allocate()
                    context.attr_mgr.set_attribute(name, "register", register)
                    context.builder.move(0, register)
                    context.builder.nop()

                    label = context.lbl_mgr.generate(name)
                    context.attr_mgr.set_attribute(name, "label", label)

                    context.attr_mgr.set_attribute(name, "bounds", var_bounds[t][name])
                    context.attr_mgr.set_attribute(inst, "bounds", var_bounds[t][name])
            elif isinstance(inst, Repeat):
                for context in contexts.values():
                    register = context.reg_mgr.allocate()
                    context.attr_mgr.set_attribute(inst, "register", register)
                    context.builder.move(0, register)
                    context.builder.nop()

                    label = context.lbl_mgr.generate("repeat")
                    context.attr_mgr.set_attribute(inst, "label", label)
            elif isinstance(inst, Acquire):
                for target in inst.quantum_targets:
                    context = contexts[target]
                    register = context.reg_mgr.allocate()
                    context.builder.move(0, register)
                    context.builder.nop()
                    context.attr_mgr.set_attribute(inst, "register", register)

        analyses.update(
            PassResultSet((hash(builder), self.id(), CodegenResultType.CONTEXTS, contexts))
        )
        return analyses


class FastQbloxEmitter(InvokerMixin):
    def __init__(self, analyses: PassResultSet = None):
        self.analyses = analyses or PassResultSet()

    def build_pass_pipeline(self, *args, **kwargs):
        pipeline = PassManager()
        pipeline.add(VariableBoundsPass())
        pipeline.add(PreCodegenPass())
        pipeline.add(CFGPass())
        return pipeline

    def emit_packages(self, builder: InstructionBuilder) -> List[QbloxPackage]:
        self.run_pass_pipeline(builder, self.analyses)
        cfg_walker = QbloxCFGWalker(self.analyses)
        cfg_walker.walk()
        return cfg_walker.create_packages()


class QbloxCFGWalker(DfsTraversal):
    def __init__(self, analyses: PassResultSet):
        super().__init__()
        self.cfg = analyses.get_result(CodegenResultType.CFG)
        self.contexts: Dict[PulseChannel, FastQbloxContext] = analyses.get_result(
            CodegenResultType.CONTEXTS
        )

    def enter(self, block):
        iterator = block.iterator()
        while (inst := next(iterator, None)) is not None:
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
                        next_inst = next(iterator, None)
                        if next_inst is None or not isinstance(next_inst, Acquire):
                            raise ValueError(
                                "Found a MeasurePulse but no Acquire instruction followed"
                            )

                        # TODO - this could move to a dedicated pass on the cfg (batching or ctrl-hw related)
                        num_bins = 1
                        for block in self._entered:
                            head = block.head()
                            if isinstance(head, Sweep):
                                name = next(iter(head.variables.keys()))
                                bounds = context.attr_mgr.get_attribute(name, "bounds")
                                num_bins *= bounds[-1]
                            elif isinstance(head, Repeat):
                                num_bins *= head.repeat_count
                        context.attr_mgr.set_attribute(next_inst, "num_bins", num_bins)
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
                FastQbloxContext.exit_repeat(inst, self.contexts)
            elif isinstance(inst, Sweep):
                FastQbloxContext.exit_sweep(inst, self.contexts)

    def walk(self):
        self.run(self.cfg)

        # TODO - run as visit to the entry/exit block
        FastQbloxContext.epilogue(self.contexts)

        # Remove empty contexts
        self.contexts = {t: c for t, c in self.contexts.items() if not c.is_empty()}

        # Remove Opcode.WAIT_SYNC instructions when the experiment contains only a singleton context
        if len(self.contexts) == 1:
            context = next(iter(self.contexts.values()))
            context.builder.optimize()

    def create_packages(self):
        return [context.create_package(target) for target, context in self.contexts.items()]
