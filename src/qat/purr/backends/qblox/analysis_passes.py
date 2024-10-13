from collections import defaultdict
from typing import Dict, Set

import numpy as np

from qat.backend.analysis_passes import BindingResult, IterBound
from qat.ir.pass_base import AnalysisPass
from qat.ir.result_base import ResultManager
from qat.purr.backends.qblox.constants import Constants
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import PulseChannel, PulseShapeType
from qat.purr.compiler.instructions import DeviceUpdate, Instruction, Pulse, Variable


class QbloxLegalisationPass(AnalysisPass):
    """
    Performs target-dependent legalisation for QBlox.

        A) A repeat instruction with a very high repetition count is illegal because acquisition memory
    on a QBlox sequencer is limited. This requires optimal batching of the repeat instruction into maximally
    supported batches of smaller repeat counts.

    This pass does not do any batching. More features and adjustments will follow in future iterations.

        B) Previously processed variables such as frequencies, phases, and amplitudes still need digital conversion
    to a representation that's required by the QBlox ISA.

    + NCO's 1GHz frequency range by 4e9 steps:
        + [-500, 500] Mhz <=> [-2e9, 2e9] steps
        + 1 Hz            <=> 4 steps
    + NCO's 360° phase range by 1e9 steps:
        + 1e9    steps    <=> 2*pi rad
        + 125e6  steps    <=> pi/4 rad
    + Time and samples are quantified in nanoseconds
    + Amplitude depends on the type of the module:
        + [-1, 1]         <=> [-2.5, 2.5] V (for QCM)
        + [-1, 1]         <=> [-0.5, 0.5] V (for QRM)
    + AWG offset:
        + [-1, 1]         <=> [-32 768, 32 767]

    The last point is interesting as it requires knowledge of physical configuration of qubits and the modules
    they are wired to. This knowledge is typically found during execution and involving it early on would upset
    the rest of the compilation flow. In fact, it complicates this pass in particular, involves allocation
    concepts that should not be treated here, and promotes a monolithic compilation style. A temporary workaround
    is to simply assume the legality of amplitudes from the start whereby users are required to convert
    the desired voltages to the equivalent ratio AOT.

    This pass performs target-dependent conversion as described in part (B). More features and adjustments
    will follow in future iterations.
    """

    @staticmethod
    def phase_as_steps(phase_rad: float) -> int:
        phase_deg = np.rad2deg(phase_rad)
        phase_deg %= 360
        return int(round(phase_deg * Constants.NCO_PHASE_STEPS_PER_DEG))

    @staticmethod
    def freq_as_steps(freq_hz: float) -> int:
        steps = int(round(freq_hz * Constants.NCO_FREQ_STEPS_PER_HZ))

        if (
            steps < -Constants.NCO_FREQ_LIMIT_STEPS
            or steps > Constants.NCO_FREQ_LIMIT_STEPS
        ):
            min_max_frequency_in_hz = (
                Constants.NCO_FREQ_LIMIT_STEPS / Constants.NCO_FREQ_STEPS_PER_HZ
            )
            raise ValueError(
                f"IF frequency must be in [-{min_max_frequency_in_hz:e}, {min_max_frequency_in_hz:e}] Hz. "
                f"Got {freq_hz:e} Hz"
            )

        return steps

    def _legalise_bound(self, name: str, bound: IterBound, inst: Instruction):
        if isinstance(inst, DeviceUpdate):
            if inst.attribute == "frequency":
                legal_bound = IterBound(
                    start=self.freq_as_steps(bound.start),
                    step=self.freq_as_steps(bound.step),
                    end=self.freq_as_steps(bound.end),
                    count=bound.count,
                )
            elif inst.attribute == "phase":
                legal_bound = IterBound(
                    start=self.phase_as_steps(bound.start),
                    step=self.phase_as_steps(bound.step),
                    end=self.phase_as_steps(bound.end),
                    count=bound.count,
                )
            else:
                raise NotImplementedError(
                    f"Unsupported processing of attribute {inst.attribute}"
                )
        elif isinstance(inst, Pulse):
            if inst.shape != PulseShapeType.SQUARE:
                raise ValueError("Cannot process non-trivial pulses")

            attribute = next(
                (
                    attr
                    for attr, var in inst.__dict__.items()
                    if isinstance(var, Variable) and var.name == name
                )
            )
            if attribute == "width":
                # TODO - as nanoseconds for now but involve calculate_duration
                legal_bound = IterBound(
                    start=bound.start * 1e9,
                    step=bound.step * 1e9,
                    end=bound.end * 1e9,
                    count=bound.count,
                )
            elif attribute == "amp":
                # TODO - Assumed legal whereby users know about ADC ratio <-> Voltage <-> bit representation rules
                legal_bound = bound
            else:
                raise NotImplementedError(
                    f"Unsupported processing of {attribute} for instruction {inst}"
                )
        else:
            raise NotImplementedError(
                f"Legalisation only supports DeviceUpdate and Pulse. Got {type(inst)} instead"
            )

        if not all(
            isinstance(val, int)
            for val in [
                legal_bound.start,
                legal_bound.step,
                legal_bound.end,
                legal_bound.end,
            ]
        ):
            raise ValueError(
                f"Illegal iter bound. Expected all attribute to be integer, but got {legal_bound} instead"
            )

        # Qblox registers are unsigned 32bit integers, but they are treated as signed integers
        return IterBound(
            start=np.array([legal_bound.start], dtype=int).view(np.uint32)[0],
            step=np.array([legal_bound.step], dtype=int).view(np.uint32)[0],
            end=np.array([legal_bound.end], dtype=int).view(np.uint32)[0],
            count=legal_bound.count,
        )

    def run(self, builder: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        binding_result: BindingResult = res_mgr.lookup_by_type(BindingResult)

        qblox_iter_bounds: Dict[PulseChannel, Dict[str, Set[IterBound]]] = defaultdict(
            lambda: defaultdict(set)
        )
        for name, instructions in binding_result.reads.items():
            for inst in instructions:
                for target in inst.quantum_targets:
                    bound = binding_result.iter_bounds[target][name]
                    legal_bound = self._legalise_bound(name, bound, inst)
                    qblox_iter_bounds[target][name].add(legal_bound)

        for target, symbol2iter_bounds in qblox_iter_bounds.items():
            for name, iter_bounds in symbol2iter_bounds.items():
                if len(iter_bounds) > 1:
                    raise ValueError(f"Ambiguous Qblox bounds for variable {name}")
                binding_result.iter_bounds[target][name] = next(iter(iter_bounds))
