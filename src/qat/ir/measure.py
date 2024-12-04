from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Set, Union

import numpy as np
from pydantic import BaseModel, ValidationInfo, field_validator

from qat.ir.instructions import Instruction, QuantumInstruction, Synchronize, Variable
from qat.ir.waveforms import Pulse, PulseType, Waveform
from qat.purr.compiler.devices import PulseChannel, Qubit

# The following things from legacy instructions are unchanged, so just import for now.
from qat.purr.compiler.instructions import (
    AcquireMode,
    PostProcessType,
    ProcessAxis,
    build_generated_name,
)
from qat.purr.utils.logger import get_default_logger


# previously also inheritted QuantumComponent: I do not know why...
# might need adding back later
class Acquire(QuantumInstruction):
    inst: Literal["Acquire"] = "Acquire"
    suffix_incrementor: int = 0
    time: Union[float, Variable] = 1e-6
    mode: AcquireMode = AcquireMode.RAW
    output_variable: str = None
    delay: Optional[float] = None
    filter: Optional[Pulse] = None

    @property
    def duration(self):
        return self.time

    @property
    def channel(self):
        return self.targets

    def __repr__(self):
        out_var = f"->{self.output_variable}" if self.output_variable else ""
        mode = f",{self.mode.value}" if self.mode is not None else ""
        return f"acquire {self.channel},{self.time}{mode}{out_var}"

    @field_validator("filter", mode="before")
    @classmethod
    def _validate_filter(cls, filter, info: ValidationInfo):
        if filter == None:
            return filter

        time = info.data["time"]
        if isinstance(time, Variable):
            get_default_logger.warning(f"Filter cannot be applied when time is variable.")
            return None

        if not isinstance(filter, Pulse):
            raise ValueError("Filter must be a Waveform.")

        if filter.duration < 0:
            raise ValueError(
                f"Filter duration {filter.duration} cannot be less than or equal to zero."
            )

        if not np.isclose(filter.duration, time, atol=1e-9):
            raise ValueError(
                f"Filter duration '{filter.duration}' must be equal to Acquire "
                f"duration '{time}'."
            )

        return filter

    @classmethod
    def with_random_output_variable(
        cls,
        target: PulseChannel,
        time: Union[float, Variable] = 1e-6,
        mode: AcquireMode = AcquireMode.RAW,
        output_variable: str = None,
        delay: float = None,
        filter: Pulse = None,
        existing_names=None,
    ):
        if not output_variable:
            output_variable = build_generated_name(existing_names, target.full_id())
        return Acquire(
            targets=target,
            time=time,
            mode=mode,
            output_variable=output_variable,
            delay=delay,
            filter=filter,
        )


class PostProcessing(Instruction):
    """
    States what post-processing should happen after data has been acquired. This can
    happen in the FPGA's or a software post-process.
    """

    inst: Literal["PostProcessing"] = "PostProcessing"
    acquire: Acquire
    process: PostProcessType
    axes: List[ProcessAxis] = []
    args: List[Any] = []
    result_needed: bool = False

    @property
    def output_variable(self):
        return self.acquire.output_variable

    def __repr__(self):
        axis = ",".join([axi.value for axi in self.axes])
        args = f",{','.join(str(arg) for arg in self.args)}" if len(self.args) > 0 else ","
        output_var = f"->{self.output_variable}" if self.output_variable is not None else ""
        return (
            f"{self.process.value} {self.acquire.output_variable}{args}{axis}{output_var}"
        )

    @field_validator("axes", mode="before")
    @classmethod
    def _axes_as_list(cls, axes):
        if axes:
            return axes if isinstance(axes, List) else [axes]
        return []


class MeasureData(BaseModel):
    mode: AcquireMode
    output_variable: Optional[str] = None
    measure: Pulse
    acquire: Acquire
    duration: float
    targets: List[str] = None


class MeasureBlock(Instruction):
    """Groups multiple qubit measurements together."""

    inst: Literal["MeasureBlock"] = "MeasureBlock"
    target_dict: Dict[str, MeasureData] = {}
    existing_names: Optional[Set[str]] = set()
    duration: float = 0.0

    @property
    def instructions(self):
        instructions = [Synchronize(self.pulse_channel_targets)]
        for val in self.target_dict.values():
            instructions.extend([val.measure, val.acquire])
        instructions.append(Synchronize(self.pulse_channel_targets))
        return instructions

    @property
    def targets(self):
        return list(self.target_dict.keys())

    @property
    def pulse_channel_targets(self):
        targets = []
        for target in self.target_dict.values():
            targets.extend(target.targets)
        return targets

    def get_acquires(self, targets: Union[Qubit, List[Qubit]]):
        if not isinstance(targets, list):
            targets = [targets]
        targets = [t.full_id() if isinstance(t, Qubit) else t for t in targets]
        return [self.target_dict[qt].acquire for qt in targets]

    @staticmethod
    def create_block(
        qubit: Union[Qubit, List[Qubit]],
        mode: AcquireMode,
        output_variables: str = None,
        existing_names=set(),
    ):
        """
        Initiate a measure block by specifying a qubit / a list of qubits and an acquire
        mode.
        """
        measure_block = MeasureBlock(existing_names=existing_names)
        measure_block.add_measurements(qubit, mode, output_variables)
        return measure_block

    def add_measurements(
        self,
        targets: Union[Qubit, List[Qubit]],
        mode: AcquireMode,
        output_variables: Union[str, List[str]] = None,
        existing_names: Set[str] = None,
    ):
        """
        Adds qubits to the measure block.

        In the future, we should consider moving this to e.g. the Instruction Builder.
        """
        targets = [targets] if not isinstance(targets, List) else targets
        invalid_items = [target for target in targets if not isinstance(target, Qubit)]
        if len(invalid_items) > 0:
            invalid_items_str = ",".join([str(item) for item in invalid_items])
            raise ValueError(f"The following items are not Qubits: {invalid_items_str}")

        if len((duplicates := [t for t in targets if t.full_id() in self.targets])) > 0:
            raise ValueError(
                "Target can only be measured once in a 'MeasureBlock'. "
                f"Duplicates: {duplicates}"
            )

        if not isinstance(output_variables, list):
            output_variables = [] if output_variables is None else [output_variables]
        if (num_out_vars := len(output_variables)) == 0:
            output_variables = [None] * len(targets)
        elif num_out_vars != len(targets):
            raise ValueError(
                f"Unsupported number of `output_variables`: {num_out_vars}, "
                f"must be `None` or match numer of targets: {len(targets)}."
            )

        for target, output_variable in zip(targets, output_variables):
            meas, acq = self._generate_measure_acquire(
                target, mode, output_variable, existing_names
            )
            duration = max(meas.duration, acq.delay + acq.duration)
            self.target_dict[target.full_id()] = MeasureData(
                mode=mode,
                output_variable=output_variable,
                measure=meas,
                acquire=acq,
                duration=duration,
                targets=[pc.full_id() for pc in target.pulse_channels.values()],
            )
            self.duration = max(self.duration, duration)

    def _generate_measure_acquire(self, qubit, mode, output_variable, existing_names):
        measure_channel = qubit.get_measure_channel()
        acquire_channel = qubit.get_acquire_channel()
        existing_names = existing_names or self.existing_names
        weights = (
            qubit.measure_acquire.get("weights", None)
            if qubit.measure_acquire.get("use_weights", False)
            else None
        )

        measure_instruction = Pulse(
            targets=measure_channel,
            waveform=Waveform(**qubit.pulse_measure),
            type=PulseType.MEASURE,
        )
        acquire_instruction = Acquire.with_random_output_variable(
            acquire_channel,
            time=(
                qubit.pulse_measure["width"]
                if qubit.measure_acquire["sync"]
                else qubit.measure_acquire["width"]
            ),
            mode=mode,
            output_variable=output_variable,
            existing_names=existing_names,
            delay=qubit.measure_acquire["delay"],
            filter=weights,
        )

        return [
            measure_instruction,
            acquire_instruction,
        ]

    def __repr__(self):
        target_strings = []
        for q, d in self.target_dict.items():
            out_var = f"->{d.output_variable}" if d.output_variable is not None else ""
            mode = f":{d.mode.value}" if d.mode is not None else ""
            target_strings.append(f"{q}{mode}{out_var}")
        return f"Measure {', '.join(target_strings)}"
