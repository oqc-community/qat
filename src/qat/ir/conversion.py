# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
from enum import Enum
from functools import singledispatchmethod
from numbers import Number
from types import NoneType

import qat.ir.instructions as pyd_instructions
import qat.ir.measure as pyd_measure
import qat.ir.waveforms as pyd_waveforms
from qat.core.pass_base import TransformPass
from qat.ir.instruction_builder import PydQuantumInstructionBuilder
from qat.ir.lowered import PartitionedIR
from qat.middleend.passes.legacy.transform import LoopCount
from qat.model.hardware_model import PhysicalHardwareModel
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import (
    PulseChannel,
    PulseChannelView,
    PulseShapeType,
    Qubit,
)
from qat.purr.compiler.hardware_models import HardwareModel
from qat.purr.compiler.instructions import (
    Assign,
    BinaryOperator,
    CustomPulse,
    Instruction,
    InstructionBlock,
    Jump,
    PostProcessing,
    Pulse,
    Reset,
    ResultsProcessing,
    Variable,
)


class ConvertToPydanticIR(TransformPass):
    def __init__(
        self,
        legacy_model: HardwareModel,
        pyd_model: PhysicalHardwareModel,
        *args,
        **kwargs,
    ):
        self.legacy_model = legacy_model
        self.pyd_model = pyd_model
        self._create_pulse_channel_map()

    def _create_pulse_channel_map(self):
        """Create a mapping of legacy pulse channels to Pydantic pulse channels."""
        self._pulse_channel_map = {}
        self._pulse_channel_to_qubit_index_map = {}
        for index, qubit in self.pyd_model.qubits.items():
            q_key = f"Q{index}"
            self._pulse_channel_map[q_key + ".drive"] = qubit.drive_pulse_channel.uuid
            self._pulse_channel_to_qubit_index_map[q_key + ".drive"] = index
            self._pulse_channel_map[q_key + ".freq_shift"] = (
                qubit.freq_shift_pulse_channel.uuid
            )
            self._pulse_channel_map[q_key + ".second_state"] = (
                qubit.second_state_pulse_channel.uuid
            )
            for other_index, channel in qubit.cross_resonance_pulse_channels.items():
                self._pulse_channel_map[f"{q_key}.Q{other_index}.cross_resonance"] = (
                    channel.uuid
                )
            for (
                other_index,
                channel,
            ) in qubit.cross_resonance_cancellation_pulse_channels.items():
                self._pulse_channel_map[
                    f"{q_key}.Q{other_index}.cross_resonance_cancellation"
                ] = channel.uuid
            resonator = qubit.resonator
            r_key = f"R{index}"
            self._pulse_channel_map[r_key + ".measure"] = (
                resonator.measure_pulse_channel.uuid
            )
            self._pulse_channel_map[r_key + ".acquire"] = (
                resonator.acquire_pulse_channel.uuid
            )

    def run(self, ir: InstructionBuilder | PartitionedIR, *args, **kwargs):
        if isinstance(ir, PartitionedIR):
            new_ir = self._convert_partitioned_ir(ir)
        elif isinstance(ir, InstructionBuilder):
            new_ir = self._convert_instruction_builder(ir)
        else:
            raise TypeError(f"Unsupported IR type: {type(ir)}")
        return new_ir

    def _convert_partitioned_ir(self, ir: PartitionedIR) -> PartitionedIR:
        new_ir = PartitionedIR()
        for attr, value in vars(ir).items():
            setattr(new_ir, attr, self._convert_element(value))
        return new_ir

    def _convert_instruction_builder(
        self, ir: InstructionBuilder
    ) -> PydQuantumInstructionBuilder:
        # TODO: Implement full compatibility of InstructionBuilders
        # Variables: COMPILER-589
        new_ir = PydQuantumInstructionBuilder(
            hardware_model=self.pyd_model,
        )
        for attr, value in vars(ir).items():
            if attr == "_instructions":
                attr = "instructions"
            setattr(new_ir, attr, self._convert_element(value))
        return new_ir

    @singledispatchmethod
    def _convert_element(self, value):
        """Default method for converting elements. This will raise an error if no specific
        conversion is defined for the type of `value`.
        """
        raise TypeError(f"Unsupported type for conversion: {type(value)}")

    @_convert_element.register(type)
    def _(
        self,
        value: type,
    ):
        """Convert a type to its Pydantic equivalent."""
        pyd_class = self._get_pyd_class(value.__name__, allow_none=True)
        if pyd_class is None:
            return value
        return pyd_class

    @_convert_element.register(Number)
    @_convert_element.register(str)
    @_convert_element.register(NoneType)
    def _(
        self,
        value: Number | str | None,
    ):
        return value

    @_convert_element.register(LoopCount)
    def _(
        self,
        value: LoopCount,
    ):
        """Convert a LoopCount instance."""
        return pyd_instructions.LoopCount(value)

    @_convert_element.register(list)
    @_convert_element.register(tuple)
    @_convert_element.register(set)
    def _(
        self,
        value: list | tuple | set,
    ):
        if all(isinstance(v, (Number, str)) for v in value):
            return value
        else:
            temp_list = []
            for item in value:
                new_item = self._convert_element(item)
                if isinstance(new_item, list) and not isinstance(item, list):
                    # If the item is a list and the original value is not a list,
                    # we need to flatten it.
                    temp_list.extend(new_item)
                else:
                    temp_list.append(new_item)
            return type(value)(temp_list)

    @_convert_element.register(dict)
    def _(
        self,
        value: dict,
    ):
        """Convert a dictionary by iterating over its items."""
        new_dict = {}
        for key, val in value.items():
            new_key = self._convert_element(key)
            if isinstance(val, (int, float, str)):
                new_dict[new_key] = val
            else:
                new_dict[new_key] = self._convert_element(val)
        return new_dict

    @_convert_element.register(HardwareModel)
    def _(
        self,
        value: HardwareModel,
    ):
        return self.pyd_model

    @_convert_element.register(Variable)
    def _(
        self,
        value: Variable,
    ):
        """Convert a Variable instance."""
        if value.var_type is LoopCount and isinstance(value.value, int):
            value.value = LoopCount(value.value)
        new_var = pyd_instructions.Variable(
            name=value.name,
            var_type=self._convert_element(value.var_type),
            value=self._convert_element(value.value),
        )
        return new_var

    @_convert_element.register(PulseChannelView)
    @_convert_element.register(PulseChannel)
    def _(
        self,
        value: PulseChannelView | PulseChannel,
    ):
        """Convert a PulseChannelView instance."""
        return self._pulse_channel_map[value.id]

    @_convert_element.register(Qubit)
    def _(
        self,
        value: Qubit,
    ):
        """Convert a Qubit instance."""
        return value.index

    @_convert_element.register(Instruction)
    @_convert_element.register(BinaryOperator)
    def _(
        self,
        value: Instruction | BinaryOperator,
    ):
        """Convert an Instruction instance."""
        data = dict()
        for name, var in vars(value).items():
            if name == "quantum_targets":
                data["targets"] = frozenset(self._convert_element(var))
            elif name == "time":
                data["duration"] = self._convert_element(var)
            else:
                data[name] = self._convert_element(var)
        pyd_class = self._get_pyd_class(value)
        return pyd_class(**data)

    @_convert_element.register(InstructionBlock)
    def _(
        self,
        value: InstructionBlock,
    ):
        """Convert an InstructionBlock instance."""
        pyd_class = self._get_pyd_class(value)
        extra_data = {}
        if "qubit_targets" in pyd_class.model_fields:
            extra_data["qubit_targets"] = frozenset(
                self._convert_element(value.quantum_targets)
            )
        new_block = pyd_class(**extra_data)
        for instruction in value.instructions:
            new_block.add(self._convert_element(instruction))
        return new_block

    @_convert_element.register(Reset)
    def _(
        self,
        value: Reset,
    ):
        """Convert a Reset instance."""
        qubit_targets = set(
            [
                self._pulse_channel_to_qubit_index_map[target.id]
                for target in value.quantum_targets
            ]
        )
        duration = self._convert_element(value.duration)
        return [
            pyd_instructions.Reset(
                qubit_targets=target,
                duration=duration,
            )
            for target in qubit_targets
        ]

    @_convert_element.register(Assign)
    @_convert_element.register(ResultsProcessing)
    @_convert_element.register(PostProcessing)
    def _(
        self,
        value: Assign | ResultsProcessing | PostProcessing,
    ):
        """Convert an Assign instance."""
        return self._get_pyd_class(value)._from_legacy(value)

    @_convert_element.register(Jump)
    def _(
        self,
        value: Jump,
    ):
        """Convert a Jump instance."""
        return pyd_instructions.Jump(
            label=self._convert_element(value.target),
            condition=self._convert_element(value.condition),
        )

    @_convert_element.register(Enum)
    def _(
        self,
        value: Enum,
    ):
        """Convert an Enum instance."""
        pyd_class = self._get_pyd_class(value)
        return pyd_class(value.value)

    @_convert_element.register(PulseShapeType)
    def _(
        self,
        value: PulseShapeType,
    ):
        """Convert a PulseShapeType instance."""
        if value.name == "GAUSSIAN_DRAG":
            class_name = "DragGaussianWaveform"
        else:
            class_name = f"{value.name.title().replace('_', '')}Waveform"
        pyd_class = getattr(pyd_waveforms, class_name, None)
        if pyd_class is None:
            raise TypeError(f"Unsupported PulseShapeType: {value}")
        return pyd_class

    @_convert_element.register(CustomPulse)
    def _(
        self,
        value: CustomPulse,
    ):
        """Convert a CustomPulse instance."""
        return pyd_waveforms.Pulse(
            waveform=pyd_waveforms.SampledWaveform(samples=value.samples),
            targets=frozenset(self._convert_element(value.quantum_targets)),
            ignore_channel_scale=value.ignore_channel_scale,
            duration=value.duration,
        )

    @_convert_element.register(Pulse)
    def _(
        self,
        value: Pulse,
    ):
        """Convert a Pulse instance."""
        waveform_data = dict()
        pulse_data = dict()
        for name, var in vars(value).items():
            if name == "quantum_targets":
                pulse_data["targets"] = frozenset(self._convert_element(var))
            elif name in ("ignore_channel_scale", "duration"):
                pulse_data[name] = var
            else:
                waveform_data[name] = self._convert_element(var)
        waveform_class = waveform_data.pop("shape")
        return pyd_waveforms.Pulse(
            waveform=waveform_class(**waveform_data),
            **pulse_data,
        )

    @staticmethod
    def _get_pyd_class(val: object | str, allow_none: bool = False) -> type | None:
        """Get the Pydantic class corresponding to the legacy instruction name."""
        name = val.__class__.__name__ if not isinstance(val, str) else val
        pyd_class = getattr(pyd_instructions, name, None)
        if pyd_class is None:
            pyd_class = getattr(pyd_measure, name, None)
        if pyd_class is None and not allow_none:
            raise TypeError(f"Unsupported type: {name}")
        return pyd_class
