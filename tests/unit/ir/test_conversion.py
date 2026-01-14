# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025-2025 Oxford Quantum Circuits Ltd
from copy import deepcopy
from enum import Enum
from functools import singledispatchmethod
from numbers import Number
from types import NoneType

import numpy as np
import pytest
from compiler_config.config import InlineResultsProcessing

from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager
from qat.ir.conversion import ConvertToPydanticIR
from qat.ir.instruction_builder import InstructionBuilder, QuantumInstructionBuilder
from qat.ir.instructions import (
    Assign,
    FrequencySet,
    InstructionBlock,
    Label,
    PhaseReset,
    QuantumInstructionBlock,
)
from qat.middleend.passes.purr.transform import LoopCount
from qat.model.convert_purr import convert_purr_echo_hw_to_pydantic
from qat.model.loaders.lucy import LucyModelLoader
from qat.model.loaders.purr import EchoModelLoader
from qat.pipelines.echo import EchoPipeline, PipelineConfig
from qat.purr.compiler import instructions
from qat.purr.compiler.devices import PulseChannel, PulseChannelView, PulseShapeType
from qat.utils.pydantic import FrozenSet, ValidatedList, ValidatedSet

from tests.unit.utils.qasm_qir import get_qasm2


class TestFlattenIRPass:
    @pytest.fixture(scope="class")
    def ir(self):
        pass

    @pytest.fixture(scope="class")
    def model(self):
        return LucyModelLoader(4).load()

    def test_flat_ir_does_nothing(self, model):
        ir = QuantumInstructionBuilder(hardware_model=model)
        ir.add(Assign(name="a", value=1))
        ir.add(Label(name="b"))
        ir.add(PhaseReset(target="c"))
        ir.add(FrequencySet(target="d", frequency=5.0))

        ref_ir = deepcopy(ir.instructions)

        ir.flatten()
        assert ir.number_of_instructions == len(ref_ir)
        for instr, ref_instr in zip(ir, ref_ir):
            assert instr == ref_instr

    def test_instruction_block_flattened(self, model):
        ir = QuantumInstructionBuilder(hardware_model=model)
        ir.add(Assign(name="a", value=1))
        ir.add(
            InstructionBlock(
                instructions=[
                    PhaseReset(target="b"),
                    FrequencySet(target="c", frequency=5.0),
                ]
            )
        )
        ref_ir = deepcopy(ir.instructions)

        ir.flatten()
        assert len(ir.instructions) == 3
        assert len(ir.instructions) > len(ref_ir)
        assert isinstance(ir.instructions, ValidatedList)

    def test_quantum_instruction_block_targets_flattened(self, model):
        ir = QuantumInstructionBuilder(hardware_model=model)
        ir.add(
            QuantumInstructionBlock(
                instructions=[
                    PhaseReset(target="b"),
                    FrequencySet(target="c", frequency=5.0),
                ]
            )
        )

        ir.flatten()
        assert len(ir.instructions) == 2
        for instr in ir:
            assert len(instr.targets) == 1

        assert ir._ir.head.target == "b"
        assert ir._ir.tail.target == "c"

    def test_nested_instruction_blocks(self, model):
        ir = QuantumInstructionBuilder(hardware_model=model)
        block = InstructionBlock(
            instructions=[Assign(name="a", value=1), PhaseReset(target="b")]
        )
        nested_block = InstructionBlock(
            instructions=[FrequencySet(target="c", frequency=5.0), block]
        )
        ir.add(block, nested_block)

        another_nested_block = InstructionBlock(instructions=[block, nested_block])
        ir.add(another_nested_block)

        ref_ir = deepcopy(ir)
        ref_nr_instructions = ref_ir.number_of_instructions

        ir.flatten()
        assert len(ir.instructions) == ref_nr_instructions
        for instr, ref_instr in zip(ir.instructions, ref_ir):
            assert instr == ref_instr


class TestConvertToPydanticIRPass:
    def setup_class(self):
        self.legacy_model = EchoModelLoader(qubit_count=16).load()
        self.pyd_model = convert_purr_echo_hw_to_pydantic(self.legacy_model)
        self.converter_pass = ConvertToPydanticIR(self.legacy_model, self.pyd_model)

    def _check_updated_class(self, legacy_element, converted_element):
        """Check that the legacy element is converted to the new model element."""
        assert converted_element.__class__.__name__ == legacy_element.__class__.__name__
        assert converted_element.__class__.__module__.startswith("qat.ir")

    @singledispatchmethod
    def _check_conversion(self, legacy_element, converted_element):
        """Check that the legacy element is converted to the new model element."""
        self._check_updated_class(legacy_element, converted_element)
        for name, value in vars(legacy_element).items():
            self._check_conversion(value, getattr(converted_element, name))

    @_check_conversion.register(type)
    def _(self, legacy_value, converted_value):
        if legacy_value.__module__.startswith("qat."):
            assert converted_value.__name__ == legacy_value.__name__, (
                f"Converted value class does not match: {converted_value.__name__} != {legacy_value.__name__}"
            )
            assert converted_value.__module__.startswith("qat.ir"), (
                f"Converted value class does not start with 'qat.ir': {converted_value.__module__}"
            )
        else:
            assert converted_value == legacy_value, (
                f"Value does not match: {converted_value} != {legacy_value}"
            )

    @_check_conversion.register(str)
    @_check_conversion.register(Number)
    @_check_conversion.register(bool)
    @_check_conversion.register(Enum)
    @_check_conversion.register(NoneType)
    def _(self, legacy_value, converted_value):
        """Check that the value of a legacy element matches the converted element."""
        assert converted_value == legacy_value, (
            f"Value does not match: {converted_value} != {legacy_value}"
        )

    @_check_conversion.register(list)
    @_check_conversion.register(set)
    @_check_conversion.register(tuple)
    def _(self, legacy_value, converted_value):
        """Check that the value of a legacy element matches the converted element."""
        if len(legacy_value) > 0 and isinstance(
            legacy_value[0], PulseChannelView | PulseChannel
        ):
            # The list of channels may not match the order in the converted value,
            # so we check if all channels are present in the converted value.
            assert len(legacy_value) == len(converted_value), (
                "Length of legacy and converted values do not match."
            )
            uuids_legacy = [
                self.converter_pass._pulse_channel_map.get(ch.id, None)
                for ch in legacy_value
            ]
            assert all(uuid in converted_value for uuid in uuids_legacy)
        elif isinstance(converted_value, list | set | tuple | ValidatedSet | ValidatedList):
            if len(legacy_value) > 0 and isinstance(
                legacy_value[0], instructions.Instruction | instructions.Variable
            ):
                # Lists of instructions are a special case since changing the nuber of targets allowed in the pydantic stack
                i: int = 0
                for legacy_val in legacy_value:
                    if (
                        not isinstance(legacy_val, instructions.Synchronize)
                        and hasattr(legacy_val, "quantum_targets")
                        and len(legacy_val.quantum_targets) > 1
                    ):
                        # If the legacy instruction has multiple quantum targets, and isn't a Synchronize instruction,
                        # the converted value should be multiple instructions.
                        target_count = len(legacy_val.quantum_targets)
                        converted_val = converted_value[i : i + target_count]
                        i += target_count
                    else:
                        converted_val = converted_value[i]
                        i += 1
                    self._check_conversion(legacy_val, converted_val)
            else:
                for legacy_val, converted_val in zip(legacy_value, converted_value):
                    self._check_conversion(legacy_val, converted_val)
        else:
            assert False, "Unsupported type for comparison"

    @_check_conversion.register(dict)
    def _(self, legacy_value, converted_value):
        """Check that the value of a legacy element matches the converted element."""
        assert isinstance(converted_value, dict), (
            f"Converted value is not a dict: {converted_value}"
        )
        assert len(legacy_value) == len(converted_value), (
            "Length of legacy and converted values do not match."
        )
        for key, value in legacy_value.items():
            if isinstance(key, PulseChannelView | PulseChannel):
                # If the key is a channel, we need to convert it to the new UUID.
                key = self.converter_pass._pulse_channel_map.get(key.id, None)
            assert key in converted_value, f"Key '{key}' not found in converted value."
            self._check_conversion(value, converted_value[key])

    @_check_conversion.register(PulseChannelView)
    @_check_conversion.register(PulseChannel)
    def _(self, legacy_channel, converted_channel):
        """Check that the value of a legacy channel matches the converted channel."""
        uuid = self.converter_pass._pulse_channel_map.get(legacy_channel.id, None)
        if isinstance(converted_channel, str):
            assert converted_channel == uuid, (
                f"Channel UUID does not match: {converted_channel} != {uuid}"
            )
        else:
            assert converted_channel.uuid == uuid, (
                f"Channel UUID does not match: {converted_channel.uuid} != {uuid}"
            )

    @_check_conversion.register(instructions.Variable)
    @_check_conversion.register(instructions.BinaryOperator)
    def _(self, legacy_value, converted_value):
        """Check that the value of a legacy variable or binary operator matches the converted element."""
        self._check_updated_class(legacy_value, converted_value)
        for name, value in vars(legacy_value).items():
            assert hasattr(converted_value, name), (
                f"Converted instruction does not have field '{name}'"
            )
            self._check_conversion(value, getattr(converted_value, name))

    @_check_conversion.register(instructions.Waveform)
    @_check_conversion.register(instructions.CustomPulse)
    def _(self, legacy_value, converted_value):
        """Check that the value of a legacy waveform matches the converted waveform."""
        for name, value in vars(legacy_value).items():
            if name == "samples":
                assert np.allclose(
                    converted_value.waveform.samples, legacy_value.samples, atol=1e-9
                ), (
                    f"Waveform samples do not match: {converted_value.waveform.samples} != {legacy_value.samples}"
                )
                continue
            elif name == "quantum_targets":
                new_value = converted_value.targets
            elif name in ["duration", "ignore_channel_scale"]:
                new_value = getattr(converted_value, name)
            elif name == "shape":
                new_value = converted_value.waveform.name().lower()
                if legacy_value == instructions.CustomPulse:
                    value = "sampled"
                elif legacy_value.shape == PulseShapeType.GAUSSIAN_DRAG:
                    value = "draggaussian"
                else:
                    value = value.value.lower().replace("_", "")
            elif name[0] == "_":
                continue
            else:
                new_value = getattr(converted_value.waveform, name)
            self._check_conversion(value, new_value)

    @_check_conversion.register(instructions.Instruction)
    def _(self, legacy_value, converted_value):
        """Check that the value of a legacy instruction matches the converted instruction."""
        assert converted_value.__class__.__name__ == legacy_value.__class__.__name__, (
            f"Converted instruction class does not match: {converted_value.__class__.__name__} != {legacy_value.__class__.__name__}"
        )
        assert converted_value.__class__.__module__.startswith("qat.ir"), (
            f"Converted instruction class does not start with 'qat.ir': "
            f"{converted_value.__class__.__module__}"
        )
        for name, value in vars(legacy_value).items():
            if name == "quantum_targets":
                # Convert legacy quantum targets to new targets
                name = "targets"
                assert isinstance(converted_value.targets, FrozenSet)
            elif name == "time":
                # Convert legacy time to duration in new model
                name = "duration"
            elif name == "id" and isinstance(legacy_value, instructions.Acquire):
                # Skip id for Acquire instructions
                continue
            assert hasattr(converted_value, name), (
                f"Converted instruction does not have field '{name}'"
            )
            self._check_conversion(value, getattr(converted_value, name))

    @_check_conversion.register(instructions.Repeat)
    def _(self, legacy_value, converted_value):
        """Check that the value of a legacy repeat instruction matches the converted instruction."""
        self._check_updated_class(legacy_value, converted_value)
        additional_data: dict = {}
        for name, value in vars(legacy_value).items():
            if name in ("repetition_period", "passive_reset_time"):
                additional_data[name] = value
            else:
                self._check_conversion(value, value)
        assert self.converter_pass._additional_data == additional_data, (
            f"Additional data does not match: {self.converter_pass._additional_data} != {additional_data}"
        )

    def test_covert_partitioned_ir(self):
        res_mgr = ResultManager()
        met_mgr = MetricsManager()
        pipe = EchoPipeline(config=PipelineConfig(name="echo"), model=self.legacy_model)
        builder = pipe.frontend.emit(get_qasm2("15qb.qasm"), res_mgr, met_mgr)
        builder = pipe.middleend.emit(builder, res_mgr, met_mgr)
        legacy_ir = pipe.backend.run_pass_pipeline(builder, res_mgr, met_mgr)

        converted_ir = self.converter_pass.run(legacy_ir, res_mgr, met_mgr)
        self._check_conversion(legacy_ir, converted_ir)

    # TODO: Ensure that the legacy instruction builder is compatible with the pydantic version.
    # Variables: COMPILER-589
    # @pytest.mark.skip("Legacy instruction builder is not compatible with pydantic version.")
    def test_convert_instruction_builder(self):
        """Test converting instruction builder."""
        res_mgr = ResultManager()
        met_mgr = MetricsManager()
        pipe = EchoPipeline(config=PipelineConfig(name="echo"), model=self.legacy_model)
        legacy_builder = pipe.frontend.emit(get_qasm2("15qb.qasm"), res_mgr, met_mgr)
        legacy_builder = pipe.middleend.emit(legacy_builder, res_mgr, met_mgr)
        legacy_builder.add(instructions.Reset(self.legacy_model.qubits[:2]))

        converted_builder = self.converter_pass.run(legacy_builder, res_mgr, met_mgr)
        assert isinstance(converted_builder, InstructionBuilder)
        self._check_conversion(
            legacy_builder._instructions, converted_builder.instructions
        )  # Ensure all fields match
        # TODO: Add more checks for the converted builder.
        # COMPILER-589

    @pytest.mark.parametrize(
        "legacy_enum",
        [
            *[val for val in instructions.AcquireMode],
            *[val for val in instructions.PostProcessType],
            *[val for val in instructions.ProcessAxis],
        ],
    )
    def test_convert_enums(self, legacy_enum):
        converted_enum = self.converter_pass._convert_element(legacy_enum)
        self._check_conversion(legacy_enum, converted_enum)  # Ensure all fields match

    @pytest.mark.parametrize(
        "legacy_instruction",
        [
            pytest.param(instructions.Assign("name_1", 8e-9), id="Assign-float"),
            pytest.param(
                instructions.Assign(
                    "name_2", instructions.Plus(instructions.Variable("name_2"), 1)
                ),
                id="Assign-plus",
            ),
            # TODO: Support Sweep and EndSweep in the pydantic stack.
            pytest.param(
                instructions.Sweep,
                id="Sweep",
                marks=pytest.mark.skip("Sweep not supported yet"),
            ),
            pytest.param(
                instructions.EndSweep,
                id="EndSweep",
                marks=pytest.mark.skip("EndSweep not supported yet"),
            ),
            pytest.param(instructions.Repeat(200), id="Repeat-no-repetition_period"),
            pytest.param(
                instructions.Repeat(205, repetition_period=1e-6),
                id="Repeat-with-repetition_period",
            ),
            pytest.param(
                instructions.Repeat(210, repetition_period=1e-6, passive_reset_time=0.5),
                id="Repeat-with-repetition_period-and-passive_reset_time",
            ),
            pytest.param(
                instructions.Repeat(215, passive_reset_time=0.5),
                id="Repeat-with-passive_reset_time",
            ),
            pytest.param(
                instructions.EndRepeat(),
                id="EndRepeat",
            ),
            pytest.param(instructions.Return(), id="Return-no-args"),
            pytest.param(instructions.Return(["var_1", "var_2"]), id="Return-with-args"),
            pytest.param(
                instructions.Jump("sting_label"), id="Jump-str-label-unconditional"
            ),
            pytest.param(
                instructions.Jump(
                    instructions.Label("Label"),
                    condition=instructions.GreaterThan(instructions.Variable("var"), 42),
                ),
                id="Jump-label-conditional",
            ),
            pytest.param(instructions.Label("label_name"), id="Label-with-name"),
            pytest.param(
                instructions.ResultsProcessing("variable", InlineResultsProcessing.Binary),
                id="ResultsProcessing",
            ),
        ],
    )
    def test_convert_classic_instruction(self, legacy_instruction):
        converted_instruction = self.converter_pass._convert_element(legacy_instruction)
        self._check_conversion(
            legacy_instruction, converted_instruction
        )  # Ensure all fields match

    @pytest.mark.parametrize(
        "instruction_type, channel_id, inst_data",
        [
            pytest.param(
                instructions.PhaseSet,
                "CH1.Q0.drive",
                {"phase": 0.5},
                id="PhaseSet",
            ),
            pytest.param(
                instructions.PhaseShift,
                "CH1.Q0.drive",
                {"phase": 0.5},
                id="PhaseShift",
            ),
            pytest.param(
                instructions.PhaseReset,
                "CH1.Q0.drive",
                {},
                id="PhaseReset",
            ),
            pytest.param(
                instructions.FrequencyShift,
                "CH1.Q0.drive",
                {"frequency": 1e9},
                id="FrequencyShift",
            ),
            # TODO: Support converting `Id` instruction in the pydantic stack.
            # COMPILER-592
            pytest.param(
                instructions.Id,
                "CH1.Q0.drive",
                {},
                id="Id",
                marks=pytest.mark.skip("Id not supported yet"),
            ),
            pytest.param(
                instructions.Delay,
                "CH1.Q0.drive",
                {"time": 100e-9},
                id="Delay",
            ),
            pytest.param(
                instructions.Acquire,
                "CH2.R0.acquire",
                {
                    "time": 200e-9,
                    "mode": instructions.AcquireMode.INTEGRATOR,
                    "output_variable": "test_var",
                    "delay": 0.0,
                    "rotation": 0.5,
                    "threshold": 0.1,
                },
                id="Acquire",
            ),
        ],
    )
    def test_convert_quantum_instruction_on_single_pulse_channel(
        self, instruction_type, channel_id, inst_data
    ):
        """Test converting quantum instructions with a single target."""
        channel = self.legacy_model.get_pulse_channel_from_id(channel_id)
        legacy_inst = instruction_type(channel, **inst_data)
        converted_inst = self.converter_pass._convert_element(legacy_inst)
        if isinstance(converted_inst, list):
            # If the converted instruction is a list, it means it has been split into multiple instructions.
            for conv_inst in converted_inst:
                self._check_conversion(legacy_inst, conv_inst)
        else:
            self._check_conversion(legacy_inst, converted_inst)  # Ensure all fields match

    @pytest.mark.parametrize(
        "targets",
        [
            pytest.param(3, id="single-qubit-int-no-channels"),
            pytest.param([0, 1], id="multiple-qubits-list-no-channels"),
            pytest.param(
                [0, 1, "CH1.Q0.drive", "CH3.Q1.drive"],
                id="multiple-qubits-list-with-channels",
            ),
            pytest.param(
                ["CH1.Q0.drive", "CH3.Q1.drive"],
                id="multiple-channels-list-no-qubits",
            ),
        ],
    )
    def test_convert_synchronize_instruction(self, targets):
        """Test converting quantum instructions with qubit target."""
        if isinstance(targets, list):
            legacy_targets = [
                (
                    self.legacy_model.get_qubit(qi)
                    if isinstance(qi, int)
                    else self.legacy_model.get_pulse_channel_from_id(qi)
                )
                for qi in targets
            ]
        elif isinstance(targets, int):
            legacy_targets = self.legacy_model.get_qubit(targets)
        else:
            legacy_targets = self.legacy_model.get_pulse_channel_from_id(targets)

        legacy_inst = instructions.Synchronize(legacy_targets)
        converted_inst = self.converter_pass._convert_element(legacy_inst)

        self._check_conversion(legacy_inst, converted_inst)  # Ensure all fields match

    @pytest.mark.parametrize(
        "qubit_indices",
        [
            pytest.param([0], id="single-qubit-list"),
            pytest.param([0, 1], id="multiple-qubits-list"),
            pytest.param(0, id="single-qubit-int"),
        ],
    )
    def test_convert_reset_instruction(self, qubit_indices):
        """Test converting quantum instructions with qubit target."""
        if isinstance(qubit_indices, int):
            legacy_qubits = self.legacy_model.get_qubit(qubit_indices)
        else:
            legacy_qubits = [self.legacy_model.get_qubit(qi) for qi in qubit_indices]
        legacy_inst = instructions.Reset(legacy_qubits)
        converted_inst = self.converter_pass._convert_element(legacy_inst)

        self._check_conversion(legacy_inst, converted_inst)  # Ensure all fields match

    @_check_conversion.register(instructions.Reset)
    def _(self, legacy_inst, converted_insts):
        """Check that the converted reset instruction matches the legacy reset instruction."""
        assert isinstance(converted_insts, list)
        assert len(converted_insts) == len(legacy_inst.quantum_targets)
        for converted_inst, target in zip(converted_insts, legacy_inst.quantum_targets):
            assert converted_inst.__class__.__name__ == "Reset"
            assert converted_inst.__class__.__module__.startswith("qat.ir")
            assert isinstance(converted_inst.targets, FrozenSet)
            assert converted_inst.qubit_targets == set(
                [self.legacy_model.get_devices_from_pulse_channel(target)[0].index]
            )
            assert converted_inst.duration == legacy_inst.duration

    @pytest.mark.parametrize(
        "pulse_class, channel_id, inst_data",
        [
            pytest.param(
                instructions.CustomPulse,
                "CH1.Q0.drive",
                {
                    "samples": [complex(0.1, 0.2), complex(0.3, 0.4)],
                    "ignore_channel_scale": True,
                },
                id="CustomPulse-ignore_channel_scale",
            ),
            pytest.param(
                instructions.DrivePulse,
                "CH1.Q0.drive",
                {
                    "shape": PulseShapeType.SQUARE,
                    "width": 100e-9,
                    "amp": 0.5,
                    "phase": 0.2,
                    "drag": 0.1,
                },
                id="DrivePulse-SQUARE",
            ),
            pytest.param(
                instructions.MeasurePulse,
                "CH2.R0.measure",
                {
                    "shape": PulseShapeType.GAUSSIAN,
                    "width": 200e-9,
                    "rise": 0.05,
                    "amp_setup": 0.6,
                    "scale_factor": 0.7,
                    "zero_at_edges": 1,
                    "beta": 0.4,
                    "frequency": 0.2,
                    "internal_phase": 0.5,
                    "std_dev": 0.2,
                    "square_width": 0.6,
                },
                id="MeasurePulse-GAUSSIAN",
            ),
            pytest.param(
                instructions.SecondStatePulse,
                "CH1.Q0.drive",
                {
                    "frequency": 1e9,
                    "shape": PulseShapeType.SOFT_SQUARE,
                    "width": 100e-9,
                },
                id="SecondStatePulse-SOFT_SQUARE",
            ),
            pytest.param(
                instructions.CrossResonancePulse,
                "CH1.Q0.Q1.cross_resonance",
                {
                    "frequency": 1e9,
                    "shape": PulseShapeType.BLACKMAN,
                    "width": 100e-9,
                },
                id="CrossResonancePulse-SQUARE",
            ),
            pytest.param(
                instructions.CrossResonanceCancelPulse,
                "CH1.Q0.Q15.cross_resonance_cancellation",
                {
                    "frequency": 1e9,
                    "shape": PulseShapeType.SETUP_HOLD,
                    "width": 100e-9,
                },
                id="CrossResonanceCancelPulse-SETUP_HOLD",
            ),
            pytest.param(
                instructions.Pulse,
                "CH1.Q0.drive",
                {
                    "shape": PulseShapeType.SOFTER_SQUARE,
                    "width": 100e-9,
                },
                id="Pulse-SOFTER_SQUARE",
            ),
            pytest.param(
                instructions.Pulse,
                "CH1.Q0.drive",
                {
                    "shape": PulseShapeType.EXTRA_SOFT_SQUARE,
                    "width": 100e-9,
                },
                id="Pulse-EXTRA_SOFT_SQUARE",
            ),
            pytest.param(
                instructions.Pulse,
                "CH1.Q0.drive",
                {
                    "shape": PulseShapeType.SOFTER_GAUSSIAN,
                    "width": 100e-9,
                },
                id="Pulse-SOFTER_GAUSSIAN",
            ),
            pytest.param(
                instructions.Pulse,
                "CH1.Q0.drive",
                {
                    "shape": PulseShapeType.ROUNDED_SQUARE,
                    "width": 100e-9,
                },
                id="Pulse-ROUNDED_SQUARE",
            ),
            pytest.param(
                instructions.Pulse,
                "CH1.Q0.drive",
                {
                    "shape": PulseShapeType.GAUSSIAN_DRAG,
                    "width": 100e-9,
                },
                id="Pulse-GAUSSIAN_DRAG",
            ),
            pytest.param(
                instructions.Pulse,
                "CH1.Q0.drive",
                {
                    "shape": PulseShapeType.GAUSSIAN_ZERO_EDGE,
                    "width": 100e-9,
                },
                id="Pulse-GAUSSIAN_ZERO_EDGE",
            ),
            pytest.param(
                instructions.Pulse,
                "CH1.Q0.drive",
                {
                    "shape": PulseShapeType.GAUSSIAN_SQUARE,
                    "width": 100e-9,
                },
                id="Pulse-GAUSSIAN_SQUARE",
            ),
            pytest.param(
                instructions.Pulse,
                "CH1.Q0.drive",
                {
                    "shape": PulseShapeType.SECH,
                    "width": 100e-9,
                },
                id="Pulse-SECH",
            ),
            pytest.param(
                instructions.Pulse,
                "CH1.Q0.drive",
                {
                    "shape": PulseShapeType.SIN,
                    "width": 100e-9,
                },
                id="Pulse-SIN",
            ),
            pytest.param(
                instructions.Pulse,
                "CH1.Q0.drive",
                {
                    "shape": PulseShapeType.COS,
                    "width": 100e-9,
                },
                id="Pulse-COS",
            ),
        ],
    )
    def test_convert_waveform(self, pulse_class, channel_id, inst_data):
        """Test converting waveform instructions."""
        channel = self.legacy_model.get_pulse_channel_from_id(channel_id)
        legacy_pulse = pulse_class(channel, **inst_data)
        converted_waveform = self.converter_pass._convert_element(legacy_pulse)

        self._check_conversion(legacy_pulse, converted_waveform)  # Ensure all fields match

    @_check_conversion.register(instructions.Waveform)
    def _(self, legacy_pulse, converted_waveform):
        assert converted_waveform.__class__.__name__ == "Pulse"
        assert converted_waveform.__class__.__module__.startswith("qat.ir")
        for name, value in vars(legacy_pulse).items():
            if name == "samples":
                assert np.allclose(
                    converted_waveform.waveform.samples, legacy_pulse.samples, atol=1e-9
                )
                continue
            elif name == "quantum_targets":
                new_value = converted_waveform.targets
            elif name in ["duration", "ignore_channel_scale"]:
                new_value = getattr(converted_waveform, name)
            elif name == "shape":
                new_value = converted_waveform.waveform.name().lower()
                if legacy_pulse == instructions.CustomPulse:
                    value = "sampled"
                elif legacy_pulse.shape == PulseShapeType.GAUSSIAN_DRAG:
                    value = "draggaussian"
                else:
                    value = value.value.lower().replace("_", "")
            else:
                new_value = getattr(converted_waveform.waveform, name)
            self._check_conversion(value, new_value)

    @pytest.mark.parametrize(
        "pp_type, axes",
        [
            pytest.param(
                instructions.PostProcessType.MEAN,
                [instructions.ProcessAxis.TIME],
                id="Mean-Time",
            ),
            pytest.param(
                instructions.PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL,
                [instructions.ProcessAxis.SEQUENCE],
                id="LinearMapComplexToReal-Sequence",
            ),
            pytest.param(
                instructions.PostProcessType.DISCRIMINATE,
                [instructions.ProcessAxis.SEQUENCE],
                id="Discriminate-Sequence",
            ),
            pytest.param(
                instructions.PostProcessType.MUL,
                [
                    instructions.ProcessAxis.SEQUENCE,
                    instructions.ProcessAxis.TIME,
                ],
                id="Mul-Sequence-Time",
            ),
        ],
    )
    def test_convert_post_process(self, pp_type, axes):
        """Test converting post-processing instructions."""
        acquire = instructions.Acquire(
            self.legacy_model.get_pulse_channel_from_id("CH2.R0.acquire"),
            time=200e-9,
            mode=instructions.AcquireMode.INTEGRATOR,
            output_variable="test_var",
            delay=0.0,
            rotation=0.5,
            threshold=0.1,
        )
        legacy_post_process = instructions.PostProcessing(
            acquire=acquire,
            process=pp_type,
            axes=axes,
            args=[0.0, 1.0],
        )
        converted_post_process = self.converter_pass._convert_element(legacy_post_process)

        self._check_conversion(legacy_post_process, converted_post_process)

    @_check_conversion.register(instructions.PostProcessing)
    def _(self, legacy_post_process, converted_post_process):
        assert (
            converted_post_process.__class__.__name__
            == legacy_post_process.__class__.__name__
        )
        assert converted_post_process.__class__.__module__.startswith("qat.ir")
        assert converted_post_process.process_type == legacy_post_process.process
        assert converted_post_process.axes == legacy_post_process.axes
        assert converted_post_process.args == legacy_post_process.args
        assert converted_post_process.output_variable == legacy_post_process.output_variable
        assert converted_post_process.result_needed == legacy_post_process.result_needed

    @pytest.mark.parametrize(
        "qubit_indices, acquire_mode, output_variables",
        [
            pytest.param(
                [0],
                instructions.AcquireMode.INTEGRATOR,
                ["test_var"],
                id="single-qubit",
            ),
            pytest.param(
                [0, 1],
                instructions.AcquireMode.SCOPE,
                ["test_var_1", "test_var_2"],
                id="multiple-qubits",
            ),
            pytest.param(
                8,
                instructions.AcquireMode.RAW,
                ["test_var"],
                id="single-qubit-int",
            ),
        ],
    )
    def test_convert_measure_block(self, qubit_indices, acquire_mode, output_variables):
        """Test converting measure block instructions."""
        if isinstance(qubit_indices, int):
            targets = self.legacy_model.get_qubit(qubit_indices)
            qubit_indices = [qubit_indices]
        else:
            targets = [self.legacy_model.get_qubit(qi) for qi in qubit_indices]
        legacy_measure_block = instructions.MeasureBlock(
            targets=targets,
            mode=acquire_mode,
            output_variables=output_variables,
        )
        converted_measure_block = self.converter_pass._convert_element(legacy_measure_block)
        self._check_conversion(
            legacy_measure_block, converted_measure_block
        )  # Ensure all fields match

    @_check_conversion.register(instructions.MeasureBlock)
    def _(self, legacy_measure_block, converted_measure_block):
        """Check that the converted measure block matches the legacy measure block."""
        assert (
            converted_measure_block.__class__.__name__
            == legacy_measure_block.__class__.__name__
        )
        assert converted_measure_block.__class__.__module__.startswith("qat.ir")
        assert list(converted_measure_block.qubit_targets) == [
            qubit.index for qubit in legacy_measure_block.quantum_targets
        ]
        assert converted_measure_block.duration == legacy_measure_block.duration
        assert converted_measure_block.output_variables == [
            target["output_variable"]
            for target in legacy_measure_block._target_dict.values()
        ]

    @pytest.mark.parametrize(
        "legacy_operator",
        [
            pytest.param(
                instructions.Equals(instructions.Variable("a"), 1), id="Equals-var-int"
            ),
            pytest.param(
                instructions.NotEquals(instructions.Variable("b"), 0.2),
                id="NotEquals-var-float",
            ),
            pytest.param(
                instructions.GreaterThan(42, instructions.Variable("register")),
                id="GreaterThan-int-var",
            ),
            pytest.param(
                instructions.GreaterOrEqualThan(1e-7, instructions.Variable("var")),
                id="GreaterOrEqualThan-float-var",
            ),
            pytest.param(instructions.LessThan(2, 1.93), id="LessThan-int-float"),
            pytest.param(instructions.LessOrEqualThan(2, 1), id="LessOrEqualThan-int-int"),
            pytest.param(
                instructions.Plus(
                    instructions.Variable("left"), instructions.Variable("right")
                ),
                id="Plus-var-var",
            ),
        ],
    )
    def test_convert_binary_operator(self, legacy_operator):
        converted_operator = self.converter_pass._convert_element(legacy_operator)
        self._check_conversion(legacy_operator, converted_operator)

    @pytest.mark.parametrize(
        "legacy_variable",
        [
            pytest.param(instructions.Variable("test_var"), id="Variable"),
            pytest.param(instructions.Variable("test_var", value=0.5), id="Variable-float"),
            pytest.param(instructions.Variable("test_var", value=42), id="Variable-int"),
            pytest.param(
                instructions.Variable("test_var", LoopCount, 0),
                id="Variable-loop_count",
            ),
            # TODO: Support converting `IndexAccessor` in the pydantic stack?
            # COMPILER-593
            pytest.param(
                instructions.IndexAccessor("index_var", 0),
                id="IndexAccessor",
                marks=pytest.mark.skip("IndexAccessor not supported yet"),
            ),
        ],
    )
    def test_convert_variable(self, legacy_variable):
        converted_variable = self.converter_pass._convert_element(legacy_variable)
        self._check_conversion(legacy_variable, converted_variable)
