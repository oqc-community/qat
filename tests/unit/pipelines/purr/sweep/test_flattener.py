# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import numpy as np
import pytest

from qat.model.loaders.purr.echo import EchoModelLoader
from qat.pipelines.purr.sweep.flattener import (
    DeviceAssignSet,
    SweepFlattener,
    VariableAccessor,
)
from qat.purr.compiler.devices import Qubit
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import (
    Acquire,
    Delay,
    DeviceUpdate,
    PhaseShift,
    Pulse,
    PulseShapeType,
    Sweep,
    Variable,
)

from .utils import (
    sweep_pulse_scales,
    sweep_pulse_widths,
    sweep_pulse_widths_and_amps,
    sweep_sequential_pulse_widths,
    sweep_zipped_parameters,
)


class TestVariableAccessor:
    model: QuantumHardwareModel = EchoModelLoader().load()
    drive_channel = model.get_qubit(0).get_drive_channel()
    acquire_channel = model.get_qubit(0).get_acquire_channel()

    @pytest.mark.parametrize(
        "variable, attribute",
        [(Variable("t"), "width"), (Variable("amp"), "amp"), (Variable("p"), "phase")],
    )
    def test_pulse_with_single_variable(self, variable, attribute):
        # set up pulse
        args = {
            "quantum_target": self.drive_channel,
            "shape": PulseShapeType.GAUSSIAN,
            "width": 20e-9,
            "amp": 1.0,
            "phase": 0.0,
        }
        args[attribute] = variable
        instr = Pulse(**args)

        # create injector
        injectors = VariableAccessor.create_accessors(instr, 0, ["t", "amp", "p"])
        assert len(injectors) == 1
        injector = injectors[0]
        assert injector.variable_name == variable.name
        assert injector.attribute == attribute

    def test_pulse_with_multiple_variables(self):
        # set up pulse
        instr = Pulse(
            quantum_target=self.drive_channel,
            shape=PulseShapeType.GAUSSIAN,
            width=Variable("t"),
            amp=Variable("amp"),
            phase=0.0,
        )

        # create injector
        injectors = VariableAccessor.create_accessors(instr, 0, ["t", "amp", "p"])
        assert len(injectors) == 2
        names = {inj.variable_name for inj in injectors}
        assert names == {"t", "amp"}
        attributes = {inj.attribute for inj in injectors}
        assert attributes == {"width", "amp"}

    def test_no_variable(self):
        instr = Pulse(
            quantum_target=self.drive_channel,
            shape=PulseShapeType.GAUSSIAN,
            width=20e-9,
            amp=1.0,
            phase=0.0,
        )
        injectors = VariableAccessor.create_accessors(instr, 0, ["t", "amp", "p"])
        assert len(injectors) == 0

    def test_delay(self):
        variable = Variable("t")
        instr = Delay(quantum_target=self.drive_channel, time=variable)
        injectors = VariableAccessor.create_accessors(instr, 0, "t")
        assert len(injectors) == 1
        injector = injectors[0]
        assert injector.variable_name == "t"
        assert injector.attribute == "time"

    def test_phase_shift(self):
        variable = Variable("p")
        instr = PhaseShift(self.drive_channel, phase=variable)
        injectors = VariableAccessor.create_accessors(instr, 0, "p")
        assert len(injectors) == 1
        injector = injectors[0]
        assert injector.variable_name == "p"
        assert injector.attribute == "phase"

    @pytest.mark.parametrize(
        "variable, attribute", [(Variable("t"), "time"), (Variable("d"), "delay")]
    )
    def test_acquire(self, variable, attribute):
        # set up acquire
        args = {
            "channel": self.acquire_channel,
            "output_variable": "x",
            "time": 800e-9,
            "delay": 80e-9,
        }
        args[attribute] = variable
        instr = Acquire(**args)

        # create injector
        injectors = VariableAccessor.create_accessors(instr, 0, ["t", "d"])
        assert len(injectors) == 1
        injector = injectors[0]
        assert injector.variable_name == variable.name
        assert injector.attribute == attribute

    def test_device_update(
        self,
    ):
        variable = Variable("f")
        instr = DeviceUpdate(target=self.drive_channel, attribute="scale", value=variable)
        injectors = VariableAccessor.create_accessors(instr, 0, ["f"])
        assert len(injectors) == 1
        injector = injectors[0]
        assert injector.variable_name == "f"
        assert injector.attribute == "value"


class TestDeviceAssignSet:
    def test_apply(self):
        model: QuantumHardwareModel = EchoModelLoader().load()
        channel = model.get_qubit(0).get_drive_channel()

        original_scale = channel.scale
        original_phase_offset = channel.phase_offset

        assign1 = DeviceUpdate(target=channel, attribute="scale", value=0.95)
        assign2 = DeviceUpdate(target=channel, attribute="phase_offset", value=0.02)
        assign_set = DeviceAssignSet([assign1, assign2])

        assert len(assign_set.assigns) == 2
        assert assign_set.assigns[0] == assign1
        assert assign_set.assigns[1] == assign2

        with assign_set.apply():
            new_channel_ref = model.get_qubit(0).get_drive_channel()
            assert new_channel_ref.scale == 0.95
            assert new_channel_ref.phase_offset == 0.02

        new_channel_ref = model.get_qubit(0).get_drive_channel()
        assert new_channel_ref.scale == original_scale
        assert new_channel_ref.phase_offset == original_phase_offset


class TestSweepFlattenerWithSingleSweep:
    model: QuantumHardwareModel = EchoModelLoader().load()
    qubit: Qubit = model.get_qubit(0)
    times = np.linspace(80e-9, 800e-9, 10)

    @pytest.fixture(scope="class")
    def sweep_builder(self):
        return sweep_pulse_widths(self.model, qubit=0, times=self.times)

    @pytest.fixture(scope="class")
    def sweep_flattener(self, sweep_builder):
        return SweepFlattener(sweep_builder)

    @pytest.fixture(scope="class")
    def flattened_builders(self, sweep_flattener):
        return sweep_flattener.create_flattened_builders()

    def test_extract_sweeps(self, sweep_flattener, flattened_builders):
        assert len(sweep_flattener.sweeps) == 1
        assert len(flattened_builders) == 10
        assert "t" in sweep_flattener.sweep_names
        assert sweep_flattener.sweep_sizes == [len(self.times)]
        assert len(sweep_flattener.device_assigns) == 0

    def test_accessors(self, sweep_flattener):
        accessors = sweep_flattener.accessors
        assert len(accessors) == 1
        accessor = accessors[0]
        assert accessor.variable_name == "t"
        assert accessor.instruction_index == 0
        assert accessor.attribute == "width"

    def test_flattened_builders_have_variables_injected(self, flattened_builders):
        for sweep_instance in flattened_builders:
            params = sweep_instance.variables
            builder = sweep_instance.builder
            assert len(params) == 1
            sweep_instrs = [
                instr for instr in builder.instructions if isinstance(instr, Sweep)
            ]
            assert len(sweep_instrs) == 0
            pulse_instrs = [
                instr for instr in builder.instructions if isinstance(instr, Pulse)
            ]
            assert len(pulse_instrs) == 2
            pulse = pulse_instrs[0]
            assert isinstance(pulse.width, float)
            assert np.isclose(pulse.width, params["t"])
            assert sweep_instance.device_assigns.assigns == []


class TestSweepFlattenerWithMultipleSweeps:
    model: QuantumHardwareModel = EchoModelLoader().load()
    qubit: Qubit = model.get_qubit(0)
    times = np.linspace(80e-9, 800e-9, 5)
    amps = np.linspace(0.1, 1.0, 4)

    @pytest.fixture(scope="class")
    def sweep_builder(self):
        return sweep_pulse_widths_and_amps(self.model, 0, self.times, self.amps)

    @pytest.fixture(scope="class")
    def sweep_flattener(self, sweep_builder):
        return SweepFlattener(sweep_builder)

    @pytest.fixture(scope="class")
    def flattened_builders(self, sweep_flattener):
        return sweep_flattener.create_flattened_builders()

    def test_extract_sweeps(self, sweep_flattener, flattened_builders):
        assert len(sweep_flattener.sweeps) == 2
        assert len(flattened_builders) == 20
        assert "t" in sweep_flattener.sweep_names
        assert "a" in sweep_flattener.sweep_names
        assert sweep_flattener.sweep_sizes == [len(self.times), len(self.amps)]
        assert len(sweep_flattener.device_assigns) == 0

    def test_accessors(self, sweep_flattener):
        accessors = sweep_flattener.accessors
        assert len(accessors) == 2
        names = {inj.variable_name for inj in accessors}
        assert names == {"t", "a"}
        attributes = {inj.attribute for inj in accessors}
        assert attributes == {"width", "amp"}

    def test_flattened_builders_have_variables_injected(self, flattened_builders):
        for sweep_instance in flattened_builders:
            params = sweep_instance.variables
            builder = sweep_instance.builder
            assert len(params) == 2
            sweep_instrs = [
                instr for instr in builder.instructions if isinstance(instr, Sweep)
            ]
            assert len(sweep_instrs) == 0
            pulse_instrs = [
                instr for instr in builder.instructions if isinstance(instr, Pulse)
            ]
            assert len(pulse_instrs) == 2
            pulse = pulse_instrs[0]
            assert isinstance(pulse.width, float)
            assert np.isclose(pulse.width, params["t"])
            assert isinstance(pulse.amp, float)
            assert np.isclose(pulse.amp, params["a"])
            assert sweep_instance.device_assigns.assigns == []


class TestSweepFlattenerWithSweepOnMultipleInstructions:
    model: QuantumHardwareModel = EchoModelLoader().load()
    qubit: Qubit = model.get_qubit(0)
    times = np.linspace(80e-9, 800e-9, 5)

    @pytest.fixture(scope="class")
    def sweep_builder(self):
        return sweep_sequential_pulse_widths(
            self.model, qubit1=0, qubit2=1, times=self.times
        )

    @pytest.fixture(scope="class")
    def sweep_flattener(self, sweep_builder):
        return SweepFlattener(sweep_builder)

    @pytest.fixture(scope="class")
    def flattened_builders(self, sweep_flattener):
        return sweep_flattener.create_flattened_builders()

    def test_extract_sweeps(self, sweep_flattener, flattened_builders):
        assert len(sweep_flattener.sweeps) == 1
        assert len(flattened_builders) == 5
        assert "t" in sweep_flattener.sweep_names
        assert sweep_flattener.sweep_sizes == [len(self.times)]
        assert len(sweep_flattener.device_assigns) == 0

    def test_accessors(self, sweep_flattener):
        accessors = sweep_flattener.accessors
        assert len(accessors) == 2
        names = {inj.variable_name for inj in accessors}
        assert names == {"t"}
        attributes = {inj.attribute for inj in accessors}
        assert attributes == {"width", "time"}
        indices = {inj.instruction_index for inj in accessors}
        assert indices == {0, 1}

    def test_flattened_builders_have_variables_injected(self, flattened_builders):
        for sweep_instance in flattened_builders:
            params = sweep_instance.variables
            builder = sweep_instance.builder
            assert len(params) == 1
            sweep_instrs = [
                instr for instr in builder.instructions if isinstance(instr, Sweep)
            ]
            assert len(sweep_instrs) == 0
            pulse_instrs = [
                instr for instr in builder.instructions if isinstance(instr, Pulse)
            ]
            assert len(pulse_instrs) == 4
            pulse = pulse_instrs[0]
            assert isinstance(pulse.width, float)
            assert np.isclose(pulse.width, params["t"])
            delay_instrs = [
                instr for instr in builder.instructions if isinstance(instr, Delay)
            ]
            assert len(delay_instrs) == 1
            delay = delay_instrs[0]
            assert isinstance(delay.time, float)
            assert np.isclose(delay.time, params["t"])
            assert sweep_instance.device_assigns.assigns == []


class TestSweepFlattenerWithDeviceAssign:
    model: QuantumHardwareModel = EchoModelLoader().load()
    qubit: Qubit = model.get_qubit(0)
    drive_channel = qubit.get_drive_channel()
    scales = np.linspace(0.1, 1.0, 10)

    @pytest.fixture(scope="class")
    def sweep_builder(self):
        return sweep_pulse_scales(self.model, qubit=0, scales=self.scales)

    @pytest.fixture(scope="class")
    def sweep_flattener(self, sweep_builder):
        return SweepFlattener(sweep_builder)

    @pytest.fixture(scope="class")
    def flattened_builders(self, sweep_flattener):
        return sweep_flattener.create_flattened_builders()

    def test_extract_sweeps(self, sweep_flattener, flattened_builders):
        assert len(sweep_flattener.sweeps) == 1
        assert len(flattened_builders) == 10
        assert "s" in sweep_flattener.sweep_names
        assert len(sweep_flattener.device_assigns) == 1
        assert sweep_flattener.device_assigns[0].attribute == "scale"
        assert sweep_flattener.device_assigns[0].target == self.drive_channel

    def test_accessors(self, sweep_flattener):
        accessors = sweep_flattener.accessors
        assert len(accessors) == 0

    def test_flattened_builders_have_variables_injected(self, flattened_builders):
        for sweep_instance in flattened_builders:
            params = sweep_instance.variables
            assert sweep_instance.device_assigns.assigns[0].value == params["s"]

    def test_apply_device_assigns(self, flattened_builders):
        original_scale = self.drive_channel.scale
        for sweep_instance in flattened_builders:
            params = sweep_instance.variables
            device_assigns = sweep_instance.device_assigns
            with device_assigns.apply():
                channel = self.model.get_qubit(0).get_drive_channel()
                assert np.isclose(channel.scale, params["s"])
        channel = self.model.get_qubit(0).get_drive_channel()
        assert np.isclose(channel.scale, original_scale)


class TestSweepFlattenerWithZippedParameters:
    model: QuantumHardwareModel = EchoModelLoader().load()
    qubit: Qubit = model.get_qubit(0)
    times = np.linspace(80e-9, 800e-9, 10)

    @pytest.fixture(scope="class")
    def sweep_builder(self):
        return sweep_zipped_parameters(self.model, 0, self.times, 2 * self.times)

    @pytest.fixture(scope="class")
    def sweep_flattener(self, sweep_builder):
        return SweepFlattener(sweep_builder)

    @pytest.fixture(scope="class")
    def flattened_builders(self, sweep_flattener):
        return sweep_flattener.create_flattened_builders()

    def test_extract_sweeps(self, sweep_flattener, flattened_builders):
        assert len(sweep_flattener.sweeps) == 1
        assert len(flattened_builders) == len(self.times)
        assert "t1" in sweep_flattener.sweep_names
        assert "t2" in sweep_flattener.sweep_names
        assert sweep_flattener.sweep_sizes == [len(self.times)]
        assert len(sweep_flattener.device_assigns) == 0

    def test_accessors(self, sweep_flattener):
        accessors = sweep_flattener.accessors
        assert len(accessors) == 2
        names = {inj.variable_name for inj in accessors}
        assert names == {"t1", "t2"}
        attributes = {inj.attribute for inj in accessors}
        assert attributes == {"width", "time"}

    def test_flattened_builders_have_variables_injected(self, flattened_builders):
        for sweep_instance in flattened_builders:
            params = sweep_instance.variables
            builder = sweep_instance.builder
            assert len(params) == 2
            sweep_instrs = [
                instr for instr in builder.instructions if isinstance(instr, Sweep)
            ]
            assert len(sweep_instrs) == 0
            pulse_instrs = [
                instr for instr in builder.instructions if isinstance(instr, Pulse)
            ]
            assert len(pulse_instrs) == 2
            pulse = pulse_instrs[0]
            assert isinstance(pulse.width, float)
            assert np.isclose(pulse.width, params["t1"])
            delay_instrs = [
                instr for instr in builder.instructions if isinstance(instr, Delay)
            ]
            assert len(delay_instrs) == 1
            delay = delay_instrs[0]
            assert isinstance(delay.time, float)
            assert np.isclose(delay.time, params["t2"])
