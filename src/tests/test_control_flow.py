import numpy as np
import pytest

from qat.purr.backends.qblox.fast.live import FastQbloxLiveEngine
from qat.purr.compiler.control_flow.instructions import EndRepeat, EndSweep
from qat.purr.compiler.builders import QuantumInstructionBuilder
from qat.purr.compiler.devices import PulseShapeType
from qat.purr.compiler.instructions import SweepValue, Variable, Repeat, Sweep
from qat.purr.compiler.runtime import get_builder
from src.tests.qblox.builder_nuggets import resonator_spect


class TestInstructionBuilder(QuantumInstructionBuilder):
    def end_repeat(self):
        return self.add(EndRepeat())

    def end_sweep(self):
        return self.add(EndSweep())


# TODO - Test for more complex builders
def _arbitrary_builders(model):
    qubit = model.get_qubit(0)
    drive_channel = qubit.get_drive_channel()
    measure_channel = qubit.get_measure_channel()
    value = np.linspace(1, 10, 10)
    return [
        (
            TestInstructionBuilder(model)
            .sweep(SweepValue(f"var_name", value))
            .device_assign(measure_channel, "frequency", Variable(f"var_name"))
            .pulse(drive_channel, PulseShapeType.SQUARE, width=100e-9, amp=0.5)
            .measure_mean_signal(qubit, output_variable=f"Q{0}")
        ),
        (
            TestInstructionBuilder(model)
            .sweep(SweepValue(f"var_name", value))
            .device_assign(measure_channel, "frequency", Variable(f"var_name"))
            .pulse(drive_channel, PulseShapeType.SQUARE, width=100e-9, amp=0.5)
            .measure_mean_signal(qubit, output_variable=f"Q{0}")
            .repeat(1000, 100e-6)
        ),
        (
            TestInstructionBuilder(model)
            .repeat(400, 500e-6)
            .device_assign(measure_channel, "frequency", Variable(f"var_name"))
            .pulse(drive_channel, PulseShapeType.SQUARE, width=100e-9, amp=0.5)
            .measure_mean_signal(qubit, output_variable=f"Q{0}")
            .sweep(SweepValue(f"var_name", value))
        ),
        (
            TestInstructionBuilder(model)
            .repeat(400, 500e-6)
            .device_assign(measure_channel, "frequency", Variable(f"var_name"))
            .pulse(drive_channel, PulseShapeType.SQUARE, width=100e-9, amp=0.5)
            .measure_mean_signal(qubit, output_variable=f"Q{0}")
            .sweep(SweepValue(f"var_name", value))
        ),
    ]


@pytest.mark.parametrize("model", [None], indirect=True)
def test_multi_dim_sweep_rewrite(model):
    start, stop, num = 1, 10, 10
    var_value = np.linspace(start, stop, num)
    builder = get_builder(model)
    engine = FastQbloxLiveEngine(model)
    builder.sweep([SweepValue(f"i", var_value), SweepValue(f"j", var_value)])
    instructions = engine.optimize(builder.instructions)
    sweeps = []
    end_sweeps = []
    for inst in instructions:
        if isinstance(inst, Sweep):
            sweeps.append(inst)
        elif isinstance(inst, EndSweep):
            end_sweeps.append(inst)

    assert len(sweeps) == len(end_sweeps) == 2
    assert len(instructions) == len(sweeps) + len(end_sweeps)


@pytest.mark.parametrize("model", [None], indirect=True)
def test_instruction_rewrite(model):
    builder = resonator_spect(model)
    engine = FastQbloxLiveEngine(model)
    instructions = engine.optimize(builder.instructions)
    sweeps = []
    end_sweeps = []
    repeats = []
    end_repeats = []
    for inst in instructions:
        if isinstance(inst, Sweep):
            sweeps.append(inst)
        elif isinstance(inst, EndSweep):
            end_sweeps.append(inst)
        elif isinstance(inst, Repeat):
            repeats.append(inst)
        elif isinstance(inst, EndRepeat):
            end_repeats.append(inst)
    assert len(sweeps) == len(end_sweeps)
    assert len(repeats) == len(end_repeats)


@pytest.mark.parametrize("model", [None], indirect=True)
def test_validate_before_rewrite_fails(model):
    engine = FastQbloxLiveEngine(model)
    builders = _arbitrary_builders(model)
    for builder in builders:
        with pytest.raises(ValueError):
            engine.validate(builder.instructions)


@pytest.mark.parametrize("model", [None], indirect=True)
def test_validate_after_rewrite_succeeds(model):
    engine = FastQbloxLiveEngine(model)
    builders = _arbitrary_builders(model)
    for builder in builders:
        try:
            instructions = engine.optimize(builder.instructions)
            engine.validate(instructions)
        except ValueError:
            pytest.fail("Validation test failed")
