# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
import pytest
from pydantic import ValidationError

from qat.ir.measure import (
    Acquire,
    AcquireMode,
    MeasureBlock,
    PostProcessing,
    PostProcessType,
)
from qat.ir.waveforms import Pulse, PulseShapeType, Waveform
from qat.purr.backends.echo import get_default_echo_hardware


class TestAcquire:
    model = get_default_echo_hardware()

    def test_initiate(self):
        chan = self.model.get_qubit(0).get_acquire_channel()
        inst = Acquire(chan)
        assert inst.time == 1e-6
        assert inst.targets == chan.full_id()

    def test_filter(self):
        chan = self.model.get_qubit(0).get_acquire_channel()
        filter = Pulse(
            targets=chan, waveform=Waveform(shape=PulseShapeType.GAUSSIAN, width=1e-6)
        )
        inst = Acquire(chan, time=1e-6, filter=filter)
        assert inst.filter == filter

    @pytest.mark.parametrize("time", [0, 5e-7, 1.01e-6, 2e-6])
    def test_filter_validation(self, time):
        chan = self.model.get_qubit(0).get_acquire_channel()
        filter = Pulse(chan, waveform=Waveform(shape=PulseShapeType.GAUSSIAN, width=time))
        with pytest.raises(ValidationError):
            inst = Acquire(chan, time=1e-6, filter=filter)


class TestPostProcessing:
    model = get_default_echo_hardware()

    @pytest.mark.parametrize("pp", PostProcessType)
    def test_initiate(self, pp):
        chan = self.model.get_qubit(0).get_acquire_channel()
        acquire = Acquire(chan)
        inst = PostProcessing(output_variable=acquire.output_variable, process=pp)
        assert inst.process == pp
        assert inst.output_variable == acquire.output_variable


class TestMeasureBlock:
    # Tests are extracted and modified from test_instructions.py

    @pytest.mark.parametrize("mode", list(AcquireMode))
    @pytest.mark.parametrize("num_qubits", [1, 3])
    def test_create_simple_measure_block(self, num_qubits, mode):
        hw = get_default_echo_hardware()
        targets = hw.qubits[:num_qubits]

        mb = MeasureBlock.create_block(targets, mode)
        assert isinstance(mb, MeasureBlock)
        assert mb.targets == [t.full_id() for t in targets]
        assert mb.target_dict[targets[0].full_id()].mode == mode

    @pytest.mark.parametrize("out_vars", [None, "c"])
    @pytest.mark.parametrize("num_qubits", [1, 3])
    def test_create_measure_block_with_output_variables(self, num_qubits, out_vars):
        hw = get_default_echo_hardware()
        targets = hw.qubits[:num_qubits]

        if isinstance(out_vars, str):
            out_vars = [f"{out_vars}[{i}]" for i in range(num_qubits)]

        mb = MeasureBlock.create_block(
            targets, AcquireMode.INTEGRATOR, output_variables=out_vars
        )
        expected = out_vars or [None] * num_qubits
        assert [val.output_variable for val in mb.target_dict.values()] == expected

    def test_add_to_measure_block(self):
        hw = get_default_echo_hardware()
        targets = [hw.qubits[0], hw.qubits[-1]]
        modes = [AcquireMode.INTEGRATOR, AcquireMode.SCOPE]
        out_vars = ["c[0]", "b[1]"]
        mb = MeasureBlock.create_block(
            targets[0],
            modes[0],
            output_variables=out_vars[:1],
        )
        assert mb.targets == [t.full_id() for t in hw.qubits[:1]]
        mb.add_measurements(targets[1], modes[1], output_variables=out_vars[1])
        assert mb.targets == [t.full_id() for t in targets]
        assert [val.mode for val in mb.target_dict.values()] == modes
        assert [val.output_variable for val in mb.target_dict.values()] == out_vars

    def test_cannot_add_duplicate_to_measure_block(self):
        hw = get_default_echo_hardware()
        targets = [hw.qubits[0], hw.qubits[-1]]
        out_vars = ["c[0]", "b[1]"]
        mb = MeasureBlock.create_block(
            targets,
            AcquireMode.INTEGRATOR,
            output_variables=out_vars,
        )
        assert mb.targets == [t.full_id() for t in targets]
        with pytest.raises(ValueError):
            mb.add_measurements(targets[1], AcquireMode.INTEGRATOR)

    def test_measure_block_duration(self):
        hw = get_default_echo_hardware()
        target = hw.qubits[0]
        mb = MeasureBlock.create_block([], AcquireMode.RAW)
        assert mb.duration == 0.0
        mb.add_measurements(target, AcquireMode.INTEGRATOR)
        acq = mb.get_acquires(target)[0]
        assert mb.duration > 0
        assert mb.duration == pytest.approx(acq.delay + acq.duration)
        mb.duration = 1
        assert mb.duration == 1

    def test_get_acquires(self):
        hw = get_default_echo_hardware()
        mb = MeasureBlock.create_block(hw.qubits, AcquireMode.INTEGRATOR)
        acquires = mb.get_acquires(hw.qubits)
        assert all([isinstance(acq, Acquire) for acq in acquires])
