# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
import numpy as np
import pytest

from qat.purr.backends.qblox.analysis_passes import TriagePass
from qat.purr.backends.qblox.codegen import QbloxEmitter
from qat.purr.backends.qblox.transform_passes import RepeatSanitisation, ReturnSanitisation
from qat.purr.compiler.devices import PulseShapeType
from qat.purr.compiler.instructions import SweepValue, Variable
from qat.purr.compiler.runtime import execute_instructions, get_builder
from qat.purr.core.metrics_base import MetricsManager
from qat.purr.core.pass_base import InvokerMixin, PassManager
from qat.purr.core.result_base import ResultManager
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


@pytest.mark.parametrize("model", [None], indirect=True)
class TestQbloxControlHardware(InvokerMixin):
    def build_pass_pipeline(self, *args, **kwargs):
        return (
            PassManager()
            | RepeatSanitisation(self.model)
            | ReturnSanitisation()
            | TriagePass()
        )

    def _do_emit(self, builder, skip_runtime=False):
        res_mgr = ResultManager()
        met_mgr = MetricsManager()

        if not skip_runtime:
            runtime = self.model.create_runtime()
            runtime.run_pass_pipeline(builder, res_mgr, met_mgr)

        self.run_pass_pipeline(builder, res_mgr, met_mgr)
        return QbloxEmitter().emit_packages(builder, res_mgr, met_mgr)

    def test_instruction_execution(self, model):
        amp = 1
        rise = 1.0 / 3.0
        phase = 0.72
        frequency = 500

        drive_channel = model.get_qubit(0).get_drive_channel()
        builder = (
            get_builder(model)
            .pulse(drive_channel, PulseShapeType.SQUARE, width=100e-9, amp=amp)
            .pulse(drive_channel, PulseShapeType.GAUSSIAN, width=150e-9, rise=rise)
            .phase_shift(drive_channel, phase)
            .frequency_shift(drive_channel, frequency)
        )

        engine = model.create_engine()
        results, _ = execute_instructions(engine, builder)
        assert results is not None

    def test_resource_allocation(self, model):
        qubit = model.get_qubit(0)
        drive_channel = qubit.get_drive_channel()

        num_points = 10
        freq_range = 10e6
        freqs = drive_channel.frequency + np.linspace(-freq_range, freq_range, num_points)
        builder = (
            get_builder(model)
            .synchronize(qubit)
            .device_assign(drive_channel, "scale", 1)
            .sweep(SweepValue(f"freq{0}", freqs))
            .device_assign(drive_channel, "frequency", Variable(f"freq{0}"))
            .pulse(
                drive_channel,
                PulseShapeType.SQUARE,
                width=5e-6,
                amp=1,
                phase=0.0,
                drag=0.0,
                rise=1.0 / 3.0,
            )
            .synchronize(qubit)
            .measure_mean_signal(qubit, output_variable=f"Q{0}")
        )

        self.model = model
        iter2packages = self._do_emit(builder, model)
        for packages in iter2packages.values():
            model.control_hardware.set_data(packages)
            assert any(model.control_hardware.id2seq)
            model.control_hardware.start_playback(None, None)
            assert not any(model.control_hardware.id2seq)
