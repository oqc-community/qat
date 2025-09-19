# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import numpy as np

from qat.model.loaders.purr.echo import EchoModelLoader
from qat.pipelines.purr.sweep.full import FullSweepPipeline
from qat.pipelines.purr.waveform_v1 import EchoPipeline

from .utils import (
    sweep_pulse_scales,
    sweep_pulse_widths,
    sweep_pulse_widths_and_amps,
    sweep_sequential_pulse_widths,
    sweep_zipped_parameters,
)


class TestFullSweepPipeline:
    model = EchoModelLoader(qubit_count=4).load()
    base_pipeline = EchoPipeline(config=dict(name="test"), model=model)
    pipeline = FullSweepPipeline(base_pipeline=base_pipeline)

    def test_single_sweep(self):
        times = np.linspace(80e-9, 800e-9, 10)
        builder = sweep_pulse_widths(self.model, qubit=0, times=times)
        results, _ = self.pipeline.run(builder)
        assert "Q0" in results
        assert len(results["Q0"]) == len(times)
        for result in results["Q0"]:
            assert len(result) == 1000

    def test_single_sweep_with_device_assigns(self):
        original_scale = self.model.get_qubit(0).get_drive_channel().scale
        scales = np.linspace(0.5, 1.5, 10)
        builder = sweep_pulse_scales(self.model, qubit=0, scales=scales)
        results, _ = self.pipeline.run(builder)
        assert "Q0" in results
        assert len(results["Q0"]) == len(scales)
        for result in results["Q0"]:
            assert len(result) == 1000
        assert self.model.get_qubit(0).get_drive_channel().scale == original_scale

    def test_multiple_sweeps(self):
        times = np.linspace(80e-9, 800e-9, 5)
        amps = np.linspace(0.5, 1.5, 5)
        builder = sweep_pulse_widths_and_amps(self.model, qubit=0, times=times, amps=amps)
        results, _ = self.pipeline.run(builder)
        assert "Q0" in results
        assert len(results["Q0"]) == len(times)
        for times_result in results["Q0"]:
            assert len(times_result) == len(amps)
            for result in times_result:
                assert len(result) == 1000

    def test_one_sweep_on_multiple_instructions(self):
        times = np.linspace(80e-9, 800e-9, 10)
        builder = sweep_sequential_pulse_widths(self.model, qubit1=0, qubit2=1, times=times)
        results, _ = self.pipeline.run(builder)
        assert set(results.keys()) == {"Q0", "Q1"}

        for qubit_result in (results["Q0"], results["Q1"]):
            assert len(qubit_result) == len(times)
            for result in qubit_result:
                assert len(result) == 1000

    def test_zipped_sweeps(self):
        times = np.linspace(80e-9, 800e-9, 10)
        builder = sweep_zipped_parameters(self.model, qubit=0, times1=times, times2=times)
        results, _ = self.pipeline.run(builder)
        assert "Q0" in results
        assert len(results["Q0"]) == len(times)
        for result in results["Q0"]:
            assert len(result) == 1000
