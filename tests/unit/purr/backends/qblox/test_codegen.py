# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import re
from contextlib import nullcontext

import numpy as np
import pytest
from more_itertools import partition

from qat import get_config, qatconfig
from qat.purr.backends.qblox.analysis_passes import (
    BindingPass,
    QbloxLegalisationPass,
    TILegalisationPass,
    TriagePass,
    TriageResult,
)
from qat.purr.backends.qblox.codegen import NewQbloxEmitter, QbloxEmitter
from qat.purr.backends.qblox.constants import Constants
from qat.purr.backends.qblox.device import QbloxPhysicalBaseband, QbloxPhysicalChannel
from qat.purr.backends.qblox.transform_passes import (
    DesugaringPass,
    RepeatSanitisation,
    ReturnSanitisation,
    ScopeSanitisation,
)
from qat.purr.compiler.devices import PulseShapeType
from qat.purr.compiler.execution import DeviceInjectors
from qat.purr.compiler.instructions import (
    Acquire,
    DeviceUpdate,
    MeasurePulse,
    Pulse,
    SweepValue,
    Variable,
    calculate_duration,
)
from qat.purr.compiler.runtime import get_builder
from qat.purr.core.metrics_base import MetricsManager
from qat.purr.core.pass_base import InvokerMixin, PassManager
from qat.purr.core.result_base import ResultManager
from qat.purr.utils.logger import get_default_logger

from tests.unit.utils.builder_nuggets import (
    delay_iteration,
    discrimination,
    empty,
    measure_acquire,
    multi_readout,
    pulse_amplitude_iteration,
    pulse_width_iteration,
    qubit_spect,
    resonator_spect,
    time_and_phase_iteration,
)

log = get_default_logger()


@pytest.mark.parametrize("model", [None], indirect=True)
class TestQbloxEmitter(InvokerMixin):
    def build_pass_pipeline(self, *args, **kwargs):
        return (
            PassManager()
            | RepeatSanitisation(self.model)
            | ScopeSanitisation()
            | ReturnSanitisation()
            | DesugaringPass()
            | TriagePass()
            | BindingPass()
        )

    def _do_emit(self, builder, model, skip_runtime=False):
        qatconfig = get_config()
        old_value = qatconfig.INSTRUCTION_VALIDATION.PULSE_DURATION_LIMITS
        try:
            qatconfig.INSTRUCTION_VALIDATION.PULSE_DURATION_LIMITS = False
            res_mgr = ResultManager()
            met_mgr = MetricsManager()

            if not skip_runtime:
                runtime = model.create_runtime()
                runtime.run_pass_pipeline(builder, res_mgr, met_mgr)

            self.model = model
            self.run_pass_pipeline(builder, res_mgr, met_mgr)
            return QbloxEmitter().emit_packages(builder, res_mgr, met_mgr)
        finally:
            qatconfig.INSTRUCTION_VALIDATION.PULSE_DURATION_LIMITS = old_value

    def test_play_guassian(self, model):
        width = 100e-9
        rise = 1.0 / 5.0
        drive_channel = model.get_qubit(0).get_drive_channel()
        gaussian = Pulse(drive_channel, PulseShapeType.GAUSSIAN, width=width, rise=rise)
        builder = get_builder(model).add(gaussian)

        iter2packages = self._do_emit(builder, model)
        for packages in iter2packages.values():
            assert len(packages) == 1
            pkg = packages[0]
            assert pkg.pulse_channel_id == drive_channel.full_id()
            assert f"GAUSSIAN_{hash(gaussian)}_I" in pkg.sequence.waveforms
            assert f"GAUSSIAN_{hash(gaussian)}_Q" in pkg.sequence.waveforms
            assert "play 0,1,100" in pkg.sequence.program

    @pytest.mark.parametrize("start_width, end_width", [(0, 100e-9), (50e-9, 100e-9)])
    def test_play_square(self, model, start_width, end_width):
        amp = 1
        num_points = 50

        drive_channel = model.get_qubit(0).get_drive_channel()
        time, step = np.linspace(start_width, end_width, num_points, retstep=True)
        builder = get_builder(model)
        builder.sweep(SweepValue("t", time))
        builder.pulse(drive_channel, PulseShapeType.SQUARE, width=Variable("t"), amp=amp)
        iter2packages = self._do_emit(builder, model)
        assert len(iter2packages) == num_points

        ignored_indices = np.squeeze(np.where(time < Constants.GRID_TIME * 1e-9)) + 1
        for i in ignored_indices:
            assert not iter2packages[i]

        non_ignored_indices = np.squeeze(np.where(time >= Constants.GRID_TIME * 1e-9)) + 1
        for i in non_ignored_indices:
            assert len(iter2packages[i]) == 1
            pkg = iter2packages[i][0]
            assert pkg.pulse_channel_id == drive_channel.full_id()
            assert not pkg.sequence.waveforms
            assert f"set_awg_offs {Constants.MAX_OFFSET},0" in pkg.sequence.program
            assert "set_awg_offs 0,0" in pkg.sequence.program

    def test_phase_and_frequency_shift(self, model):
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
        iter2packages = self._do_emit(builder, model, skip_runtime=True)
        for packages in iter2packages.values():
            for package in packages:
                expected_phase = QbloxLegalisationPass.phase_as_steps(phase)
                assert f"set_ph_delta {expected_phase}" in package.sequence.program
                expected_freq = QbloxLegalisationPass.freq_as_steps(
                    drive_channel.baseband_if_frequency + frequency
                )
                assert f"set_freq {expected_freq}" in package.sequence.program

    def test_measure_acquire(self, model):
        qubit = model.get_qubit(0)
        acquire_channel = qubit.get_acquire_channel()
        measure_channel = qubit.get_measure_channel()
        assert measure_channel == acquire_channel

        time = 7.5e-6
        i_offs_steps = int(qubit.pulse_measure["amp"] * Constants.MAX_OFFSET)
        delay = qubit.measure_acquire["delay"]

        builder = get_builder(model)
        builder.acquire(acquire_channel, time, delay=delay)
        iter2packages = self._do_emit(builder, model)
        for packages in iter2packages.values():
            assert len(packages) == 0

        builder = get_builder(model)
        builder.add(
            [
                MeasurePulse(measure_channel, **qubit.pulse_measure),
                Acquire(acquire_channel, time=time, delay=delay),
            ]
        )
        iter2packages = self._do_emit(builder, model)
        remaining_width = int(qubit.pulse_measure["width"] * 1e9) - int(delay * 1e9)

        pattern = r"set_awg_offs {0},0\nupd_param {1}\nacquire 0,R\d{{1,2}},{2}\nset_awg_offs 0,0\nupd_param 4".format(
            i_offs_steps, int(delay * 1e9), remaining_width
        )
        for packages in iter2packages.values():
            assert len(packages) == 1
            pkg = packages[0]
            assert not pkg.sequence.waveforms
            assert qubit.pulse_measure["shape"] == PulseShapeType.SQUARE
            assert re.search(pattern, pkg.sequence.program, re.MULTILINE)

            channel = acquire_channel.physical_channel
            assert isinstance(channel, QbloxPhysicalChannel)
            assert isinstance(channel.baseband, QbloxPhysicalBaseband)
            assert len(channel.config.sequencers) > 0
            assert pkg.seq_config.square_weight_acq.integration_length == int(time * 1e9)

    @pytest.mark.parametrize(
        ("width_seconds", "context"),
        [
            (np.full(10, 5e-6), nullcontext()),
            (
                np.full(5, (Constants.MAX_SAMPLE_SIZE_WAVEFORMS / 2) * 1e-9),
                nullcontext(),
            ),  # 8.192e-6
            (
                np.full(1, Constants.MAX_SAMPLE_SIZE_WAVEFORMS * 1e-9),
                pytest.raises(ValueError),
            ),  # 16.384e-6
        ],
    )
    def test_waveform_caching(self, model, width_seconds, context):
        width_samples = np.astype(width_seconds * 1e9, int)
        assert 2 * np.sum(width_samples) > Constants.MAX_SAMPLE_SIZE_WAVEFORMS

        drive_channel = model.get_qubit(0).get_drive_channel()
        builder = get_builder(model)
        for val in width_seconds:
            builder.pulse(drive_channel, PulseShapeType.GAUSSIAN, width=val, rise=1.0 / 5.0)

        with context:
            iter2packages = self._do_emit(builder, model)
            for packages in iter2packages.values():
                assert len(packages) == 1

    def test_single_resonator_spect(self, model):
        index = 0
        builder = resonator_spect(model, [index])
        qubit = model.get_qubit(index)
        acquire_channel = qubit.get_acquire_channel()

        iter2packages = self._do_emit(builder, model)
        for packages in iter2packages.values():
            assert len(packages) == 1
            acquire_pkg = packages[0]
            assert acquire_pkg.pulse_channel_id == acquire_channel.full_id()

            measure_pulse = next(
                (inst for inst in builder.instructions if isinstance(inst, MeasurePulse))
            )
            assert acquire_channel in measure_pulse.quantum_targets

            assert acquire_pkg.sequence.acquisitions

            if measure_pulse.shape == PulseShapeType.SQUARE:
                assert not acquire_pkg.sequence.waveforms
                assert "play" not in acquire_pkg.sequence.program
                assert "set_awg_offs" in acquire_pkg.sequence.program
                assert "upd_param" in acquire_pkg.sequence.program
            else:
                assert acquire_pkg.sequence.waveforms
                assert "play" in acquire_pkg.sequence.program
                assert "set_awg_offs" not in acquire_pkg.sequence.program
                assert "upd_param" not in acquire_pkg.sequence.program

    def test_multi_resonator_spect(self, model):
        indices = [0, 1]
        builder = resonator_spect(model, indices)
        iter2packages = self._do_emit(builder, model)
        for packages in iter2packages.values():
            assert len(packages) == len(indices)

            for i, index in enumerate(indices):
                qubit = model.get_qubit(index)
                acquire_channel = qubit.get_acquire_channel()
                acquire_pkg = next(
                    (
                        pkg
                        for pkg in packages
                        if pkg.pulse_channel_id == acquire_channel.full_id()
                    )
                )
                assert acquire_pkg.sequence.acquisitions

                measure_pulse = next(
                    (
                        inst
                        for inst in builder.instructions
                        if isinstance(inst, MeasurePulse)
                        and acquire_channel in inst.quantum_targets
                    )
                )

                if measure_pulse.shape == PulseShapeType.SQUARE:
                    assert not acquire_pkg.sequence.waveforms
                    assert "play" not in acquire_pkg.sequence.program
                    assert "set_awg_offs" in acquire_pkg.sequence.program
                    assert "upd_param" in acquire_pkg.sequence.program
                else:
                    assert acquire_pkg.sequence.waveforms
                    assert "play" in acquire_pkg.sequence.program
                    assert "set_awg_offs" not in acquire_pkg.sequence.program
                    assert "upd_param" not in acquire_pkg.sequence.program

    def test_single_qubit_spect(self, model):
        qubit_index = 0
        builder = qubit_spect(model, [qubit_index])
        qubit = model.get_qubit(qubit_index)
        drive_channel = qubit.get_drive_channel()
        acquire_channel = qubit.get_acquire_channel()

        iter2packages = self._do_emit(builder, model)
        for packages in iter2packages.values():
            assert len(packages) == 2

            # Drive
            drive_pkg = next(
                (pkg for pkg in packages if pkg.pulse_channel_id == drive_channel.full_id())
            )
            drive_pulse = next(
                (inst for inst in builder.instructions if isinstance(inst, Pulse))
            )
            assert drive_channel in drive_pulse.quantum_targets

            assert not drive_pkg.sequence.acquisitions

            if drive_pulse.shape == PulseShapeType.SQUARE:
                assert not drive_pkg.sequence.waveforms
                assert "play" not in drive_pkg.sequence.program
                assert "set_awg_offs" in drive_pkg.sequence.program
                assert "upd_param" in drive_pkg.sequence.program
            else:
                assert drive_pkg.sequence.waveforms
                assert "play" in drive_pkg.sequence.program
                assert "set_awg_offs" not in drive_pkg.sequence.program
                assert "upd_param" not in drive_pkg.sequence.program

            # Readout
            acquire_pkg = next(
                (
                    pkg
                    for pkg in packages
                    if pkg.pulse_channel_id == acquire_channel.full_id()
                )
            )
            measure_pulse = next(
                (inst for inst in builder.instructions if isinstance(inst, MeasurePulse))
            )
            assert acquire_channel in measure_pulse.quantum_targets

            assert acquire_pkg.sequence.acquisitions

            if measure_pulse.shape == PulseShapeType.SQUARE:
                assert not acquire_pkg.sequence.waveforms
                assert "play" not in acquire_pkg.sequence.program
                assert "set_awg_offs" in acquire_pkg.sequence.program
                assert "upd_param" in acquire_pkg.sequence.program
            else:
                assert acquire_pkg.sequence.waveforms
                assert "play" in acquire_pkg.sequence.program
                assert "set_awg_offs" not in acquire_pkg.sequence.program
                assert "upd_param" not in acquire_pkg.sequence.program

    def test_multi_qubit_spect(self, model):
        indices = [0, 1]
        builder = qubit_spect(model, indices)
        iter2packages = self._do_emit(builder, model)
        for packages in iter2packages.values():
            assert len(packages) == 2 * len(indices)

            for i, index in enumerate(indices):
                qubit = model.get_qubit(index)

                # Drive
                drive_channel = qubit.get_drive_channel()
                drive_pkg = next(
                    (
                        pkg
                        for pkg in packages
                        if pkg.pulse_channel_id == drive_channel.full_id()
                    )
                )
                drive_pulse = next(
                    (
                        inst
                        for inst in builder.instructions
                        if isinstance(inst, Pulse) and drive_channel in inst.quantum_targets
                    )
                )
                if drive_pulse.shape == PulseShapeType.SQUARE:
                    assert not drive_pkg.sequence.waveforms
                    assert "play" not in drive_pkg.sequence.program
                    assert "set_awg_offs" in drive_pkg.sequence.program
                    assert "upd_param" in drive_pkg.sequence.program
                else:
                    assert drive_pkg.sequence.waveforms
                    assert "play" in drive_pkg.sequence.program
                    assert "set_awg_offs" not in drive_pkg.sequence.program
                    assert "upd_param" not in drive_pkg.sequence.program

                # Readout
                acquire_channel = qubit.get_acquire_channel()
                acquire_pkg = next(
                    (
                        pkg
                        for pkg in packages
                        if pkg.pulse_channel_id == acquire_channel.full_id()
                    )
                )
                assert acquire_pkg.sequence.acquisitions
                measure_pulse = next(
                    (
                        inst
                        for inst in builder.instructions
                        if isinstance(inst, MeasurePulse)
                        and acquire_channel in inst.quantum_targets
                    )
                )

                if measure_pulse.shape == PulseShapeType.SQUARE:
                    assert not acquire_pkg.sequence.waveforms
                    assert "play" not in acquire_pkg.sequence.program
                    assert "set_awg_offs" in acquire_pkg.sequence.program
                    assert "upd_param" in acquire_pkg.sequence.program
                else:
                    assert acquire_pkg.sequence.waveforms
                    assert "play" in acquire_pkg.sequence.program
                    assert "set_awg_offs" not in acquire_pkg.sequence.program
                    assert "upd_param" not in acquire_pkg.sequence.program

    @pytest.mark.parametrize("qubit_indices", [[0], [0, 1]])
    @pytest.mark.parametrize("enable_weights", [True, False])
    @pytest.mark.parametrize("do_X", [True, False])
    def test_scope_acquisition(self, model, qubit_indices, enable_weights, do_X):
        builder = measure_acquire(model, qubit_indices, do_X=do_X)
        iter2packages = self._do_emit(builder, model)
        for packages in iter2packages.values():
            if do_X:
                assert len(packages) == 2 * len(qubit_indices)
            else:
                assert len(packages) == len(qubit_indices)

        builder = measure_acquire(model, qubit_indices, enable_weights=enable_weights)
        iter2packages = self._do_emit(builder, model)
        for packages in iter2packages.values():
            for acquire_pkg in packages:
                if enable_weights:
                    assert "acquire_weighed" in acquire_pkg.sequence.program
                else:
                    assert "acquire" in acquire_pkg.sequence.program

    @pytest.mark.parametrize("qubit_indices", [[0]])
    @pytest.mark.parametrize("num_acquires", [1, 2, 3])
    def test_multi_readout(self, model, qubit_indices, num_acquires):
        builder = multi_readout(model, qubit_indices, do_X=False, num_acquires=num_acquires)
        iter2packages = self._do_emit(builder, model)
        for packages in iter2packages.values():
            assert len(packages) == len(qubit_indices)

            for index in qubit_indices:
                qubit = model.get_qubit(index)
                measure_channel = qubit.get_measure_channel()

                # Readout
                measure_pkg = next(
                    (
                        pkg
                        for pkg in packages
                        if pkg.pulse_channel_id == measure_channel.full_id()
                    )
                )
                assert measure_pkg.sequence.acquisitions
                assert measure_pkg.sequence.program.count("acquire") == 2 * num_acquires

        builder = multi_readout(model, qubit_indices, do_X=True, num_acquires=num_acquires)
        with pytest.raises(ValueError):
            self._do_emit(builder, model)

        old_value = qatconfig.INSTRUCTION_VALIDATION.NO_MID_CIRCUIT_MEASUREMENT
        try:
            qatconfig.INSTRUCTION_VALIDATION.NO_MID_CIRCUIT_MEASUREMENT = False
            iter2packages = self._do_emit(builder, model)
            for packages in iter2packages.values():
                assert len(packages) == 2 * len(qubit_indices)

                for index in qubit_indices:
                    qubit = model.get_qubit(index)
                    measure_channel = qubit.get_measure_channel()

                    # Readout
                    measure_pkg = next(
                        (
                            pkg
                            for pkg in packages
                            if pkg.pulse_channel_id == measure_channel.full_id()
                        )
                    )
                    assert measure_pkg.sequence.acquisitions
                    assert measure_pkg.sequence.program.count("acquire") == 2 * num_acquires
        finally:
            qatconfig.INSTRUCTION_VALIDATION.NO_MID_CIRCUIT_MEASUREMENT = old_value

    @pytest.mark.parametrize("pulse_width", [0, Constants.GRID_TIME, 1e3, 1e6])
    @pytest.mark.parametrize(
        "delay_width", [0, Constants.GRID_TIME, 100, 1e3 - Constants.GRID_TIME, 1e3]
    )
    def test_measure_acquire_operands(self, model, pulse_width, delay_width):
        qubit_indices = [0]
        for index in qubit_indices:
            qubit = model.get_qubit(index)
            qubit.pulse_measure["width"] = pulse_width * 1e-9
            qubit.measure_acquire["delay"] = delay_width * 1e-9

        builder = measure_acquire(model, qubit_indices)

        effective_width = max(min(pulse_width, delay_width), Constants.GRID_TIME)
        if 0 < pulse_width < effective_width + Constants.GRID_TIME:
            with pytest.raises(ValueError):
                self._do_emit(builder, model)

        else:
            iter2packages = self._do_emit(builder, model)
            for packages in iter2packages.values():
                if pulse_width < Constants.GRID_TIME:
                    assert len(packages) == 0
                else:
                    assert len(packages) == len(qubit_indices)
                    for pkg in packages:
                        program = pkg.sequence.program
                        quotient = effective_width // Constants.MAX_WAIT_TIME
                        remainder = effective_width % Constants.MAX_WAIT_TIME
                        if quotient > 1:
                            assert f"wait {remainder}" in program

    @pytest.mark.parametrize("qubit_indices", [[0], [0, 1]])
    def test_discrimination(self, model, qubit_indices):
        qubits = [model.get_qubit(index) for index in qubit_indices]

        builder = discrimination(model, qubit_indices)
        acquire = next(inst for inst in builder.instructions if isinstance(inst, Acquire))
        acq_width = int(calculate_duration(acquire))

        iter2packages = self._do_emit(builder, model)
        for packages in iter2packages.values():
            assert len(packages) == len(qubits)
            qub_pkg_zip = [
                x
                for x in zip(qubits, packages)
                if x[0].get_measure_channel().full_id() == x[1].pulse_channel_id
            ]

            for qubit, measure_pkg in qub_pkg_zip:
                A, B = qubit.mean_z_map_args[0], qubit.mean_z_map_args[1]
                mean_g = (1 - B) / A
                mean_e = (-1 - B) / A

                rotation = np.mod(-np.angle(mean_e - mean_g), 2 * np.pi)
                threshold = (np.exp(1j * rotation) * (mean_e + mean_g)).real / 2

                assert measure_pkg.seq_config.thresholded_acq.rotation == np.rad2deg(
                    rotation
                )
                assert (
                    measure_pkg.seq_config.thresholded_acq.threshold
                    == acq_width * threshold
                )


@pytest.mark.parametrize("model", [None], indirect=True)
class TestNewQbloxEmitter(InvokerMixin):
    def build_pass_pipeline(self, *args, **kwargs):
        return (
            PassManager()
            | RepeatSanitisation(self.model)
            | ScopeSanitisation()
            | ReturnSanitisation()
            | DesugaringPass()
            | TriagePass()
            | BindingPass()
            | TILegalisationPass()
            | QbloxLegalisationPass()
        )

    def test_prologue_epilogue(self, model):
        builder = empty(model)
        res_mgr = ResultManager()
        met_mgr = MetricsManager()

        self.model = model
        self.run_pass_pipeline(builder, res_mgr, met_mgr)
        assert len(builder.instructions) == 6

        packages = NewQbloxEmitter().emit_packages(
            builder, res_mgr, met_mgr, ignore_empty=False
        )
        assert len(packages) == 2

        for pkg in packages:
            assert pkg.timeline.size == 4
            assert not pkg.sequence.waveforms
            assert not pkg.sequence.acquisitions
            assert not pkg.sequence.weights

            assert "set_mrk 3\nset_latch_en 1,4\nupd_param 4" in pkg.sequence.program
            assert "stop" in pkg.sequence.program

    @pytest.mark.parametrize("num_points", [1, 10, 100])
    @pytest.mark.parametrize("qubit_indices", [[0], [0, 1]])
    def test_resonator_spect(self, model, num_points, qubit_indices):
        builder = resonator_spect(model, qubit_indices, num_points)
        res_mgr = ResultManager()
        met_mgr = MetricsManager()
        runtime = model.create_runtime()
        runtime.run_pass_pipeline(builder, res_mgr, met_mgr)

        self.model = model
        self.run_pass_pipeline(builder, res_mgr, met_mgr)
        triage_result = res_mgr.lookup_by_type(TriageResult)

        packages = NewQbloxEmitter().emit_packages(builder, res_mgr, met_mgr)
        assert len(packages) == len(qubit_indices)

        for index in qubit_indices:
            qubit = model.get_qubit(index)
            acquire_channel = qubit.get_acquire_channel()
            acquire_pkg = next(
                (
                    pkg
                    for pkg in packages
                    if pkg.pulse_channel_id == acquire_channel.full_id()
                )
            )
            measure_pulse = next(
                (
                    inst
                    for inst in builder.instructions
                    if isinstance(inst, MeasurePulse)
                    and acquire_channel in inst.quantum_targets
                )
            )

            assert acquire_pkg.sequence.acquisitions
            for sweep in triage_result.sweeps:
                assert f"{hash(sweep)}_0" in acquire_pkg.sequence.program

            assert "wait_sync" in acquire_pkg.sequence.program

            if measure_pulse.shape == PulseShapeType.SQUARE:
                assert not acquire_pkg.sequence.waveforms
                assert "play" not in acquire_pkg.sequence.program
                assert "set_awg_offs" in acquire_pkg.sequence.program
                assert "upd_param" in acquire_pkg.sequence.program
            else:
                assert acquire_pkg.sequence.waveforms
                assert "play" in acquire_pkg.sequence.program
                assert "set_awg_offs" not in acquire_pkg.sequence.program
                assert "upd_param" not in acquire_pkg.sequence.program

    @pytest.mark.parametrize("num_points", [1, 10, 100])
    @pytest.mark.parametrize("qubit_indices", [[0], [0, 1]])
    def test_qubit_spect(self, model, num_points, qubit_indices):
        builder = qubit_spect(model, qubit_indices, num_points)
        res_mgr = ResultManager()
        met_mgr = MetricsManager()
        self.model = model
        model.create_engine()
        runtime = model.create_runtime()
        runtime.run_pass_pipeline(builder, res_mgr, met_mgr)

        # TODO - A skeptical usage of DeviceInjectors on static device updates
        # TODO - Figure out what they mean w/r to scopes and control flow
        remaining, static_dus = partition(
            lambda inst: isinstance(inst, DeviceUpdate)
            and not isinstance(inst.value, Variable),
            builder.instructions,
        )
        remaining, static_dus = list(remaining), list(static_dus)

        assert len(static_dus) == len(qubit_indices)

        injectors = DeviceInjectors(static_dus)
        try:
            injectors.inject()
            self.run_pass_pipeline(builder, res_mgr, met_mgr)

            packages = NewQbloxEmitter().emit_packages(builder, res_mgr, met_mgr)
            assert len(packages) == 2 * len(qubit_indices)

            for index in qubit_indices:
                qubit = model.get_qubit(index)
                drive_channel = qubit.get_drive_channel()
                acquire_channel = qubit.get_acquire_channel()

                # Drive
                drive_pkg = next(
                    (
                        pkg
                        for pkg in packages
                        if pkg.pulse_channel_id == drive_channel.full_id()
                    )
                )
                drive_pulse = next(
                    (
                        inst
                        for inst in builder.instructions
                        if isinstance(inst, Pulse) and drive_channel in inst.quantum_targets
                    )
                )

                assert not drive_pkg.sequence.acquisitions

                assert "wait_sync" in drive_pkg.sequence.program

                if drive_pulse.shape == PulseShapeType.SQUARE:
                    assert not drive_pkg.sequence.waveforms
                    assert "play" not in drive_pkg.sequence.program
                    assert "set_awg_offs" in drive_pkg.sequence.program
                    assert "upd_param" in drive_pkg.sequence.program
                else:
                    assert drive_pkg.sequence.waveforms
                    assert "play" in drive_pkg.sequence.program
                    assert "set_awg_offs" not in drive_pkg.sequence.program
                    assert "upd_param" not in drive_pkg.sequence.program

                # Readout
                acquire_pkg = next(
                    (
                        pkg
                        for pkg in packages
                        if pkg.pulse_channel_id == acquire_channel.full_id()
                    )
                )
                measure_pulse = next(
                    (
                        inst
                        for inst in builder.instructions
                        if isinstance(inst, MeasurePulse)
                        and acquire_channel in inst.quantum_targets
                    )
                )

                assert acquire_pkg.sequence.acquisitions

                assert "wait_sync" in acquire_pkg.sequence.program

                if measure_pulse.shape == PulseShapeType.SQUARE:
                    assert not acquire_pkg.sequence.waveforms
                    assert "play" not in acquire_pkg.sequence.program
                    assert "set_awg_offs" in acquire_pkg.sequence.program
                    assert "upd_param" in acquire_pkg.sequence.program
                else:
                    assert acquire_pkg.sequence.waveforms
                    assert "play" in acquire_pkg.sequence.program
                    assert "set_awg_offs" not in acquire_pkg.sequence.program
                    assert "upd_param" not in acquire_pkg.sequence.program
        finally:
            injectors.revert()

    @pytest.mark.parametrize("num_points", [100])
    @pytest.mark.parametrize("qubit_indices", [[0]])
    def test_delay_iteration(self, model, num_points, qubit_indices):
        builder = delay_iteration(model, qubit_indices, num_points)
        res_mgr = ResultManager()
        met_mgr = MetricsManager()
        runtime = model.create_runtime()
        runtime.run_pass_pipeline(builder, res_mgr, met_mgr)

        self.model = model
        self.run_pass_pipeline(builder, res_mgr, met_mgr)
        packages = NewQbloxEmitter().emit_packages(builder, res_mgr, model)
        assert len(packages) == 2 * len(qubit_indices)

    @pytest.mark.parametrize("num_points", [10])
    @pytest.mark.parametrize("qubit_indices", [[0]])
    def test_pulse_width_iteration(self, model, num_points, qubit_indices):
        builder = pulse_width_iteration(model, qubit_indices, num_points)
        res_mgr = ResultManager()
        met_mgr = MetricsManager()
        runtime = model.create_runtime()
        runtime.run_pass_pipeline(builder, res_mgr, met_mgr)

        self.model = model
        self.run_pass_pipeline(builder, res_mgr, met_mgr)
        packages = NewQbloxEmitter().emit_packages(builder, res_mgr, model)
        assert len(packages) == 2 * len(qubit_indices)

    @pytest.mark.parametrize("num_points", [10])
    @pytest.mark.parametrize("qubit_indices", [[0]])
    def test_pulse_amplitude_iteration(self, model, num_points, qubit_indices):
        builder = pulse_amplitude_iteration(model, qubit_indices, num_points)
        res_mgr = ResultManager()
        met_mgr = MetricsManager()
        runtime = model.create_runtime()
        runtime.run_pass_pipeline(builder, res_mgr, met_mgr)

        self.model = model
        self.run_pass_pipeline(builder, res_mgr, met_mgr)
        packages = NewQbloxEmitter().emit_packages(builder, res_mgr, model)
        assert len(packages) == 2 * len(qubit_indices)

    @pytest.mark.parametrize("num_points", [10])
    @pytest.mark.parametrize("qubit_indices", [[0]])
    def test_time_and_phase_iteration(self, model, num_points, qubit_indices):
        builder = time_and_phase_iteration(model, qubit_indices, num_points)
        res_mgr = ResultManager()
        met_mgr = MetricsManager()
        runtime = model.create_runtime()
        runtime.run_pass_pipeline(builder, res_mgr, met_mgr)

        self.model = model
        self.run_pass_pipeline(builder, res_mgr, met_mgr)
        packages = NewQbloxEmitter().emit_packages(builder, res_mgr, model)
        assert len(packages) == 2 * len(qubit_indices)
