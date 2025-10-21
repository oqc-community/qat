# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from contextlib import nullcontext

import numpy as np
import pytest
from more_itertools import partition

from qat import qatconfig
from qat.backend.passes.purr.analysis import (
    BindingPass,
    TILegalisationPass,
    TriagePass,
)
from qat.backend.passes.purr.transform import DesugaringPass
from qat.backend.qblox.codegen import QbloxBackend1, QbloxBackend2
from qat.backend.qblox.config.constants import Constants, QbloxTargetData
from qat.backend.qblox.passes.analysis import QbloxLegalisationPass
from qat.core.metrics_base import MetricsManager
from qat.core.pass_base import PassManager
from qat.core.result_base import ResultManager
from qat.middleend.passes.purr.transform import (
    DeviceUpdateSanitisation,
    PhaseOptimisation,
    PostProcessingSanitisation,
    RepeatSanitisation,
    ReturnSanitisation,
    ScopeSanitisation,
)
from qat.middleend.passes.purr.validation import InstructionValidation, ReadoutValidation
from qat.purr.backends.qblox.device import QbloxPhysicalBaseband, QbloxPhysicalChannel
from qat.purr.compiler.devices import PulseShapeType
from qat.purr.compiler.execution import DeviceInjectors
from qat.purr.compiler.instructions import (
    Acquire,
    CustomPulse,
    DeviceUpdate,
    MeasurePulse,
    Pulse,
    Sweep,
    SweepValue,
    Variable,
    calculate_duration,
)
from qat.purr.compiler.runtime import get_builder
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


@pytest.mark.parametrize("qblox_model", [None], indirect=True)
class TestQbloxBackend1:
    def middleend_pipeline(self, qblox_model):
        target_data = QbloxTargetData.default()

        return (
            PassManager()
            | PhaseOptimisation()
            | PostProcessingSanitisation()
            | DeviceUpdateSanitisation()
            | InstructionValidation(target_data)
            | ReadoutValidation(qblox_model)
            | RepeatSanitisation(qblox_model, target_data)
            | ReturnSanitisation()
            | TriagePass()
        )

    def _do_emit(self, builder, qblox_model, ignore_empty=True):
        res_mgr = ResultManager()
        met_mgr = MetricsManager()
        self.middleend_pipeline(qblox_model).run(builder, res_mgr, met_mgr)
        return QbloxBackend1(qblox_model).emit(builder, res_mgr, met_mgr, ignore_empty)

    def test_very_long_pulse(self, qblox_model):
        q0 = qblox_model.get_qubit(0)
        drive_channel = q0.get_drive_channel()
        builder = get_builder(qblox_model).pulse(
            drive_channel, PulseShapeType.SOFT_SQUARE, amp=0.1, width=1e-5, rise=1e-8
        )

        with pytest.raises(ValueError):
            self._do_emit(builder, qblox_model)

    def test_play_guassian(self, qblox_model):
        width = 100e-9
        rise = 1.0 / 5.0
        drive_channel = qblox_model.get_qubit(0).get_drive_channel()
        gaussian = Pulse(drive_channel, PulseShapeType.GAUSSIAN, width=width, rise=rise)
        builder = qblox_model.create_builder().add(gaussian)

        ordered_executables = self._do_emit(builder, qblox_model)
        for executable in ordered_executables.values():
            assert len(executable.packages) == 1
            pulse_channel_id, pkg = next(iter(executable.packages.items()))
            assert pulse_channel_id == drive_channel.full_id()
            assert f"GAUSSIAN_{hash(gaussian)}_I" in pkg.sequence.waveforms
            assert f"GAUSSIAN_{hash(gaussian)}_Q" in pkg.sequence.waveforms
            assert "play 0,1,100" in pkg.sequence.program

    @pytest.mark.parametrize("start_width, end_width", [(0, 100e-9), (50e-9, 100e-9)])
    def test_play_square(self, qblox_model, start_width, end_width):
        amp = 1
        num_points = 50

        drive_channel = qblox_model.get_qubit(0).get_drive_channel()
        time, step = np.linspace(start_width, end_width, num_points, retstep=True)
        builder = get_builder(qblox_model)
        builder.sweep(SweepValue("t", time))
        builder.pulse(drive_channel, PulseShapeType.SQUARE, width=Variable("t"), amp=amp)
        ordered_executables = self._do_emit(builder, qblox_model)
        assert len(ordered_executables) == num_points

        ignored_indices = np.squeeze(np.where(time < Constants.GRID_TIME * 1e-9)) + 1
        for i in ignored_indices:
            assert not ordered_executables[i].packages

        non_ignored_indices = np.squeeze(np.where(time >= Constants.GRID_TIME * 1e-9)) + 1
        for i in non_ignored_indices:
            assert len(ordered_executables[i].packages) == 1
            pkg = ordered_executables[i].packages[drive_channel.full_id()]
            assert not pkg.sequence.waveforms
            assert f"set_awg_offs {Constants.MAX_OFFSET},0" in pkg.sequence.program
            assert "set_awg_offs 0,0" in pkg.sequence.program

    def test_phase_and_frequency_shift(self, qblox_model):
        amp = 1
        rise = 1.0 / 3.0
        phase = 0.72
        frequency = 500

        drive_channel = qblox_model.get_qubit(0).get_drive_channel()
        builder = (
            qblox_model.create_builder()
            .pulse(drive_channel, PulseShapeType.SQUARE, width=100e-9, amp=amp)
            .pulse(drive_channel, PulseShapeType.GAUSSIAN, width=150e-9, rise=rise)
            .phase_shift(drive_channel, phase)
            .frequency_shift(drive_channel, frequency)
            .pulse(drive_channel, PulseShapeType.SQUARE, width=100e-9, amp=amp)
            .pulse(drive_channel, PulseShapeType.GAUSSIAN, width=150e-9, rise=rise)
        )
        ordered_executables = self._do_emit(builder, qblox_model)
        for executable in ordered_executables.values():
            for pkg in executable.packages.values():
                expected_phase = QbloxLegalisationPass.phase_as_steps(phase)
                assert f"set_ph_delta {expected_phase}" in pkg.sequence.program
                expected_freq = QbloxLegalisationPass.freq_as_steps(
                    drive_channel.baseband_if_frequency + frequency
                )
                assert f"set_freq {expected_freq}" in pkg.sequence.program

    def test_measure_acquire(self, qblox_model):
        qubit = qblox_model.get_qubit(0)
        acquire_channel = qubit.get_acquire_channel()
        measure_channel = qubit.get_measure_channel()
        assert measure_channel == acquire_channel

        time = 7.5e-6
        i_offs_steps = int(qubit.pulse_measure["amp"] * Constants.MAX_OFFSET)
        delay = qubit.measure_acquire["delay"]

        builder = qblox_model.create_builder()
        builder.acquire(acquire_channel, time, delay=delay)
        ordered_executables = self._do_emit(builder, qblox_model)
        for executable in ordered_executables.values():
            assert len(executable.packages) == 0

        builder = qblox_model.create_builder()
        builder.add(
            [
                MeasurePulse(measure_channel, **qubit.pulse_measure),
                Acquire(acquire_channel, time=time, delay=delay),
            ]
        )
        ordered_executables = self._do_emit(builder, qblox_model)
        remaining_width = int(qubit.pulse_measure["width"] * 1e9) - int(delay * 1e9)
        for executable in ordered_executables.values():
            assert len(executable.packages) == 1
            pkg = next(iter(executable.packages.values()))
            assert not pkg.sequence.waveforms
            assert qubit.pulse_measure["shape"] == PulseShapeType.SQUARE
            assert (
                f"set_awg_offs {i_offs_steps},0\nupd_param {int(delay * 1e9)}\nacquire 0,R0,{remaining_width}\nset_awg_offs 0,0"
                in pkg.sequence.program
            )

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
    def test_waveform_caching(self, qblox_model, width_seconds, context):
        width_samples = np.astype(width_seconds * 1e9, int)
        assert 2 * np.sum(width_samples) > Constants.MAX_SAMPLE_SIZE_WAVEFORMS

        drive_channel = qblox_model.get_qubit(0).get_drive_channel()
        builder = qblox_model.create_builder()
        for val in width_seconds:
            builder.pulse(drive_channel, PulseShapeType.GAUSSIAN, width=val, rise=1.0 / 5.0)

        with context:
            ordered_executables = self._do_emit(builder, qblox_model)
            for executable in ordered_executables.values():
                assert len(executable.packages) == 1

    @pytest.mark.parametrize("qubit_indices", [[0], [0, 1]])
    def test_resonator_spect(self, qblox_model, qubit_indices):
        qubit_indices = [0, 1]
        builder = resonator_spect(qblox_model, qubit_indices)
        ordered_executables = self._do_emit(builder, qblox_model)
        for executable in ordered_executables.values():
            assert len(executable.packages) == len(qubit_indices)

            for i, index in enumerate(qubit_indices):
                qubit = qblox_model.get_qubit(index)
                acquire_channel = qubit.get_acquire_channel()
                acquire_pkg = next(
                    (
                        pkg
                        for pulse_channel_id, pkg in executable.packages.items()
                        if pulse_channel_id == acquire_channel.full_id()
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
    def test_qubit_spect(self, qblox_model, qubit_indices):
        builder = qubit_spect(qblox_model, qubit_indices)
        ordered_executables = self._do_emit(builder, qblox_model)
        for executable in ordered_executables.values():
            assert len(executable.packages) == 2 * len(qubit_indices)

            for i, index in enumerate(qubit_indices):
                qubit = qblox_model.get_qubit(index)

                # Drive
                drive_channel = qubit.get_drive_channel()
                drive_pkg = next(
                    (
                        pkg
                        for pulse_channel_id, pkg in executable.packages.items()
                        if pulse_channel_id == drive_channel.full_id()
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
                        for pulse_channel_id, pkg in executable.packages.items()
                        if pulse_channel_id == acquire_channel.full_id()
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
    def test_scope_acquisition(self, qblox_model, qubit_indices):
        builder = measure_acquire(qblox_model, qubit_indices)
        ordered_executables = self._do_emit(builder, qblox_model)
        for executable in ordered_executables.values():
            assert len(executable.packages) == len(qubit_indices)

            acquire_pkg = next((pkg for pkg in executable.packages.values()))
            assert "weighed_acquire" not in acquire_pkg.sequence.program

        builder = measure_acquire(qblox_model, qubit_indices, do_X=True)
        ordered_executables = self._do_emit(builder, qblox_model)
        for executable in ordered_executables.values():
            assert len(executable.packages) == 2 * len(qubit_indices)

            # Enable weights
            builder = measure_acquire(qblox_model, qubit_indices)

            for index in qubit_indices:
                qubit = qblox_model.get_qubit(index)
                num_samples = int(qubit.measure_acquire["width"] * 1e9)
                acquire = next(
                    (inst for inst in builder.instructions if isinstance(inst, Acquire))
                )
                weights = np.random.rand(num_samples) + 1j * np.random.rand(num_samples)
                qubit.measure_acquire["weights"] = weights
                acquire.filter = CustomPulse(acquire.channel, weights)

        ordered_executables = self._do_emit(builder, qblox_model)
        for executable in ordered_executables.values():
            acquire_pkg = next((pkg for pkg in executable.packages.values()))
            assert "acquire_weighed" in acquire_pkg.sequence.program

    @pytest.mark.parametrize("qubit_indices", [[0]])
    def test_multi_readout(self, qblox_model, qubit_indices):
        builder = multi_readout(qblox_model, qubit_indices, do_X=False)
        ordered_executables = self._do_emit(builder, qblox_model)
        for executable in ordered_executables.values():
            assert len(executable.packages) == len(qubit_indices)

            for index in qubit_indices:
                qubit = qblox_model.get_qubit(index)
                measure_channel = qubit.get_measure_channel()

                # Readout
                measure_pkg = next(
                    (
                        pkg
                        for pulse_channel_id, pkg in executable.packages.items()
                        if pulse_channel_id == measure_channel.full_id()
                    )
                )
                assert measure_pkg.sequence.acquisitions
                assert measure_pkg.sequence.program.count("acquire") == 2

        builder = multi_readout(qblox_model, qubit_indices, do_X=True)
        with pytest.raises(ValueError):
            self._do_emit(builder, qblox_model)

        old_value = qatconfig.INSTRUCTION_VALIDATION.NO_MID_CIRCUIT_MEASUREMENT
        try:
            qatconfig.INSTRUCTION_VALIDATION.NO_MID_CIRCUIT_MEASUREMENT = False
            ordered_executables = self._do_emit(builder, qblox_model)
            for executable in ordered_executables.values():
                assert len(executable.packages) == 2 * len(qubit_indices)

                for index in qubit_indices:
                    qubit = qblox_model.get_qubit(index)
                    measure_channel = qubit.get_measure_channel()

                    # Readout
                    measure_pkg = next(
                        (
                            pkg
                            for pulse_channel_id, pkg in executable.packages.items()
                            if pulse_channel_id == measure_channel.full_id()
                        )
                    )

                    assert measure_pkg.sequence.acquisitions
                    assert measure_pkg.sequence.program.count("acquire") == 2
        finally:
            qatconfig.INSTRUCTION_VALIDATION.NO_MID_CIRCUIT_MEASUREMENT = old_value

    @pytest.mark.parametrize("pulse_width", [0, Constants.GRID_TIME, 1e3, 1e6])
    @pytest.mark.parametrize(
        "delay_width", [0, Constants.GRID_TIME, 100, 1e3 - Constants.GRID_TIME, 1e3]
    )
    def test_measure_acquire_operands(self, qblox_model, pulse_width, delay_width):
        qubit_indices = [0]
        for index in qubit_indices:
            qubit = qblox_model.get_qubit(index)
            qubit.pulse_measure["width"] = pulse_width * 1e-9
            qubit.measure_acquire["delay"] = delay_width * 1e-9

        builder = measure_acquire(qblox_model, qubit_indices)

        effective_width = max(min(pulse_width, delay_width), Constants.GRID_TIME)
        if 0 < pulse_width < effective_width + Constants.GRID_TIME:
            with pytest.raises(ValueError):
                self._do_emit(builder, qblox_model)

        else:
            ordered_executables = self._do_emit(builder, qblox_model)
            for executable in ordered_executables.values():
                packages = executable.packages
                if pulse_width < Constants.GRID_TIME:
                    assert len(packages) == 0
                else:
                    assert len(packages) == len(qubit_indices)
                    for pkg in packages.values():
                        program = pkg.sequence.program
                        quotient = effective_width // Constants.MAX_WAIT_TIME
                        remainder = effective_width % Constants.MAX_WAIT_TIME
                        if quotient > 1:
                            assert f"wait {remainder}" in program

    @pytest.mark.parametrize("qubit_indices", [[0], [0, 1]])
    def test_discrimination(self, qblox_model, qubit_indices):
        qubits = [qblox_model.get_qubit(index) for index in qubit_indices]

        builder = discrimination(qblox_model, qubit_indices)
        acquire = next(inst for inst in builder.instructions if isinstance(inst, Acquire))
        acq_width = int(calculate_duration(acquire))

        ordered_executables = self._do_emit(builder, qblox_model)
        for executable in ordered_executables.values():
            assert len(executable.packages) == len(qubits)
            qub_pkg_zip = [
                (qub, pkg)
                for (qub, (pulse_channel_id, pkg)) in zip(
                    qubits, executable.packages.items()
                )
                if qub.get_measure_channel()
                == qblox_model.get_pulse_channel_from_id(pulse_channel_id)
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


@pytest.mark.parametrize("qblox_model", [None], indirect=True)
class TestQbloxBackend2:
    def middleend_pipeline(self, qblox_model):
        target_data = QbloxTargetData.default()
        return (
            PassManager()
            | PhaseOptimisation()
            | PostProcessingSanitisation()
            | DeviceUpdateSanitisation()  #
            | InstructionValidation(target_data)  #
            | ReadoutValidation(qblox_model)
            | RepeatSanitisation(qblox_model, target_data)
            | ScopeSanitisation()
            | ReturnSanitisation()
            | DesugaringPass()  #
            | TriagePass()  # Backend, PartitionedIR pass?
            | BindingPass()  # Backend
            | TILegalisationPass()  # Backend
            | QbloxLegalisationPass()  # Backend
        )

    def _do_emit(self, builder, qblox_model, ignore_empty=True):
        res_mgr = ResultManager()
        met_mgr = MetricsManager()
        self.middleend_pipeline(qblox_model).run(builder, res_mgr, met_mgr)
        return QbloxBackend2(qblox_model).emit(builder, res_mgr, met_mgr, ignore_empty)

    def test_prologue_epilogue(self, qblox_model):
        builder = empty(qblox_model)
        assert len(builder.instructions) == 3

        ordered_executables = self._do_emit(builder, qblox_model, ignore_empty=False)
        assert len(ordered_executables) == 1
        for executable in ordered_executables.values():
            assert len(executable.packages) == 2

            for pkg in executable.packages.values():
                assert pkg.timeline.size == 4
                assert not pkg.sequence.waveforms
                assert not pkg.sequence.acquisitions
                assert not pkg.sequence.weights

                assert "set_mrk 3\nset_latch_en 1,4\nupd_param 4" in pkg.sequence.program
                assert "stop" in pkg.sequence.program

    @pytest.mark.parametrize("num_points", [1, 10, 100])
    @pytest.mark.parametrize("qubit_indices", [[0], [0, 1]])
    def test_resonator_spect(self, qblox_model, num_points, qubit_indices):
        builder = resonator_spect(qblox_model, qubit_indices, num_points)
        sweeps = [inst for inst in builder.instructions if isinstance(inst, Sweep)]
        ordered_executables = self._do_emit(builder, qblox_model)
        assert len(ordered_executables) == 1
        for executable in ordered_executables.values():
            assert len(executable.packages) == len(qubit_indices)

            for index in qubit_indices:
                qubit = qblox_model.get_qubit(index)
                acquire_channel = qubit.get_acquire_channel()
                acquire_pkg = next(
                    (
                        pkg
                        for pulse_channel_id, pkg in executable.packages.items()
                        if pulse_channel_id == acquire_channel.full_id()
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
                for sweep in sweeps:
                    assert f"sweep_{hash(sweep)}_0" in acquire_pkg.sequence.program

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
    def test_qubit_spect(self, qblox_model, num_points, qubit_indices):
        builder = qubit_spect(qblox_model, qubit_indices, num_points)

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
            ordered_executables = self._do_emit(builder, qblox_model)
            assert len(ordered_executables) == 1
            for executable in ordered_executables.values():
                assert len(executable.packages) == 2 * len(qubit_indices)

                for index in qubit_indices:
                    qubit = qblox_model.get_qubit(index)
                    drive_channel = qubit.get_drive_channel()
                    acquire_channel = qubit.get_acquire_channel()

                    # Drive
                    drive_pkg = next(
                        (
                            pkg
                            for pulse_channel_id, pkg in executable.packages.items()
                            if pulse_channel_id == drive_channel.full_id()
                        )
                    )
                    drive_pulse = next(
                        (
                            inst
                            for inst in builder.instructions
                            if isinstance(inst, Pulse)
                            and drive_channel in inst.quantum_targets
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
                            for pulse_channel_id, pkg in executable.packages.items()
                            if pulse_channel_id == acquire_channel.full_id()
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
    def test_delay_iteration(self, qblox_model, num_points, qubit_indices):
        builder = delay_iteration(qblox_model, qubit_indices, num_points)
        ordered_executables = self._do_emit(builder, qblox_model)
        assert len(ordered_executables) == 1
        for executable in ordered_executables.values():
            assert len(executable.packages) == 2 * len(qubit_indices)

    @pytest.mark.parametrize("num_points", [10])
    @pytest.mark.parametrize("qubit_indices", [[0]])
    def test_pulse_width_iteration(self, qblox_model, num_points, qubit_indices):
        builder = pulse_width_iteration(qblox_model, qubit_indices, num_points)
        ordered_executables = self._do_emit(builder, qblox_model)
        assert len(ordered_executables) == 1
        for executable in ordered_executables.values():
            assert len(executable.packages) == 2 * len(qubit_indices)

    @pytest.mark.parametrize("num_points", [10])
    @pytest.mark.parametrize("qubit_indices", [[0]])
    def test_pulse_amplitude_iteration(self, qblox_model, num_points, qubit_indices):
        builder = pulse_amplitude_iteration(qblox_model, qubit_indices, num_points)
        ordered_executables = self._do_emit(builder, qblox_model)
        assert len(ordered_executables) == 1
        for executable in ordered_executables.values():
            assert len(executable.packages) == 2 * len(qubit_indices)

    @pytest.mark.parametrize("num_points", [10])
    @pytest.mark.parametrize("qubit_indices", [[0]])
    def test_time_and_phase_iteration(self, qblox_model, num_points, qubit_indices):
        builder = time_and_phase_iteration(qblox_model, qubit_indices, num_points)
        ordered_executables = self._do_emit(builder, qblox_model)
        assert len(ordered_executables) == 1
        for executable in ordered_executables.values():
            assert len(executable.packages) == 2 * len(qubit_indices)
