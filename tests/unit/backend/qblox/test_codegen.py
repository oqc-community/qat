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
from qat.backend.qblox.config.specification import SequencerConfig
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
    Variable,
)
from qat.purr.utils.logger import get_default_logger

from tests.unit.utils.builder_nuggets import (
    delay_iteration,
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
class TestQbloxBackend1:
    def middleend_pipeline(self, model):
        target_data = QbloxTargetData.default()

        return (
            PassManager()
            | PhaseOptimisation()
            | PostProcessingSanitisation()
            | DeviceUpdateSanitisation()
            | InstructionValidation(target_data)
            | ReadoutValidation(model)
            | RepeatSanitisation(model, target_data)
            | ReturnSanitisation()
            | TriagePass()
        )

    def _do_emit(self, builder, model, ignore_empty=True):
        res_mgr = ResultManager()
        met_mgr = MetricsManager()
        self.middleend_pipeline(model).run(builder, res_mgr, met_mgr)
        return QbloxBackend1(model, ignore_empty).emit(builder, res_mgr, met_mgr)

    def test_play_guassian(self, model):
        width = 100e-9
        rise = 1.0 / 5.0
        drive_channel = model.get_qubit(0).get_drive_channel()
        gaussian = Pulse(drive_channel, PulseShapeType.GAUSSIAN, width=width, rise=rise)
        builder = model.create_builder().add(gaussian)

        ordered_executables = self._do_emit(builder, model)
        for executable in ordered_executables.values():
            assert len(executable.packages) == 1
            channel_id, pkg = next(iter(executable.packages.items()))
            assert channel_id == drive_channel.full_id()
            assert f"GAUSSIAN_{hash(gaussian)}_I" in pkg.sequence.waveforms
            assert f"GAUSSIAN_{hash(gaussian)}_Q" in pkg.sequence.waveforms
            assert "play 0,1,100" in pkg.sequence.program

    def test_play_square(self, model):
        width = 100e-9
        amp = 1

        drive_channel = model.get_qubit(0).get_drive_channel()
        builder = model.create_builder().pulse(
            drive_channel, PulseShapeType.SQUARE, width=width, amp=amp
        )
        ordered_executables = self._do_emit(builder, model)
        for executable in ordered_executables.values():
            assert len(executable.packages) == 1
            channel_id, pkg = next(iter(executable.packages.items()))
            assert channel_id == drive_channel.full_id()
            assert not pkg.sequence.waveforms
            assert (
                f"set_awg_offs {Constants.MAX_OFFSET},0\nupd_param {4}\nwait {100 - 4}\nset_awg_offs 0,0"
                in pkg.sequence.program
            )

    def test_phase_and_frequency_shift(self, model):
        amp = 1
        rise = 1.0 / 3.0
        phase = 0.72
        frequency = 500

        drive_channel = model.get_qubit(0).get_drive_channel()
        builder = (
            model.create_builder()
            .pulse(drive_channel, PulseShapeType.SQUARE, width=100e-9, amp=amp)
            .pulse(drive_channel, PulseShapeType.GAUSSIAN, width=150e-9, rise=rise)
            .phase_shift(drive_channel, phase)
            .frequency_shift(drive_channel, frequency)
            .pulse(drive_channel, PulseShapeType.SQUARE, width=100e-9, amp=amp)
            .pulse(drive_channel, PulseShapeType.GAUSSIAN, width=150e-9, rise=rise)
        )
        ordered_executables = self._do_emit(builder, model)
        for executable in ordered_executables.values():
            for pkg in executable.packages.values():
                expected_phase = QbloxLegalisationPass.phase_as_steps(phase)
                assert f"set_ph_delta {expected_phase}" in pkg.sequence.program
                expected_freq = QbloxLegalisationPass.freq_as_steps(
                    drive_channel.baseband_if_frequency + frequency
                )
                assert f"set_freq {expected_freq}" in pkg.sequence.program

    def test_measure_acquire(self, model):
        qubit = model.get_qubit(0)
        acquire_channel = qubit.get_acquire_channel()
        measure_channel = qubit.get_measure_channel()
        assert measure_channel == acquire_channel

        time = 7.5e-6
        i_offs_steps = int(qubit.pulse_measure["amp"] * Constants.MAX_OFFSET)
        delay = qubit.measure_acquire["delay"]

        builder = model.create_builder()
        builder.acquire(acquire_channel, time, delay=delay)
        ordered_executables = self._do_emit(builder, model)
        for executable in ordered_executables.values():
            assert len(executable.packages) == 0

        builder = model.create_builder()
        builder.add(
            [
                MeasurePulse(measure_channel, **qubit.pulse_measure),
                Acquire(acquire_channel, time=time, delay=delay),
            ]
        )
        ordered_executables = self._do_emit(builder, model)
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
            assert pkg.sequencer_config.square_weight_acq.integration_length == int(
                time * 1e9
            )

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
        builder = model.create_builder()
        for val in width_seconds:
            builder.pulse(drive_channel, PulseShapeType.GAUSSIAN, width=val, rise=1.0 / 5.0)

        with context:
            ordered_executables = self._do_emit(builder, model)
            for executable in ordered_executables.values():
                assert len(executable.packages) == 1

    @pytest.mark.parametrize("qubit_indices", [[0], [0, 1]])
    def test_resonator_spect(self, model, qubit_indices):
        qubit_indices = [0, 1]
        builder = resonator_spect(model, qubit_indices)
        ordered_executables = self._do_emit(builder, model)
        for executable in ordered_executables.values():
            assert len(executable.packages) == len(qubit_indices)

            for i, index in enumerate(qubit_indices):
                qubit = model.get_qubit(index)
                acquire_channel = qubit.get_acquire_channel()
                acquire_pkg = next(
                    (
                        pkg
                        for channel_id, pkg in executable.packages.items()
                        if channel_id == acquire_channel.full_id()
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
    def test_qubit_spect(self, model, qubit_indices):
        builder = qubit_spect(model, qubit_indices)
        ordered_executables = self._do_emit(builder, model)
        for executable in ordered_executables.values():
            assert len(executable.packages) == 2 * len(qubit_indices)

            for i, index in enumerate(qubit_indices):
                qubit = model.get_qubit(index)

                # Drive
                drive_channel = qubit.get_drive_channel()
                drive_pkg = next(
                    (
                        pkg
                        for channel_id, pkg in executable.packages.items()
                        if channel_id == drive_channel.full_id()
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
                        for channel_id, pkg in executable.packages.items()
                        if channel_id == acquire_channel.full_id()
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
    def test_scope_acquisition(self, model, qubit_indices):
        builder = measure_acquire(model, qubit_indices)
        ordered_executables = self._do_emit(builder, model)
        for executable in ordered_executables.values():
            assert len(executable.packages) == len(qubit_indices)

            acquire_pkg = next((pkg for pkg in executable.packages.values()))
            assert "weighed_acquire" not in acquire_pkg.sequence.program

        builder = measure_acquire(model, qubit_indices, do_X=True)
        ordered_executables = self._do_emit(builder, model)
        for executable in ordered_executables.values():
            assert len(executable.packages) == 2 * len(qubit_indices)

            # Enable weights
            builder = measure_acquire(model, qubit_indices)

            for index in qubit_indices:
                qubit = model.get_qubit(index)
                num_samples = int(qubit.measure_acquire["width"] * 1e9)
                acquire = next(
                    (inst for inst in builder.instructions if isinstance(inst, Acquire))
                )
                weights = np.random.rand(num_samples) + 1j * np.random.rand(num_samples)
                qubit.measure_acquire["weights"] = weights
                acquire.filter = CustomPulse(acquire.channel, weights)

        ordered_executables = self._do_emit(builder, model)
        for executable in ordered_executables.values():
            acquire_pkg = next((pkg for pkg in executable.packages.values()))
            assert "acquire_weighed" in acquire_pkg.sequence.program

    @pytest.mark.parametrize("qubit_indices", [[0]])
    def test_multi_readout(self, model, qubit_indices):
        builder = multi_readout(model, qubit_indices, do_X=False)
        ordered_executables = self._do_emit(builder, model)
        for executable in ordered_executables.values():
            assert len(executable.packages) == len(qubit_indices)

            for index in qubit_indices:
                qubit = model.get_qubit(index)
                measure_channel = qubit.get_measure_channel()

                # Readout
                measure_pkg = next(
                    (
                        pkg
                        for channel_id, pkg in executable.packages.items()
                        if channel_id == measure_channel.full_id()
                    )
                )
                assert measure_pkg.sequence.acquisitions
                assert measure_pkg.sequence.program.count("acquire") == 2

        builder = multi_readout(model, qubit_indices, do_X=True)
        with pytest.raises(ValueError):
            self._do_emit(builder, model)

        old_value = qatconfig.INSTRUCTION_VALIDATION.NO_MID_CIRCUIT_MEASUREMENT
        try:
            qatconfig.INSTRUCTION_VALIDATION.NO_MID_CIRCUIT_MEASUREMENT = False
            ordered_executables = self._do_emit(builder, model)
            for executable in ordered_executables.values():
                assert len(executable.packages) == 2 * len(qubit_indices)

                for index in qubit_indices:
                    qubit = model.get_qubit(index)
                    measure_channel = qubit.get_measure_channel()

                    # Readout
                    measure_pkg = next(
                        (
                            pkg
                            for channel_id, pkg in executable.packages.items()
                            if channel_id == measure_channel.full_id()
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
            ordered_executables = self._do_emit(builder, model)
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


@pytest.mark.parametrize("model", [None], indirect=True)
class TestQbloxBackend2:
    def middleend_pipeline(self, model):
        target_data = QbloxTargetData.default()
        return (
            PassManager()
            | PhaseOptimisation()
            | PostProcessingSanitisation()
            | DeviceUpdateSanitisation()
            | InstructionValidation(target_data)
            | ReadoutValidation(model)
            | RepeatSanitisation(model, target_data)
            | ScopeSanitisation()
            | ReturnSanitisation()
            | DesugaringPass()
            | TriagePass()
            | BindingPass()
            | TILegalisationPass()
            | QbloxLegalisationPass()
        )

    def _do_emit(self, builder, model, ignore_empty=True):
        res_mgr = ResultManager()
        met_mgr = MetricsManager()
        self.middleend_pipeline(model).run(builder, res_mgr, met_mgr)
        return QbloxBackend2(model, ignore_empty).emit(builder, res_mgr, met_mgr)

    def test_prologue_epilogue(self, model):
        builder = empty(model)
        assert len(builder.instructions) == 3

        executable = self._do_emit(builder, model, ignore_empty=False)
        assert len(executable.packages) == 2

        for pkg in executable.packages.values():
            assert pkg.sequencer_config == SequencerConfig()
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
        sweeps = [inst for inst in builder.instructions if isinstance(inst, Sweep)]
        executable = self._do_emit(builder, model)
        assert len(executable.packages) == len(qubit_indices)

        for index in qubit_indices:
            qubit = model.get_qubit(index)
            acquire_channel = qubit.get_acquire_channel()
            acquire_pkg = next(
                (
                    pkg
                    for channel_id, pkg in executable.packages.items()
                    if channel_id == acquire_channel.full_id()
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
    def test_qubit_spect(self, model, num_points, qubit_indices):
        builder = qubit_spect(model, qubit_indices, num_points)

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
            executable = self._do_emit(builder, model)
            assert len(executable.packages) == 2 * len(qubit_indices)

            for index in qubit_indices:
                qubit = model.get_qubit(index)
                drive_channel = qubit.get_drive_channel()
                acquire_channel = qubit.get_acquire_channel()

                # Drive
                drive_pkg = next(
                    (
                        pkg
                        for channel_id, pkg in executable.packages.items()
                        if channel_id == drive_channel.full_id()
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
                        for channel_id, pkg in executable.packages.items()
                        if channel_id == acquire_channel.full_id()
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
        executable = self._do_emit(builder, model)
        assert len(executable.packages) == 2 * len(qubit_indices)

    @pytest.mark.parametrize("num_points", [10])
    @pytest.mark.parametrize("qubit_indices", [[0]])
    def test_pulse_width_iteration(self, model, num_points, qubit_indices):
        builder = pulse_width_iteration(model, qubit_indices, num_points)
        executable = self._do_emit(builder, model)
        assert len(executable.packages) == 2 * len(qubit_indices)

    @pytest.mark.parametrize("num_points", [10])
    @pytest.mark.parametrize("qubit_indices", [[0]])
    def test_pulse_amplitude_iteration(self, model, num_points, qubit_indices):
        builder = pulse_amplitude_iteration(model, qubit_indices, num_points)
        executable = self._do_emit(builder, model)
        assert len(executable.packages) == 2 * len(qubit_indices)

    @pytest.mark.parametrize("num_points", [10])
    @pytest.mark.parametrize("qubit_indices", [[0]])
    def test_time_and_phase_iteration(self, model, num_points, qubit_indices):
        builder = time_and_phase_iteration(model, qubit_indices, num_points)
        executable = self._do_emit(builder, model)
        assert len(executable.packages) == 2 * len(qubit_indices)
