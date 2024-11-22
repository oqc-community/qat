import numpy as np
import pytest

from qat.backend.analysis_passes import (
    BindingPass,
    TILegalisationPass,
    TriagePass,
    TriageResult,
)
from qat.backend.transform_passes import (
    DesugaringPass,
    RepeatSanitisation,
    ReturnSanitisation,
    ScopeSanitisation,
)
from qat.ir.pass_base import InvokerMixin, PassManager
from qat.ir.result_base import ResultManager
from qat.purr.backends.qblox.analysis_passes import QbloxLegalisationPass
from qat.purr.backends.qblox.codegen import (
    NewQbloxEmitter,
    QbloxEmitter,
    get_nco_phase_arguments,
    get_nco_set_frequency_arguments,
)
from qat.purr.backends.qblox.constants import Constants
from qat.purr.backends.qblox.device import QbloxPhysicalBaseband, QbloxPhysicalChannel
from qat.purr.compiler.devices import PulseShapeType
from qat.purr.compiler.emitter import InstructionEmitter
from qat.purr.compiler.execution import DeviceInjectors
from qat.purr.compiler.instructions import (
    Acquire,
    DeviceUpdate,
    MeasurePulse,
    Pulse,
    Variable,
)
from qat.purr.compiler.runtime import get_builder
from qat.purr.utils.logger import get_default_logger
from qat.utils.algorithm import stable_partition

from tests.qat.utils.builder_nuggets import qubit_spect, resonator_spect

log = get_default_logger()


@pytest.mark.parametrize("model", [None], indirect=True)
class TestQbloxEmitter:
    def test_play_guassian(self, model):
        width = 100e-9
        rise = 1.0 / 5.0
        drive_channel = model.get_qubit(0).get_drive_channel()
        gaussian = Pulse(drive_channel, PulseShapeType.GAUSSIAN, width=width, rise=rise)
        builder = get_builder(model).add(gaussian)

        qat_file = InstructionEmitter().emit(builder.instructions, model)
        packages = QbloxEmitter().emit(qat_file)
        assert len(packages) == 1
        pkg = packages[0]
        assert pkg.target == drive_channel
        assert f"GAUSSIAN_{hash(gaussian)}_I" in pkg.sequence.waveforms
        assert f"GAUSSIAN_{hash(gaussian)}_Q" in pkg.sequence.waveforms
        assert "play 0,1,100" in pkg.sequence.program

    def test_play_square(self, model):
        width = 100e-9
        amp = 1

        drive_channel = model.get_qubit(0).get_drive_channel()
        builder = get_builder(model).pulse(
            drive_channel, PulseShapeType.SQUARE, width=width, amp=amp
        )
        qat_file = InstructionEmitter().emit(builder.instructions, model)
        packages = QbloxEmitter().emit(qat_file)
        assert len(packages) == 1
        pkg = packages[0]
        assert pkg.target == drive_channel
        assert not pkg.sequence.waveforms
        assert (
            f"set_awg_offs {int(Constants.MAX_OFFSET_SIZE // 2)},0\nupd_param {100}\nset_awg_offs 0,0"
            in pkg.sequence.program
        )

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
        qat_file = InstructionEmitter().emit(builder.instructions, model)
        packages = QbloxEmitter().emit(qat_file)
        for package in packages:
            expected_phase = get_nco_phase_arguments(phase)
            assert f"set_ph_delta {expected_phase}" in package.sequence.program
            expected_freq = get_nco_set_frequency_arguments(
                drive_channel.baseband_if_frequency + frequency
            )
            assert f"set_freq {expected_freq}" in package.sequence.program

    def test_measure_acquire(self, model):
        qubit = model.get_qubit(0)
        acquire_channel = qubit.get_acquire_channel()
        measure_channel = qubit.get_measure_channel()
        assert measure_channel == acquire_channel

        time = 7.5e-6
        measure_width = int(qubit.pulse_measure["width"] * 1e9)
        i_offs_steps = int(qubit.pulse_measure["amp"] * (Constants.MAX_OFFSET_SIZE // 2))
        delay = qubit.measure_acquire["delay"]

        builder = get_builder(model).acquire(acquire_channel, time, delay=delay)
        qat_file = InstructionEmitter().emit(builder.instructions, model)
        packages = QbloxEmitter().emit(qat_file)
        assert len(packages) == 0

        builder = get_builder(model)
        builder.add(
            [
                MeasurePulse(measure_channel, **qubit.pulse_measure),
                Acquire(acquire_channel, time=time, delay=delay),
            ]
        )
        qat_file = InstructionEmitter().emit(builder.instructions, model)
        packages = QbloxEmitter().emit(qat_file)
        assert len(packages) == 1
        pkg = packages[0]
        assert not pkg.sequence.waveforms
        assert qubit.pulse_measure["shape"] == PulseShapeType.SQUARE
        assert (
            f"set_awg_offs {i_offs_steps},0\nwait {int(delay*1e9)}\nacquire 0,R0,{measure_width}\nset_awg_offs 0,0"
            in pkg.sequence.program
        )

        channel = acquire_channel.physical_channel
        assert isinstance(channel, QbloxPhysicalChannel)
        assert isinstance(channel.baseband, QbloxPhysicalBaseband)
        assert len(channel.config.sequencers) > 0
        assert pkg.sequencer_config.square_weight_acq.integration_length == int(time * 1e9)

    @pytest.mark.parametrize(
        "width_seconds",
        [
            np.full(30, 5e-6),
            np.full(50, 8e-6),
            np.full(20, (Constants.MAX_SAMPLE_SIZE_WAVEFORMS / 2) * 1e-9),  # 8.192e-6
            np.full(20, 10e-6),
        ],
    )
    def test_waveform_caching(self, model, width_seconds):
        rise = 1.0 / 5.0
        drive_channel = model.get_qubit(0).get_drive_channel()
        builder = get_builder(model)

        width_samples = width_seconds * 1e9
        assert 2 * int(np.sum(width_samples)) > Constants.MAX_SAMPLE_SIZE_WAVEFORMS

        for val in width_seconds:
            builder.pulse(drive_channel, PulseShapeType.GAUSSIAN, width=val, rise=rise)
        qat_file = InstructionEmitter().emit(builder.instructions, model)

        if (width_seconds > (Constants.MAX_SAMPLE_SIZE_WAVEFORMS / 2) * 1e-9).any():
            with pytest.raises(ValueError):
                QbloxEmitter().emit(qat_file)
        else:
            packages = QbloxEmitter().emit(qat_file)
            assert len(packages) == 1

    def test_single_resonator_spect(self, model):
        index = 0
        builder = resonator_spect(model, [index])
        qubit = model.get_qubit(index)
        acquire_channel = qubit.get_acquire_channel()

        qat_file = InstructionEmitter().emit(builder.instructions, model)
        packages = QbloxEmitter().emit(qat_file)
        assert len(packages) == 1
        acquire_pkg = packages[0]
        assert acquire_pkg.target == acquire_channel

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
        qat_file = InstructionEmitter().emit(builder.instructions, model)
        packages = QbloxEmitter().emit(qat_file)
        assert len(packages) == len(indices)

        for i, index in enumerate(indices):
            qubit = model.get_qubit(index)
            acquire_channel = qubit.get_acquire_channel()
            acquire_pkg = next((pkg for pkg in packages if pkg.target == acquire_channel))
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

        qat_file = InstructionEmitter().emit(builder.instructions, model)
        packages = QbloxEmitter().emit(qat_file)
        assert len(packages) == 2

        # Drive
        drive_pkg = next((pkg for pkg in packages if pkg.target == drive_channel))
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
        acquire_pkg = next((pkg for pkg in packages if pkg.target == acquire_channel))
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
        qat_file = InstructionEmitter().emit(builder.instructions, model)
        packages = QbloxEmitter().emit(qat_file)
        assert len(packages) == 2 * len(indices)

        for i, index in enumerate(indices):
            qubit = model.get_qubit(index)

            # Drive
            drive_channel = qubit.get_drive_channel()
            drive_pkg = next((pkg for pkg in packages if pkg.target == drive_channel))
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
            acquire_pkg = next((pkg for pkg in packages if pkg.target == acquire_channel))
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


@pytest.mark.parametrize("model", [None], indirect=True)
class TestNewQbloxEmitter(InvokerMixin):
    def build_pass_pipeline(self, *args, **kwargs):
        return (
            PassManager()
            | RepeatSanitisation()
            | ScopeSanitisation()
            | ReturnSanitisation()
            | DesugaringPass()
            | TriagePass()
            | BindingPass()
            | TILegalisationPass()
            | QbloxLegalisationPass()
        )

    @pytest.mark.parametrize("num_points", [1, 10, 100])
    @pytest.mark.parametrize("qubit_indices", [[0], [0, 1]])
    def test_resonator_spect(self, model, num_points, qubit_indices):
        builder = resonator_spect(model, qubit_indices, num_points)
        res_mgr = ResultManager()
        engine = model.create_engine()
        runtime = model.create_runtime()
        runtime.run_pass_pipeline(builder, res_mgr, model, engine)

        self.run_pass_pipeline(builder, res_mgr, model)
        triage_result: TriageResult = res_mgr.lookup_by_type(TriageResult)

        packages = NewQbloxEmitter().emit_packages(builder, res_mgr, model)
        assert len(packages) == len(qubit_indices)

        for index in qubit_indices:
            qubit = model.get_qubit(index)
            acquire_channel = qubit.get_acquire_channel()
            acquire_pkg = next((pkg for pkg in packages if pkg.target == acquire_channel))
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
                assert f"sweep_{hash(sweep)}_0" in acquire_pkg.sequence.program

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
        engine = model.create_engine()
        runtime = model.create_runtime()
        runtime.run_pass_pipeline(builder, res_mgr, model, engine)

        # TODO - A skeptical usage of DeviceInjectors on static device updates
        # TODO - Figure out what they mean w/r to scopes and control flow
        static_dus, builder.instructions = stable_partition(
            builder.instructions,
            lambda inst: isinstance(inst, DeviceUpdate)
            and not isinstance(inst.value, Variable),
        )

        assert len(static_dus) == len(qubit_indices)

        injectors = DeviceInjectors(static_dus)
        try:
            injectors.inject()
            self.run_pass_pipeline(builder, res_mgr, model)

            packages = NewQbloxEmitter().emit_packages(builder, res_mgr, model)
            assert len(packages) == 2 * len(qubit_indices)

            for index in qubit_indices:
                qubit = model.get_qubit(index)
                drive_channel = qubit.get_drive_channel()
                acquire_channel = qubit.get_acquire_channel()

                # Drive
                drive_pkg = next((pkg for pkg in packages if pkg.target == drive_channel))
                drive_pulse = next(
                    (
                        inst
                        for inst in builder.instructions
                        if isinstance(inst, Pulse) and drive_channel in inst.quantum_targets
                    )
                )

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
                    (pkg for pkg in packages if pkg.target == acquire_channel)
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
