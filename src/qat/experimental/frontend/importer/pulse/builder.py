# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
from xdsl.dialects.arith import ConstantOp as ArithConstantOp, IndexCastOp
from xdsl.dialects.builtin import BoolAttr, FloatAttr, IndexType, IntAttr, f64, i32
from xdsl.interpreters.scf import scf
from xdsl.ir import Block, Operation, Region, SSAValue

from qat.experimental.dialect.pulse.ir import (
    AcquireOp,
    AmplitudeAttr,
    BlackmanWaveformOp,
    ConstantOp,
    CreateFrameOp,
    FrameType,
    FrequencyAttr,
    GaussianSquareWaveformOp,
    GaussianWaveformOp,
    IntegrateOp,
    PhaseAttr,
    PhaseSetOp,
    PhaseShiftOp,
    PulseOp,
    ReturnOp,
    RoundedSquareWaveformOp,
    SampledWaveformAttr,
    SechWaveformOp,
    SetupHoldWaveformOp,
    SinusoidalWaveformOp,
    SoftSquareWaveformOp,
    SquareWaveformOp,
    SynchronizeOp,
    TimeAttr,
    WaitOp,
    WaveformType,
    WeightsAttr,
)
from qat.experimental.dialect.pulse.ir.ops import KernelOp
from qat.experimental.dialect.results.ir import (
    CreateOp,
    RecordType,
    ResultsCollectionType,
    StoreOp,
)
from qat.experimental.frontend.importer.environment import EnvironmentTracker


def _create_time_constant_op(time: float) -> ConstantOp:
    """Create a constant operation for a given time value."""

    # TODO: COMPILER-1388, convert to ps
    return ConstantOp(TimeAttr(time))


def _create_amplitude_constant_op(amplitude: complex | float) -> ConstantOp:
    """Create a constant operation for a given amplitude value."""
    return ConstantOp(AmplitudeAttr(amplitude))


def _create_frequency_constant_op(frequency: float) -> ConstantOp:
    """Create a constant operation for a given frequency value."""

    # TODO: COMPILER-1388, convert to Hz
    return ConstantOp(FrequencyAttr(frequency))


def _create_phase_constant_op(phase: float) -> ConstantOp:
    """Create a constant operation for a given phase value."""
    return ConstantOp(PhaseAttr(phase))


def _create_float_constant_op(value: float) -> ArithConstantOp:
    """Create a builtin float constant for dimensionless waveform parameters."""
    return ArithConstantOp(FloatAttr(value, 64), f64)


def _create_drag_constant_ops(
    drag_coefficients: tuple[float, ...],
) -> list[ArithConstantOp]:
    """Create builtin float constants for DRAG coefficients."""
    return [_create_float_constant_op(value) for value in drag_coefficients]


class PulseKernelBuilder:
    """Base class for importers that translate a program into the Pulse dialect.

    Provides common machinery for assembling programs at the pulse-level, which can be used
    to assemble a pulse-level kernel.

    It does not yet support custom control flow constructs, but it provides the ability
    to pass through the number of shots, and eventually sweep parameters, which is then used
    to build the structured control flow around the program. Likewise, it does not support
    parameterisation of arguments in the operations it constructs (or kernel arguments).
    This is likely to change in the future.

    .. warning::

        This is intended for internal compiler use and is not part of the public API. We use
        it in our importers for pulse-level programs, and it is a useful tool for testing.
        We cannot guarantee that it will remain stable or supported in future releases. Use
        with caution.

    The builder allows for symbolic construction of frames, waveforms and acquisition
    results that can be used in subsequent operations. The methods to build operations
    return the builder instance to allow for method chaining. For example,

    .. code-block:: python

        kernel = (
            PulseKernelBuilder("my_kernel", shots=1000)
            .create_frame("q0/drive", 4.8e9, "q0")
            .create_frame("q0/readout", 8.8e9, "r0")
            .create_frame("q0/acquire", 8.8e9, "r0")
            .phase_set("q0/drive", 0.0)
            .phase_set("q0/acquire", 0.0)
            .create_gaussian_waveform("q0/drive", 0.5, 80e-9, 0.47, False)
            .create_square_waveform("q0/readout", 0.5, 800e-9)
            .pulse("q0/drive", "q0/drive")
            .phase_shift("q0/drive", 1.57)
            .pulse("q0/drive", "q0/drive")
            .synchronize("q0/drive", "q0/readout", "q0/acquire")
            .pulse("q0/readout", "q0/readout")
            .wait("q0/acquire", 80e-9)
            .acquire("q0/acquire", "q0_meas", 800e-9, integrate=True)
            .synchronize("q0/drive", "q0/readout", "q0/acquire")
            .wait("q0/drive", 500e-6)
            .synchronize("q0/drive", "q0/readout", "q0/acquire")
        )

    The previous program would implement a simple experiment to drive a qubit and measure
    it, with a reset time between shots.
    """

    def __init__(self, name: str, shots: int | None = None):
        """Initialise a new PulseKernelBuilder.

        :param name: The name of the kernel to build.
        :param shots: The number of shots to run the kernel for. Setting it to ``None``
            (default) will add no shots.
        """
        self._name = name
        self._num_shots = shots
        self._pulse_block = Block()
        self._frames = EnvironmentTracker[FrameType]()
        self._acquires = EnvironmentTracker()
        self._waveforms = EnvironmentTracker[WaveformType]()

    def create_frame(
        self, frame_name: str, frequency: float, port: str
    ) -> "PulseKernelBuilder":
        """Create a new frame and add it to the pulse block.

        :param frame_name: The name of the frame to create.
        :param frequency: The frequency of the frame to create.
        :param port: The port of the frame to create.
        :returns: The PulseKernelBuilder instance for method chaining.
        """
        freq_op = _create_frequency_constant_op(frequency)
        frame_op = CreateFrameOp(freq_op, port)
        self._add_ops(freq_op, frame_op)
        self._frames.set_by_name(frame_name, frame_op.result)
        return self

    def phase_set(self, frame_name: str, phase: float) -> "PulseKernelBuilder":
        """Set the phase of a frame.

        :param frame_name: The name of the frame to set the phase for.
        :param phase: The phase value to set.
        :returns: The PulseKernelBuilder instance for method chaining.
        """
        phase_op = _create_phase_constant_op(phase)
        frame_op = self._get_frame(frame_name)
        set_phase_op = PhaseSetOp(frame_op, phase_op)
        self._add_ops(phase_op, set_phase_op)
        self._update_frame(frame_name, set_phase_op.result)
        return self

    def phase_shift(self, frame_name: str, phase: float) -> "PulseKernelBuilder":
        """Shift the phase of a frame.

        :param frame_name: The name of the frame to shift the phase for.
        :param phase: The phase value to shift.
        :returns: The PulseKernelBuilder instance for method chaining.
        """
        phase_op = _create_phase_constant_op(phase)
        frame_op = self._get_frame(frame_name)
        shift_phase_op = PhaseShiftOp(frame_op, phase_op)
        self._add_ops(phase_op, shift_phase_op)
        self._update_frame(frame_name, shift_phase_op.result)
        return self

    def wait(self, frame_name: str, duration: float) -> "PulseKernelBuilder":
        """Add a wait operation to the pulse block.

        :param frame_name: The name of the frame to wait on.
        :param duration: The duration of the wait.
        :returns: The PulseKernelBuilder instance for method chaining.
        """
        time_op = _create_time_constant_op(duration)
        frame_op = self._get_frame(frame_name)
        wait_op = WaitOp(frame_op, time_op)
        self._add_ops(time_op, wait_op)
        self._update_frame(frame_name, wait_op.result)
        return self

    def synchronize(self, *frame_names: str) -> "PulseKernelBuilder":
        """Add a synchronize operation to the pulse block.

        :param frame_names: The names of the frames to synchronize.
        :returns: The PulseKernelBuilder instance for method chaining.
        """
        if len(frame_names) < 2:
            return self

        frame_ops = [self._get_frame(name) for name in frame_names]
        sync_op = SynchronizeOp(*frame_ops)
        self._add_ops(sync_op)
        for name, result in zip(frame_names, sync_op.result, strict=False):
            self._update_frame(name, result)
        return self

    def acquire(
        self,
        frame_name: str,
        acquire_name: str,
        duration: float,
        weights: list[float | complex] | None = None,
        integrate: bool = True,
    ) -> "PulseKernelBuilder":
        """Add an acquire operation to the pulse block.

        :param frame_name: The name of the frame to acquire from.
        :param acquire_name: The name used to track the acquisition result that can be used
            in post-processing.
        :param duration: The duration of the acquisition.
        :param weights: The weights to use for the acquisition.
        :param integrate: Whether to integrate the acquisition result to give an IQ value.
        :returns: The PulseKernelBuilder instance for method chaining.
        """
        time_op = _create_time_constant_op(duration)
        frame_op = self._get_frame(frame_name)
        if weights is not None:
            weights_attr = WeightsAttr(weights)
            acquire_op = AcquireOp(frame_op, time_op, weights_attr, label=acquire_name)
        else:
            acquire_op = AcquireOp(frame_op, time_op, label=acquire_name)
        self._add_ops(time_op, acquire_op)
        self._update_frame(frame_name, acquire_op.frame_result)
        self._acquires.set_by_name(acquire_name, acquire_op.acquisition_result)

        if integrate:
            integrate_op = IntegrateOp(acquire_op.acquisition_result)
            self._add_ops(integrate_op)
            self._acquires.set_by_name(acquire_name, integrate_op.result)

        return self

    def pulse(self, frame_name: str, waveform_name: str) -> "PulseKernelBuilder":
        """Add a pulse operation to the pulse block.

        :param frame_name: The name of the frame to pulse.
        :param waveform_name: The name of the waveform to use for the pulse.
        :returns: The PulseKernelBuilder instance for method chaining.
        """
        frame_op = self._get_frame(frame_name)
        waveform_op = self._get_waveform(waveform_name)
        pulse_op = PulseOp(frame_op, waveform_op)
        self._add_ops(pulse_op)
        self._update_frame(frame_name, pulse_op.result)
        return self

    def create_square_waveform(
        self,
        waveform_name: str,
        amplitude: float,
        duration: float,
    ) -> "PulseKernelBuilder":
        """Create a square waveform and add it to the pulse block.

        :param waveform_name: The name of the waveform to create.
        :param amplitude: The amplitude of the waveform.
        :param duration: The duration of the waveform.
        :returns: The PulseKernelBuilder instance for method chaining.
        """
        amplitude_op = _create_amplitude_constant_op(amplitude)
        time_op = _create_time_constant_op(duration)
        square_waveform_op = SquareWaveformOp(time_op, amplitude_op)
        self._add_ops(amplitude_op, time_op, square_waveform_op)
        self._waveforms.set_by_name(waveform_name, square_waveform_op.result)
        return self

    def create_gaussian_waveform(
        self,
        waveform_name: str,
        amplitude: float,
        duration: float,
        fractional_breadth: float,
        regularize: bool = False,
        *drag_coefficients: float,
    ) -> "PulseKernelBuilder":
        """Create a Gaussian waveform and add it to the pulse block.

        :param waveform_name: The name of the waveform to create.
        :param amplitude: The amplitude of the waveform.
        :param duration: The duration of the waveform.
        :param fractional_breadth: Gaussian width proportion in normalized units.
        :param regularize: Whether to regularize the shape at the edges.
        :param drag_coefficients: DRAG coefficients by derivative order, starting at order
            1.
        :returns: The PulseKernelBuilder instance for method chaining.
        """
        amplitude_op = _create_amplitude_constant_op(amplitude)
        time_op = _create_time_constant_op(duration)
        fractional_breadth_op = _create_float_constant_op(fractional_breadth)
        drag_ops = _create_drag_constant_ops(drag_coefficients)
        gaussian_waveform_op = GaussianWaveformOp(
            time_op,
            amplitude_op,
            fractional_breadth_op,
            BoolAttr(regularize, value_type=1),
            *(op.result for op in drag_ops),
        )
        self._add_ops(
            amplitude_op,
            time_op,
            fractional_breadth_op,
            *drag_ops,
            gaussian_waveform_op,
        )
        self._waveforms.set_by_name(waveform_name, gaussian_waveform_op.result)
        return self

    def create_soft_square_waveform(
        self,
        waveform_name: str,
        amplitude: float,
        duration: float,
        fractional_top_width: float,
        fractional_rise: float,
        regularize: bool = False,
        *drag_coefficients: float,
    ) -> "PulseKernelBuilder":
        """Create a soft square waveform and add it to the pulse block.

        :param waveform_name: The name of the waveform to create.
        :param amplitude: The amplitude of the waveform.
        :param duration: The duration of the waveform.
        :param fractional_top_width: Flat-top proportion in normalized units.
        :param fractional_rise: Rise and fall width proportion in normalized units.
        :param regularize: Whether to regularize the shape at the edges.
        :param drag_coefficients: DRAG coefficients by derivative order, starting at order
            1.
        :returns: The PulseKernelBuilder instance for method chaining.
        """
        amplitude_op = _create_amplitude_constant_op(amplitude)
        time_op = _create_time_constant_op(duration)
        fractional_top_width_op = _create_float_constant_op(fractional_top_width)
        fractional_rise_op = _create_float_constant_op(fractional_rise)
        drag_ops = _create_drag_constant_ops(drag_coefficients)
        soft_square_waveform_op = SoftSquareWaveformOp(
            time_op,
            amplitude_op,
            fractional_top_width_op,
            fractional_rise_op,
            BoolAttr(regularize, value_type=1),
            *(op.result for op in drag_ops),
        )
        self._add_ops(
            amplitude_op,
            time_op,
            fractional_top_width_op,
            fractional_rise_op,
            *drag_ops,
            soft_square_waveform_op,
        )
        self._waveforms.set_by_name(waveform_name, soft_square_waveform_op.result)
        return self

    def create_gaussian_square_waveform(
        self,
        waveform_name: str,
        amplitude: float,
        duration: float,
        fractional_rise: float,
        fractional_top_width: float,
        regularize: bool = False,
        drag_coefficient: float | None = None,
    ) -> "PulseKernelBuilder":
        """Create a Gaussian square waveform and add it to the pulse block.

        :param waveform_name: The name of the waveform to create.
        :param amplitude: The amplitude of the waveform.
        :param duration: The duration of the waveform.
        :param fractional_rise: Gaussian edge-width proportion in normalized units.
        :param fractional_top_width: Flat-top proportion in normalized units.
        :param regularize: Whether to regularize the shape at the edges.
        :param drag_coefficient: DRAG coefficient. Gaussian-square supports at most one.
        :returns: The PulseKernelBuilder instance for method chaining.
        """
        drag_coefficients = (drag_coefficient,) if drag_coefficient is not None else ()
        amplitude_op = _create_amplitude_constant_op(amplitude)
        time_op = _create_time_constant_op(duration)
        fractional_rise_op = _create_float_constant_op(fractional_rise)
        fractional_top_width_op = _create_float_constant_op(fractional_top_width)
        drag_ops = _create_drag_constant_ops(drag_coefficients)
        gaussian_square_waveform_op = GaussianSquareWaveformOp(
            time_op,
            amplitude_op,
            fractional_rise_op,
            fractional_top_width_op,
            BoolAttr(regularize, value_type=1),
            *(op.result for op in drag_ops),
        )
        self._add_ops(
            amplitude_op,
            time_op,
            fractional_rise_op,
            fractional_top_width_op,
            *drag_ops,
            gaussian_square_waveform_op,
        )
        self._waveforms.set_by_name(waveform_name, gaussian_square_waveform_op.result)
        return self

    def create_blackman_waveform(
        self,
        waveform_name: str,
        amplitude: float,
        duration: float,
        *drag_coefficients: float,
    ) -> "PulseKernelBuilder":
        """Create a Blackman waveform and add it to the pulse block.

        :param waveform_name: The name of the waveform to create.
        :param amplitude: The amplitude of the waveform.
        :param duration: The duration of the waveform.
        :param drag_coefficients: DRAG coefficients by derivative order, starting at order
            1.
        :returns: The PulseKernelBuilder instance for method chaining.
        """
        amplitude_op = _create_amplitude_constant_op(amplitude)
        time_op = _create_time_constant_op(duration)
        drag_ops = _create_drag_constant_ops(drag_coefficients)
        blackman_waveform_op = BlackmanWaveformOp(
            time_op, amplitude_op, *(op.result for op in drag_ops)
        )
        self._add_ops(amplitude_op, time_op, *drag_ops, blackman_waveform_op)
        self._waveforms.set_by_name(waveform_name, blackman_waveform_op.result)
        return self

    def create_setup_hold_waveform(
        self,
        waveform_name: str,
        amplitude: float,
        duration: float,
        setup: float,
        fractional_rise: float,
    ) -> "PulseKernelBuilder":
        """Create a setup hold waveform and add it to the pulse block.

        :param waveform_name: The name of the waveform to create.
        :param amplitude: The amplitude of the waveform.
        :param duration: The duration of the waveform.
        :param setup: Relative setup amplitude.
        :param fractional_rise: Fraction of width occupied by the setup segment.
        :returns: The PulseKernelBuilder instance for method chaining.
        """
        amplitude_op = _create_amplitude_constant_op(amplitude)
        time_op = _create_time_constant_op(duration)
        setup_op = _create_float_constant_op(setup)
        fractional_rise_op = _create_float_constant_op(fractional_rise)
        setup_hold_waveform_op = SetupHoldWaveformOp(
            time_op, amplitude_op, setup_op, fractional_rise_op
        )
        self._add_ops(
            amplitude_op,
            time_op,
            setup_op,
            fractional_rise_op,
            setup_hold_waveform_op,
        )
        self._waveforms.set_by_name(waveform_name, setup_hold_waveform_op.result)
        return self

    def create_rounded_square_waveform(
        self,
        waveform_name: str,
        amplitude: float,
        duration: float,
        fractional_top_width: float,
        fractional_rise: float,
        *drag_coefficients: float,
    ) -> "PulseKernelBuilder":
        """Create a rounded square waveform and add it to the pulse block.

        :param waveform_name: The name of the waveform to create.
        :param amplitude: The amplitude of the waveform.
        :param duration: The duration of the waveform.
        :param fractional_top_width: Flat-top proportion in normalized units.
        :param fractional_rise: Rise and fall width proportion in normalized units.
        :param drag_coefficients: DRAG coefficients by derivative order, starting at order
            1.
        :returns: The PulseKernelBuilder instance for method chaining.
        """
        amplitude_op = _create_amplitude_constant_op(amplitude)
        time_op = _create_time_constant_op(duration)
        fractional_top_width_op = _create_float_constant_op(fractional_top_width)
        fractional_rise_op = _create_float_constant_op(fractional_rise)
        drag_ops = _create_drag_constant_ops(drag_coefficients)
        rounded_square_waveform_op = RoundedSquareWaveformOp(
            time_op,
            amplitude_op,
            fractional_top_width_op,
            fractional_rise_op,
            *(op.result for op in drag_ops),
        )
        self._add_ops(
            amplitude_op,
            time_op,
            fractional_top_width_op,
            fractional_rise_op,
            *drag_ops,
            rounded_square_waveform_op,
        )
        self._waveforms.set_by_name(waveform_name, rounded_square_waveform_op.result)
        return self

    def create_sech_waveform(
        self,
        waveform_name: str,
        amplitude: float,
        duration: float,
        fractional_breadth: float,
        regularize: bool = False,
        *drag_coefficients: float,
    ) -> "PulseKernelBuilder":
        """Create a hyperbolic secant (sech) waveform and add it to the pulse block.

        :param waveform_name: The name of the waveform to create.
        :param amplitude: The amplitude of the waveform.
        :param duration: The duration of the waveform.
        :param fractional_breadth: Sech width proportion in normalized units.
        :param regularize: Whether to regularize the shape at the edges.
        :param drag_coefficients: DRAG coefficients by derivative order, starting at order
            1.
        :returns: The PulseKernelBuilder instance for method chaining.
        """
        amplitude_op = _create_amplitude_constant_op(amplitude)
        time_op = _create_time_constant_op(duration)
        fractional_breadth_op = _create_float_constant_op(fractional_breadth)
        drag_ops = _create_drag_constant_ops(drag_coefficients)
        sech_waveform_op = SechWaveformOp(
            time_op,
            amplitude_op,
            fractional_breadth_op,
            BoolAttr(regularize, value_type=1),
            *(op.result for op in drag_ops),
        )
        self._add_ops(
            amplitude_op,
            time_op,
            fractional_breadth_op,
            *drag_ops,
            sech_waveform_op,
        )
        self._waveforms.set_by_name(waveform_name, sech_waveform_op.result)
        return self

    def create_sinusoidal_waveform(
        self,
        waveform_name: str,
        amplitude: float,
        duration: float,
        number_of_periods: float,
        internal_phase: float = 0.0,
        *drag_coefficients: float,
    ) -> "PulseKernelBuilder":
        """Create a sinusoidal waveform and add it to the pulse block.

        :param waveform_name: The name of the waveform to create.
        :param amplitude: The amplitude of the waveform.
        :param duration: The duration of the waveform.
        :param number_of_periods: Number of periods across the waveform domain.
        :param internal_phase: Internal phase offset in radians.
        :param drag_coefficients: DRAG coefficients by derivative order, starting at order
            1.
        :returns: The PulseKernelBuilder instance for method chaining.
        """
        amplitude_op = _create_amplitude_constant_op(amplitude)
        time_op = _create_time_constant_op(duration)
        number_of_periods_op = _create_float_constant_op(number_of_periods)
        internal_phase_op = _create_phase_constant_op(internal_phase)
        drag_ops = _create_drag_constant_ops(drag_coefficients)
        sinusoidal_waveform_op = SinusoidalWaveformOp(
            time_op,
            amplitude_op,
            number_of_periods_op,
            internal_phase_op,
            *(op.result for op in drag_ops),
        )
        self._add_ops(
            amplitude_op,
            time_op,
            number_of_periods_op,
            internal_phase_op,
            *drag_ops,
            sinusoidal_waveform_op,
        )
        self._waveforms.set_by_name(waveform_name, sinusoidal_waveform_op.result)
        return self

    def create_custom_waveform(
        self, waveform_name: str, samples: list[float | complex], duration: float
    ) -> "PulseKernelBuilder":
        """Create a custom waveform and add it to the pulse block.

        :param waveform_name: The name of the waveform to create.
        :param samples: The samples of the waveform.
        :param duration: The duration of the waveform.
        :returns: The PulseKernelBuilder instance for method chaining.
        """
        if len(samples) == 0:
            raise ValueError("Samples list cannot be empty for the sampled waveform.")

        sample_time = duration / len(samples)
        sampled_waveform_attr = SampledWaveformAttr(
            samples, TimeAttr(duration), TimeAttr(sample_time)
        )
        custom_waveform_op = ConstantOp(sampled_waveform_attr)
        self._add_ops(custom_waveform_op)
        self._waveforms.set_by_name(waveform_name, custom_waveform_op.result)
        return self

    def finalize(self) -> KernelOp:
        """Finalizes the kernel, adding the surrounding control flow and adding it to a
        kernel.

        :returns: The finalized pulse kernel.
        """

        record_op = self._create_record_op()
        self._add_ops(record_op)
        record_schema = record_op.result.type.schema
        record_type = RecordType(record_schema)

        if self._num_shots is None:
            return_op = ReturnOp(record_op.result)
            self._add_ops(return_op)
            return KernelOp(
                self._name,
                (
                    (),
                    (record_type,),
                ),
                Region(self._pulse_block),
            )

        collection_type = ResultsCollectionType(record_schema, IntAttr(self._num_shots))

        # Add an induction variable and results collection iteration argument to the block
        self._pulse_block.insert_arg(IndexType(), 0)
        self._pulse_block.insert_arg(collection_type, 1)

        cast_index_op = IndexCastOp(self._pulse_block.args[0], i32)
        store_record_op = StoreOp.record_in_collection(
            self._pulse_block.args[1], cast_index_op.result, record_op.result
        )
        yield_op = scf.YieldOp(store_record_op.result)
        self._add_ops(cast_index_op, store_record_op, yield_op)

        index_type = IndexType()
        lb = ArithConstantOp.from_int_and_width(0, index_type)
        ub = ArithConstantOp.from_int_and_width(self._num_shots, index_type)
        increment = ArithConstantOp.from_int_and_width(1, index_type)
        record_collection_op = CreateOp.for_empty_collection(record_schema, self._num_shots)
        loop = scf.ForOp(
            lb,
            ub,
            increment,
            [record_collection_op.result],
            Region(self._pulse_block),
        )
        return_op = ReturnOp(loop.results[0])
        block = Block([lb, ub, increment, record_collection_op, loop, return_op])
        return KernelOp(
            self._name,
            (
                (),
                (collection_type,),
            ),
            Region(block),
        )

    def _add_ops(self, *ops: Operation):
        """Add operations to the pulse block."""
        self._pulse_block.add_ops(ops)

    def _get_frame(self, frame_name: str) -> SSAValue[FrameType]:
        """Get the SSA value for a frame by name.

        :param frame_name: The name of the frame to get.
        :returns: The SSA value for the frame.
        """
        frame = self._frames.get_by_name(frame_name)
        if frame is None:
            raise KeyError(f"Frame '{frame_name}' not found in environment.")
        return frame

    def _update_frame(self, frame_name: str, new_frame: SSAValue[FrameType]):
        """Update the SSA value for a frame by name.

        :param frame_name: The name of the frame to update.
        :param new_frame: The new SSA value for the frame.
        """
        self._frames.set_by_name(frame_name, new_frame)

    def _get_waveform(self, waveform_name: str) -> SSAValue[WaveformType]:
        """Get the SSA value for a waveform by name.

        :param waveform_name: The name of the waveform to get.
        :returns: The SSA value for the waveform.
        """
        waveform = self._waveforms.get_by_name(waveform_name)
        if waveform is None:
            raise KeyError(f"Waveform '{waveform_name}' not found in environment.")
        return waveform

    def _create_record_op(self) -> Operation:
        """Create a record operation for the kernel.

        :returns: The record operation.
        """

        items = list(self._acquires.items())
        acquire_names, acquire_values = zip(*items, strict=False) if items else ([], [])
        return CreateOp.for_record(acquire_names, acquire_values)
