# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from functools import singledispatchmethod

import numpy as np
from xdsl.dialects.arith import ConstantOp as ArithConstantOp
from xdsl.dialects.builtin import (
    BoolAttr,
    FloatAttr,
    IntegerAttr,
    ModuleOp,
    StringAttr,
    f64,
    i32,
)
from xdsl.ir import SSAValue
from xdsl.irdl import IRDLOperation

from qat.experimental.dialect.pulse.ir import (
    AcquireOp,
    AmplitudeAttr,
    BlackmanWaveformOp,
    ConstantOp,
    CosWaveformOp,
    CreateFrameOp,
    DiscriminateOp,
    DragGaussianWaveformOp,
    EqualiseAttr,
    EqualiseOp,
    ExtraSoftSquareWaveformOp,
    FrequencyAttr,
    GaussianSquareWaveformOp,
    GaussianWaveformOp,
    GaussianZeroEdgeWaveformOp,
    IntegrateOp,
    IQResultType,
    PhaseAttr,
    PhaseSetOp,
    PhaseShiftOp,
    PulseNumericTypedAttr,
    PulseOp,
    RealThresholdPolicyAttr,
    RoundedSquareWaveformOp,
    SechWaveformOp,
    SetupHoldWaveformOp,
    SinWaveformOp,
    SofterGaussianWaveformOp,
    SofterSquareWaveformOp,
    SoftSquareWaveformOp,
    SquareWaveformOp,
    SynchronizeOp,
    TimeAttr,
    WaitOp,
    WeightsAttr,
)
from qat.experimental.frontend.importer.base import BaseLinearImporter
from qat.purr.compiler.builders import QuantumInstructionBuilder
from qat.purr.compiler.devices import PulseChannel, PulseShapeType
from qat.purr.compiler.instructions import (
    Acquire,
    AcquireMode,
    CustomPulse,
    Delay,
    DeviceUpdate,
    EndRepeat,
    EndSweep,
    PhaseReset,
    PhaseSet,
    PhaseShift,
    PostProcessing,
    PostProcessType,
    Pulse,
    QuantumInstruction,
    Repeat,
    Sweep,
    Synchronize,
    Variable,
)


class PurrImporter(BaseLinearImporter):
    """Concrete :class:`BaseLinearImporter` for the legacy Purr IR.

    Walks a :class:`~qat.purr.compiler.builders.QuantumInstructionBuilder`
    and emits an equivalent :class:`ModuleOp` in the Pulse dialect.
    Pulse channels are lowered to ``FrameType`` SSA values which are then
    tracked via the ``_environment_variable`` map. :class:`Repeat` /
    :class:`Sweep` instructions are translated into `scf.ForOps` with
    SSA values appreciate flowing through them. Purr IR variables (and
    frames) are global.
    """

    def build(self, purr_ir: QuantumInstructionBuilder) -> ModuleOp:
        """Translate all instructions in *purr_ir* into the Pulse dialect.

        :param purr_ir: A populated Purr instruction builder.
        :returns: An xDSL ``ModuleOp`` containing the translated
            Pulse-dialect operations.
        """
        for instr in purr_ir.instructions:
            self.translate(instr)
        self._add_final_return()
        return self.module

    @staticmethod
    def get_frame_key(quantum_target: PulseChannel) -> str:
        """Return the environment-variable key used for *quantum_target*.

        Pulse channel-derived key from Purr pulse channel id used as variable names for
        frames.

        :param quantum_target: The Purr pulse channel.
        :returns: A string key for a Pulse Channel.
        """
        return "purr_frame_" + quantum_target.full_id()

    def create_frame(
        self,
        target: PulseChannel,
        freq_op: IRDLOperation | SSAValue | None = None,
    ) -> SSAValue:
        """Create a new Pulse-dialect frame for *target*.

        Emits a :class:`FrameType` (and, when needed, a
        :class:`ConstantOp` for the frequency and ``CreateFrameOp)
        in the current block.

        :param target: The Purr pulse channel to create a frame for.
        :param freq_op: An optional pre-built op or SSA value carrying
            the frame's frequency.  When ``None`` a fresh
            :class:`ConstantOp` is synthesised from
            ``target.frequency``.  When an :class:`IRDLOperation` is
            supplied it is also inserted into the current block.
        :returns: The SSA value representing the new frame.
        """
        if freq_op is None:
            freq_op = ConstantOp(FrequencyAttr(target.frequency))
        frame_op = CreateFrameOp(freq_op, StringAttr(target.physical_channel.full_id()))
        if isinstance(freq_op, IRDLOperation):
            self._current_block.add_ops([freq_op, frame_op])
        else:
            self._current_block.add_ops([frame_op])
        return frame_op.result

    def get_frames(self, quantum_targets: list[PulseChannel]) -> list[SSAValue]:
        """Look up or create Pulse-dialect frames for *quantum_targets*.

        Searches the environment-variable map for an existing frame
        keyed by :meth:`get_frame_key`.  When no binding is found a
        new frame is created in the current block and registered in
        the environment.

        :param quantum_targets: Purr pulse channels to resolve.
        :returns: Ordered list of SSA frame values corresponding to
            each target.
        """
        frames = []
        for target in quantum_targets:
            frame_key = self.get_frame_key(target)
            frame = self._current_environment_variables.get(frame_key)
            if frame is None:
                frame = self.create_frame(target)
                self._current_environment_variables[frame_key] = frame
            frames.append(frame)
        return frames

    def _get_const_or_var_ssa(
        self,
        value,
        attr: type[PulseNumericTypedAttr] | None = None,
    ) -> SSAValue:
        """Resolve a Purr value to an SSA value, materialising and a constant if needed.

        If *value* is a :class:`Variable`, the corresponding SSA value
        is fetched from the environment-variable map.  Otherwise a
        fresh constant op is emitted into the current block:

        * when *attr* is one of :class:`TimeAttr`, :class:`FrequencyAttr`,
          :class:`PhaseAttr` or :class:`AmplitudeAttr`, a Pulse-dialect
          :class:`ConstantOp` carrying that typed attribute is used;
        * when *attr* is ``None``, an ``arith.constant`` is emitted of
          type ``f64`` for ``float`` values or ``i32`` for ``int``
          values.

        :param value: The literal Python value or :class:`Variable` to
            translate.
        :param attr: Optional Pulse-dialect attribute class controlling
            the type of the materialised constant.
        :returns: The :class:`SSAValue` representing *value*.
        :raises KeyError: If *value* is a :class:`Variable` not
            currently bound in the environment.
        :raises ValueError: If *value* has an unsupported type or
            *attr* is not one of the supported attribute classes.
        """
        if isinstance(value, Variable):
            return self._current_environment_variables[value.name]
        if attr is None:
            if isinstance(value, float):
                value_op = ArithConstantOp(FloatAttr(value, f64), f64)
            elif isinstance(value, int):
                value_op = ArithConstantOp(IntegerAttr(value, i32), i32)
            else:
                raise ValueError(f"Unsupported value {value} type {type(value)}")
        elif attr in [TimeAttr, FrequencyAttr, PhaseAttr, AmplitudeAttr]:
            value_op = ConstantOp(attr(value))
        else:
            raise ValueError(f"Unsupported type {attr}")

        self._add_ops([value_op])
        return value_op.result

    @singledispatchmethod
    def translate(self, instruction: QuantumInstruction) -> None:
        """Dispatch *instruction* to the appropriate Pulse-dialect emitter.

        Uses :func:`functools.singledispatchmethod` to select a
        handler based on the concrete :class:`QuantumInstruction`
        subclass.

        :param instruction: The Purr instruction to translate.
        :raises ValueError: If no handler is registered for the
            instruction type.
        """
        raise ValueError(f"{instruction} not a supported instruction.")

    @translate.register(PhaseSet)
    @translate.register(PhaseReset)
    def _(self, value):
        """Emit ``PhaseSetOp`` for each target frame.

        If *value* is a :class:`PhaseReset` the phase is set to zero;
        otherwise the phase from the instruction is used.

        :param value: A phase-set or phase-reset instruction.
        """
        frames = self.get_frames(value.quantum_targets)
        phase = 0 if isinstance(value, PhaseReset) else value.phase
        self._add_ops(
            [
                PhaseSetOp(frame, self._get_const_or_var_ssa(phase, PhaseAttr))
                for frame in frames
            ]
        )

    @translate.register
    def _(self, value: PhaseShift):
        """Emit ``PhaseShiftOp`` for each target frame.

        :param value: A phase-shift instruction carrying the relative phase offset.
        """
        frames = self.get_frames(value.quantum_targets)
        self._add_ops(
            [
                PhaseShiftOp(frame, self._get_const_or_var_ssa(value.phase, PhaseAttr))
                for frame in frames
            ]
        )

    @translate.register
    def _(self, value: Delay):
        """Emit ``WaitOp`` for each target frame.

        :param value: A delay instruction specifying the wait duration and target channels.
        """
        frames = self.get_frames(value.quantum_targets)
        self._add_ops(
            [
                WaitOp(frame, self._get_const_or_var_ssa(value.time, TimeAttr))
                for frame in frames
            ]
        )

    @translate.register
    def _(self, value: Synchronize):
        """Emit a ``SynchronizeOp`` across all target frames.

        A synchronisation barrier is only emitted when more than one frame is involved.

        :param value: A synchronise instruction listing the channels to align.
        """
        frames = self.get_frames(value.quantum_targets)
        if len(frames) > 1:
            sync_op = SynchronizeOp(*frames)
            self._add_ops([sync_op])

    def _waveform_to_op(self, purr_waveform: Pulse) -> IRDLOperation:
        """Materialise a Purr :class:`Pulse` as a Pulse-dialect waveform op.

        Dispatches on the pulse's :class:`PulseShapeType` and emits the
        constants required for each shape's operands via
        :meth:`_get_const_or_var_ssa` (so :class:`Variable` parameters
        are resolved against the environment-variable map and literals
        become ``arith.constant`` / Pulse-dialect ``ConstantOp`` values
        in the current block).  The returned waveform op itself is *not*
        added to the block; callers are responsible for inserting it
        alongside the consuming op (typically a :class:`PulseOp` or an
        :class:`AcquireOp` with a filter).

        :param purr_waveform: The Purr pulse instruction whose shape
            and parameters describe the waveform to build.
        :returns: The newly constructed Pulse-dialect waveform
            operation (e.g. :class:`SquareWaveformOp`,
            :class:`GaussianWaveformOp`, ...).
        :raises ValueError: If ``purr_waveform.shape`` is not a
            supported :class:`PulseShapeType`.
        """
        width_ssa = self._get_const_or_var_ssa(purr_waveform.width)
        amp_ssa = self._get_const_or_var_ssa(purr_waveform.amp)
        match purr_waveform.shape:
            case PulseShapeType.SQUARE:
                wave_op = SquareWaveformOp(width_ssa, amp_ssa)
            case PulseShapeType.GAUSSIAN:
                std_ssa = self._get_const_or_var_ssa(purr_waveform.rise)
                wave_op = GaussianWaveformOp(width_ssa, amp_ssa, std_ssa)
            case PulseShapeType.SOFT_SQUARE:
                rise_ssa = self._get_const_or_var_ssa(purr_waveform.rise)
                wave_op = SoftSquareWaveformOp(width_ssa, amp_ssa, rise_ssa)
            case PulseShapeType.SOFTER_SQUARE:
                std_ssa = self._get_const_or_var_ssa(purr_waveform.std_dev)
                rise_ssa = self._get_const_or_var_ssa(purr_waveform.rise)
                wave_op = SofterSquareWaveformOp(width_ssa, amp_ssa, std_ssa, rise_ssa)
            case PulseShapeType.EXTRA_SOFT_SQUARE:
                std_ssa = self._get_const_or_var_ssa(purr_waveform.std_dev)
                rise_ssa = self._get_const_or_var_ssa(purr_waveform.rise)
                wave_op = ExtraSoftSquareWaveformOp(width_ssa, amp_ssa, std_ssa, rise_ssa)
            case PulseShapeType.GAUSSIAN_SQUARE:
                std_ssa = self._get_const_or_var_ssa(purr_waveform.std_dev)
                sq_ssa = self._get_const_or_var_ssa(purr_waveform.square_width)
                wave_op = GaussianSquareWaveformOp(
                    width_ssa,
                    amp_ssa,
                    std_ssa,
                    sq_ssa,
                    BoolAttr(purr_waveform.zero_at_edges, value_type=1),
                )
            case PulseShapeType.SOFTER_GAUSSIAN:
                std_op = self._get_const_or_var_ssa(purr_waveform.rise)
                wave_op = SofterGaussianWaveformOp(width_ssa, amp_ssa, std_op)
            case PulseShapeType.BLACKMAN:
                wave_op = BlackmanWaveformOp(width_ssa, amp_ssa)
            case PulseShapeType.SETUP_HOLD:
                amp_setup_ssa = self._get_const_or_var_ssa(purr_waveform.amp_setup)
                rise_ssa = self._get_const_or_var_ssa(purr_waveform.rise)
                wave_op = SetupHoldWaveformOp(width_ssa, amp_ssa, amp_setup_ssa, rise_ssa)
            case PulseShapeType.ROUNDED_SQUARE:
                rise_ssa = self._get_const_or_var_ssa(purr_waveform.rise)
                std_ssa = self._get_const_or_var_ssa(purr_waveform.std_dev)
                wave_op = RoundedSquareWaveformOp(width_ssa, amp_ssa, rise_ssa, std_ssa)
            case PulseShapeType.GAUSSIAN_DRAG:
                std_ssa = self._get_const_or_var_ssa(purr_waveform.std_dev)
                beta_ssa = self._get_const_or_var_ssa(purr_waveform.beta)
                wave_op = DragGaussianWaveformOp(
                    width_ssa,
                    amp_ssa,
                    std_ssa,
                    beta_ssa,
                    BoolAttr(purr_waveform.zero_at_edges, value_type=1),
                )
            case PulseShapeType.GAUSSIAN_ZERO_EDGE:
                std_ssa = self._get_const_or_var_ssa(purr_waveform.std_dev)
                wave_op = GaussianZeroEdgeWaveformOp(
                    width_ssa,
                    amp_ssa,
                    std_ssa,
                    BoolAttr(purr_waveform.zero_at_edges, value_type=1),
                )
            case PulseShapeType.SECH:
                std_ssa = self._get_const_or_var_ssa(purr_waveform.std_dev)
                wave_op = SechWaveformOp(width_ssa, amp_ssa, std_ssa)
            case PulseShapeType.COS:
                freq_ssa = self._get_const_or_var_ssa(purr_waveform.frequency)
                phase_ssa = self._get_const_or_var_ssa(purr_waveform.internal_phase)
                wave_op = CosWaveformOp(width_ssa, amp_ssa, freq_ssa, phase_ssa)
            case PulseShapeType.SIN:
                freq_ssa = self._get_const_or_var_ssa(purr_waveform.frequency)
                phase_ssa = self._get_const_or_var_ssa(purr_waveform.internal_phase)
                wave_op = SinWaveformOp(width_ssa, amp_ssa, freq_ssa, phase_ssa)
            case _:
                raise ValueError(f"Unsupported shape, {purr_waveform.shape}.")
        return wave_op

    @translate.register
    def _(self, value: Pulse):
        """Emit a waveform op and a ``PulseOp`` for the target frame.

        Delegates waveform construction to :meth:`_waveform_to_op` and
        plays the result on the first target frame.

        :param value: A pulse instruction carrying shape, width,
            amplitude, and shape-specific parameters.
        :raises ValueError: If the pulse shape is not supported by
            :meth:`_waveform_to_op`.
        """
        frame = self.get_frames(value.quantum_targets)[0]
        wave_op = self._waveform_to_op(value)
        pulse_op = PulseOp(frame, wave_op)
        self._add_ops([wave_op, pulse_op])

    @translate.register
    def _(self, value: CustomPulse):
        """Translate a custom-pulse instruction.

        :param value: A custom-pulse instruction.
        :raises ValueError: Always; sample-array pulses are not yet representable in the
            Pulse dialect.
        """
        raise ValueError("Not yet supported by IR")

    @translate.register
    def _(self, value: Acquire):
        """Emit an ``AcquireOp`` on the target frame.

        When the acquire instruction carries a ``filter`` as a :class:`CustomPulse`, the
        filter's samples are materialised as a ``WeightsAttr`` on the acquire op.

        :param value: An acquire instruction specifying the
            measurement duration and optional filter waveform.
        """
        frame = self.get_frames(value.quantum_targets)[0]

        weights = None
        if value.filter is not None:
            if not isinstance(value.filter, CustomPulse):
                raise ValueError(
                    f"Acquire filter must be a CustomPulse, got {type(value.filter).__name__}"
                )
            weights = WeightsAttr(value.filter.samples)

        acquire_op = AcquireOp(
            frame,
            self._get_const_or_var_ssa(value.duration, TimeAttr),
            weights=weights,
        )

        ops = [acquire_op]

        acquire_result = acquire_op.acquisition_result
        if value.mode == AcquireMode.INTEGRATOR:
            integration_op = IntegrateOp(acquire_result)
            ops.append(integration_op)
            acquire_result = integration_op.result
        elif value.mode == AcquireMode.SCOPE:
            # TODO: COMPILER-1333
            raise NotImplementedError(
                "Scope mode is not yet supported by the PurrImporter."
            )
        self._add_ops(ops)

        acquire_key = f"acquire_{value.output_variable}"
        self._current_environment_variables.forceput(acquire_key, acquire_result)

    @translate.register
    def _(self, value: PostProcessing):
        """Handle a post-processing instruction.

        :param value: A post-processing instruction.
        """

        acquire_key = f"acquire_{value.quantum_targets[0].output_variable}"
        prev_acquire_result = self._current_environment_variables.get(acquire_key)
        if prev_acquire_result is None:
            raise ValueError(
                f"Post-processing instruction references acquisition '{acquire_key}' "
                "but no prior acquisition found in the environment."
            )

        if not isinstance(prev_acquire_result.type, IQResultType):
            raise ValueError(
                f"Post-processing expects an IQResultType (from integrated acquisition), "
                f"but got {prev_acquire_result.type}. Ensure the acquire has mode INTEGRATOR."
            )

        match value.process:
            case PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL:
                # This operation practically takes Re(az + b), which can be represented as
                # an affine transformation of 0.5 * (az + a*z*) + Re(b).

                args = value.args
                if len(args) != 2:
                    raise ValueError(
                        f"LINEAR_MAP_COMPLEX_TO_REAL post-processing expects exactly 2 "
                        f"arguments, got {len(args)}."
                    )
                linear_coeff = 0.5 * args[0]
                conj_coeff = 0.5 * np.conj(args[0])
                offset = np.real(args[1])

                affine_attr = EqualiseAttr(
                    linear_coefficient=linear_coeff,
                    conjugate_coefficient=conj_coeff,
                    translation=offset,
                )
                pp_op = EqualiseOp(prev_acquire_result, affine_attr)
            case PostProcessType.DISCRIMINATE:
                if len(value.args) != 1:
                    raise ValueError(
                        "DISCRIMINATE post-processing expects exactly 1 argument, "
                        f"got {len(value.args)}."
                    )

                threshold = value.args[0]
                # Legacy PuRR software post-processing emits {-1, 1} labels, while the
                # pulse dialect currently represents threshold discrimination with
                # RealThresholdPolicyAttr and StateKeyType(0, 1). Downstream state-mapping
                # operations can remap these labels when needed.
                policy_attr = RealThresholdPolicyAttr(threshold=threshold)
                pp_op = DiscriminateOp(prev_acquire_result, policy_attr)
            case _:
                raise ValueError(f"Unsupported post-processing type {value.process}.")

        self._current_environment_variables.forceput(acquire_key, pp_op.result)
        self._add_ops([pp_op])

    @translate.register
    def _(self, value: Sweep):
        """Lower a sweep instruction to an ``scf.for`` over swept arrays.

        Not yet implemented.
        """
        # Init array
        raise ValueError("Not yet implemented.")

    @translate.register
    def _(self, value: EndSweep):
        raise ValueError("Not yet implemented.")

    @translate.register
    def _(self, value: Repeat):
        """Open an ``scf.for`` loop for the given repeat count.

        :param value: A repeat instruction specifying the number of iterations.
        """
        self.enter_for_loop(0, value.repeat_count, 1)

    @translate.register
    def _(self, value: EndRepeat):
        """Close the innermost ``scf.for`` loop opened by a :class:`Repeat`.

        :param value: An end-repeat instruction.
        """
        self.exit_for_loop()

    @translate.register
    def _(self, value: DeviceUpdate):
        """Apply a device-update by rebinding the affected frame.

        Currently only :class:`PulseChannel` targets with a
        ``frequency`` attribute update are supported.  The update
        materialises the new frequency value (constant or swept
        variable) and creates a fresh frame in the current block,
        replacing the existing binding for the channel's frame key in
        the environment-variable map so subsequent translations pick
        up the new SSA value.

        :param value: A device-update instruction carrying the target
            device, attribute name, and new value (literal or
            :class:`Variable`).
        :raises ValueError: If the target device is not a
            :class:`PulseChannel`, or the attribute being updated is
            not ``"frequency"``.
        """
        purr_device = value.target
        purr_device_attribute = value.attribute
        purr_var = value.value

        if isinstance(purr_device, PulseChannel):
            if purr_device_attribute == "frequency":
                freq_value_op = self._get_const_or_var_ssa(purr_var, FrequencyAttr)
                key = self.get_frame_key(purr_device)
                frame = self.create_frame(purr_device, freq_op=freq_value_op)
                self._current_environment_variables.forceput(key, frame)
            else:
                raise ValueError(
                    f"Unsupported pulse channel attribute {purr_device_attribute}"
                )
        else:
            raise ValueError(f"Unsupported device {purr_device}")
